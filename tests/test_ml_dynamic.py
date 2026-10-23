"""Tests for ML scheduling and actual runtime separation."""

from types import SimpleNamespace

import pytest

from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.ml_dynamic import MLDynamicSimulator
from hybridsched.ml_predictor import job_features
from hybridsched.resources import Resources


def cluster():
    return ClusterSpec(
        "local",
        "ON_PREMISE",
        Resources(4, 2, 16),
    )


def job(jid, arrival, requested):
    return Job(
        jid,
        arrival,
        requested,
        Resources(2, 2, 8),
    )


class FakePredictor:
    def predict_many(self, jobs):
        return {
            j.job_id: SimpleNamespace(
                point_s=30,
                p50_s=40,
                p90_s=80,
            )
            for j in jobs
        }


def test_features_order():
    j = job("A", 10, 100)
    values = job_features(j)

    assert values[0] == 100
    assert values[1] == 2
    assert values[2] == 2
    assert values[5] == 10


def test_actual_runtime_controls_completion():
    j = job("A", 0, 100)

    sim = MLDynamicSimulator(
        [j],
        [cluster()],
        {"A": 250},
        FakePredictor(),
        "p50",
    )

    result = sim.run()[0]

    assert result.start_s == 0
    assert result.finish_s == 250


def test_predicted_duration_controls_priority():
    jobs = [
        job("A", 0, 200),
        job("B", 0, 100),
    ]

    class Predictor(FakePredictor):
        def predict_many(self, jobs):
            result = super().predict_many(jobs)
            for j in jobs:
                result[j.job_id].p50_s = (
                    10 if j.job_id == "A" else 300
                )
            return result

    sim = MLDynamicSimulator(
        jobs,
        [cluster()],
        {"A": 200, "B": 100},
        Predictor(),
        "p50",
    )

    results = {
        r.job_id: r for r in sim.run()
    }

    assert results["A"].start_s == 0
    assert results["B"].start_s == 200


def test_missing_actual_runtime_rejected():
    with pytest.raises(ValueError):
        MLDynamicSimulator(
            [job("A", 0, 100)],
            [cluster()],
            {},
        )


def test_unknown_prediction_mode_rejected():
    with pytest.raises(ValueError):
        MLDynamicSimulator(
            [job("A", 0, 100)],
            [cluster()],
            {"A": 100},
            mode="unknown",
        )

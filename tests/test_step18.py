"""Tests for calibration, ablation and uncertainty policies."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments.conditional_quantile_calibration import (
    duration_group,
    gpu_group,
)
from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.ml_ablation import MLAblationSimulator
from hybridsched.resources import Resources
from hybridsched.uncertainty_aware import UncertaintyAwareSimulator


def cluster():
    return ClusterSpec(
        "local", "ON_PREMISE",
        Resources(4, 2, 16),
    )


def job(jid, arrival, requested, deadline=None):
    return Job(
        jid, arrival, requested,
        Resources(2, 2, 8),
        deadline_s=deadline,
    )


class FakePredictor:
    def predict_many(self, jobs):
        return {
            j.job_id: SimpleNamespace(
                point_s=50,
                p10_s=20,
                p50_s=40,
                p90_s=100,
            )
            for j in jobs
        }


def test_duration_groups():
    assert duration_group(100) == "SHORT"
    assert duration_group(600) == "MEDIUM"
    assert duration_group(1200) == "LONG"


def test_gpu_groups():
    assert gpu_group(1) == "GPU_1"
    assert gpu_group(2) == "GPU_2"
    assert gpu_group(4) == "GPU_3_PLUS"


def test_ablation_actual_runtime():
    j = job("A", 0, 200)

    sim = MLAblationSimulator(
        [j], [cluster()],
        {"A": 300},
        FakePredictor(),
        ranking_mode="p50",
        placement_mode="p90",
    )

    assert sim.run()[0].finish_s == 300


def test_ablation_invalid_mode():
    with pytest.raises(ValueError):
        MLAblationSimulator(
            [job("A", 0, 100)],
            [cluster()],
            {"A": 100},
            FakePredictor(),
            ranking_mode="invalid",
        )


@pytest.mark.parametrize(
    "deadline,expected",
    [(30, 100), (200, 70), (500, 40)],
)
def test_risk_duration_levels(deadline, expected):
    j = job("A", 0, 100, deadline)

    sim = UncertaintyAwareSimulator(
        [j], [cluster()],
        {"A": 150},
        FakePredictor(),
    )

    assert sim._risk_duration(j, 0) == expected


def test_uncertainty_executes_actual_runtime():
    j = job("A", 0, 100, 150)

    sim = UncertaintyAwareSimulator(
        [j], [cluster()],
        {"A": 250},
        FakePredictor(),
    )

    result = sim.run()[0]

    assert result.finish_s == 250
    assert result.tardiness_s == 100


def test_uncertainty_never_starts_before_arrival():
    jobs = [
        job("A", 20, 100, 200),
        job("B", 30, 100, 220),
    ]

    sim = UncertaintyAwareSimulator(
        jobs, [cluster()],
        {"A": 120, "B": 130},
        FakePredictor(),
    )

    results = sim.run()
    assert all(r.start_s >= r.arrival_s for r in results)

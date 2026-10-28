"""STEP 22 multiobjective scheduling tests."""

from types import SimpleNamespace

import pytest

from experiments.benchmark_multiobjective import dominates
from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.multiobjective_deadline import (
    MultiobjectiveDeadlineSimulator,
)
from hybridsched.resources import Resources


def cluster():
    return ClusterSpec(
        "local", "ON_PREMISE",
        Resources(4, 2, 16),
    )


def job(jid, arrival=0, deadline=150):
    return Job(
        job_id=jid,
        arrival_s=arrival,
        requested_walltime_s=100,
        resources=Resources(2, 2, 8),
        deadline_s=deadline,
    )


class Predictor:
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


def simulator(jobs, actual):
    return MultiobjectiveDeadlineSimulator(
        jobs, [cluster()], actual, Predictor()
    )


def test_actual_runtime_controls_finish():
    result = simulator(
        [job("A")], {"A": 250}
    ).run()[0]
    assert result.finish_s == 250
    assert result.tardiness_s == 100


def test_respects_arrival():
    result = simulator(
        [job("A", arrival=30)],
        {"A": 100},
    ).run()[0]
    assert result.start_s >= 30


def test_score_is_finite():
    s = simulator([job("A")], {"A": 100})
    value = s._candidate_score(job("A"), cluster(), 0)
    assert isinstance(value, (int, float))
    assert float("-inf") < value < float("inf")


def test_invalid_weights():
    with pytest.raises(ValueError):
        MultiobjectiveDeadlineSimulator(
            [job("A")], [cluster()],
            {"A": 100}, Predictor(),
            cost_weight=-0.1,
        )


def test_pareto_dominance():
    a = {
        "late_jobs": 10,
        "tardiness_s": 100,
        "cost_units": 1000,
    }
    b = {
        "late_jobs": 12,
        "tardiness_s": 120,
        "cost_units": 1000,
    }
    assert dominates(a, b)
    assert not dominates(b, a)

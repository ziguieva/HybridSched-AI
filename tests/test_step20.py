"""Decision-focused scheduling and shift regression tests."""

from types import SimpleNamespace

import pytest

from experiments.analyze_decision_shift import bootstrap_ci
from experiments.decision_shift_benchmark import transform_workload
from hybridsched.decision_focused import DecisionFocusedSimulator
from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.resources import Resources


def cluster():
    return ClusterSpec(
        "local",
        "ON_PREMISE",
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


def test_actual_runtime_not_prediction():
    sim = DecisionFocusedSimulator(
        [job("A")],
        [cluster()],
        {"A": 250},
        FakePredictor(),
    )

    result = sim.run()[0]
    assert result.finish_s == 250
    assert result.tardiness_s == 100


def test_arrival_constraint():
    sim = DecisionFocusedSimulator(
        [job("A", arrival=30, deadline=200)],
        [cluster()],
        {"A": 100},
        FakePredictor(),
    )

    assert sim.run()[0].start_s >= 30


def test_invalid_weights():
    with pytest.raises(ValueError):
        DecisionFocusedSimulator(
            [job("A")],
            [cluster()],
            {"A": 100},
            FakePredictor(),
            p90_late_weight=-1,
        )


def test_shift_is_reproducible():
    jobs = [job("A", 0), job("B", 100)]
    actual = {"A": 100, "B": 200}

    first = transform_workload(
        jobs, actual, "COMBINED", 42
    )
    second = transform_workload(
        jobs, actual, "COMBINED", 42
    )

    assert first == second


def test_shift_preserves_job_identity():
    jobs = [job("A", 0), job("B", 100)]
    actual = {"A": 100, "B": 200}

    shifted, changed_actual = transform_workload(
        jobs, actual, "GPU_HEAVY", 42
    )

    assert {j.job_id for j in shifted} == {"A", "B"}
    assert set(changed_actual) == {"A", "B"}


def test_bootstrap_constant_difference():
    avg, low, high = bootstrap_ci(
        [-10.0] * 10
    )

    assert avg == -10
    assert low == -10
    assert high == -10


def test_invalid_shift_rejected():
    with pytest.raises(ValueError):
        transform_workload(
            [job("A")],
            {"A": 100},
            "INVALID",
            42,
        )

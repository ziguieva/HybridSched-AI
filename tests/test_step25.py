"""STEP 25 — Dynamic ML scheduler correctness audit."""

from types import SimpleNamespace

import pytest

from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.ml_dynamic import MLDynamicSimulator
from hybridsched.ml_deadline import MLDeadlineSimulator
from hybridsched.multiobjective_deadline import (
    MultiobjectiveDeadlineSimulator,
)
from hybridsched.resources import Resources


class FixedPredictor:
    def predict_many(self, jobs):
        return {
            job.job_id: SimpleNamespace(
                point_s=40.0,
                p10_s=20.0,
                p50_s=40.0,
                p90_s=90.0,
            )
            for job in jobs
        }


def make_cluster():
    return ClusterSpec(
        name="audit-cluster",
        provider="TEST",
        capacity=Resources(4, 2, 16),
    )


def make_jobs():
    return [
        Job(
            job_id=f"A{i}",
            arrival_s=arrival,
            requested_walltime_s=100,
            resources=Resources(2, 2, 8),
            deadline_s=arrival + 180,
        )
        for i, arrival in enumerate(
            [0, 0, 30, 30, 100, 100]
        )
    ]


def actual_durations():
    return {
        "A0": 90,
        "A1": 110,
        "A2": 75,
        "A3": 130,
        "A4": 65,
        "A5": 95,
    }


def build_simulator(scheduler, jobs=None, actual=None):
    jobs = make_jobs() if jobs is None else jobs
    actual = actual_durations() if actual is None else actual

    if scheduler == "ML_DYNAMIC":
        return MLDynamicSimulator(
            jobs, [make_cluster()],
            actual_runtimes=actual,
            predictor=FixedPredictor(),
            mode="p90",
        )

    if scheduler == "ML_DEADLINE":
        return MLDeadlineSimulator(
            jobs, [make_cluster()],
            actual_runtimes=actual,
            predictor=FixedPredictor(),
            mode="p90",
        )

    if scheduler == "MULTI":
        return MultiobjectiveDeadlineSimulator(
            jobs, [make_cluster()],
            actual, FixedPredictor(),
        )

    raise ValueError(scheduler)


def verify_schedule(results, jobs, actual, clusters):
    by_id = {job.job_id: job for job in jobs}
    capacities = {c.name: c.capacity for c in clusters}

    assert len(results) == len(jobs)
    assert len({r.job_id for r in results}) == len(jobs)

    for result in results:
        job = by_id[result.job_id]
        cluster = next(
            c for c in clusters if c.name == result.cluster
        )

        assert result.start_s >= job.arrival_s
        assert result.finish_s > result.start_s

        assert (
            result.waiting_s
            == result.start_s - result.arrival_s
        )

        assert (
            result.finish_s - result.start_s
            == cluster.execution_time(
                __import__("math").ceil(actual[result.job_id])
            )
        )

        assert result.tardiness_s == max(
            0, result.finish_s - job.deadline_s
        )

    # Half-open execution intervals [start, finish).
    # At each event, simultaneous completions release
    # resources before new starts.
    for cluster_name, capacity in capacities.items():
        relevant = [
            r for r in results
            if r.cluster == cluster_name
        ]

        times = sorted({
            t for r in relevant
            for t in (r.start_s, r.finish_s)
        })

        for t in times:
            active = [
                by_id[r.job_id].resources
                for r in relevant
                if r.start_s <= t < r.finish_s
            ]

            assert (
                sum(x.cpu_cores for x in active)
                <= capacity.cpu_cores
            )
            assert (
                sum(x.gpus for x in active)
                <= capacity.gpus
            )
            assert (
                sum(x.memory_gib for x in active)
                <= capacity.memory_gib
            )


@pytest.mark.parametrize(
    "scheduler",
    ["ML_DYNAMIC", "ML_DEADLINE", "MULTI"],
)
def test_ml_scheduler_capacity_and_timing(scheduler):
    sim = build_simulator(scheduler)
    jobs = make_jobs()
    actual = actual_durations()

    results = sim.run()
    verify_schedule(
        results, jobs, actual, sim.clusters
    )

    assert all(
        sim.used[c.name] == Resources()
        for c in sim.clusters
    )


def test_simultaneous_completion_and_arrival():
    jobs = [
        Job(
            "A", 0, 100,
            Resources(2, 2, 8),
            deadline_s=200,
        ),
        Job(
            "B", 100, 100,
            Resources(2, 2, 8),
            deadline_s=300,
        ),
    ]

    sim = build_simulator(
        "ML_DYNAMIC",
        jobs,
        {"A": 100, "B": 100},
    )

    results = {r.job_id: r for r in sim.run()}

    assert results["A"].finish_s == 100
    assert results["B"].start_s == 100


def test_actual_runtime_not_predicted_runtime():
    jobs = [
        Job(
            "A", 0, 100,
            Resources(2, 2, 8),
            deadline_s=150,
        )
    ]

    sim = build_simulator(
        "ML_DYNAMIC",
        jobs,
        {"A": 250},
    )

    result = sim.run()[0]

    assert result.finish_s == 250
    assert result.tardiness_s == 100


def test_ml_results_are_deterministic():
    first = build_simulator("MULTI").run()
    second = build_simulator("MULTI").run()

    assert first == second


def test_cluster_speed_applies_to_actual_duration():
    cluster = ClusterSpec(
        name="fast",
        provider="TEST",
        capacity=Resources(4, 2, 16),
        speed_num=2,
        speed_den=1,
        startup_s=20,
    )

    job = Job(
        "A", 0, 100,
        Resources(2, 2, 8),
        deadline_s=200,
    )

    sim = MLDynamicSimulator(
        [job], [cluster],
        actual_runtimes={"A": 101},
        predictor=FixedPredictor(),
        mode="p90",
    )

    result = sim.run()[0]

    assert result.finish_s == 71
    assert result.tardiness_s == 0

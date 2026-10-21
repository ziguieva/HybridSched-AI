"""Regression tests for improved rolling-horizon scheduling."""

from hybridsched.dynamic import DynamicHybridSimulator
from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.resources import Resources


def make_cluster():
    return ClusterSpec(
        "local",
        "ON_PREMISE",
        Resources(4, 2, 16),
    )


def make_jobs():
    return [
        Job("A", 0, 100, Resources(2, 2, 8)),
        Job("B", 10, 50, Resources(2, 2, 8)),
    ]


def test_running_job_is_not_preempted():
    simulator = DynamicHybridSimulator(
        make_jobs(),
        [make_cluster()],
        policy="rolling",
    )

    results = {
        result.job_id: result
        for result in simulator.run()
    }

    assert results["A"].start_s == 0
    assert results["A"].finish_s == 100
    assert results["B"].start_s == 100
    assert results["B"].finish_s == 150


def test_future_resource_release():
    cluster = make_cluster()
    job_a, job_b = make_jobs()

    simulator = DynamicHybridSimulator(
        [job_a, job_b],
        [cluster],
        policy="rolling",
    )

    simulator._start(job_a, cluster, 0)
    simulator.waiting.append(job_b)

    status, plan = simulator._rolling_plan(10)

    assert status in ("OPTIMAL", "FEASIBLE")
    assert len(plan) == 1
    assert plan[0][0] == 100
    assert plan[0][1].job_id == "B"


def test_delayed_start_is_respected():
    cluster = make_cluster()
    job = make_jobs()[0]

    simulator = DynamicHybridSimulator(
        [job],
        [cluster],
        policy="rolling",
    )

    simulator.waiting = [job]

    simulator._rolling_plan = lambda now: (
        "OPTIMAL",
        [(50, job, cluster)],
    )

    next_wakeup = simulator._rolling_dispatch(0)

    assert next_wakeup == 50
    assert simulator.results == []
    assert simulator.waiting == [job]

"""Bounded rolling-horizon regression tests."""

import pytest

from hybridsched.dynamic import DynamicHybridSimulator
from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.resources import Resources


def cluster():
    return ClusterSpec(
        "local", "ON_PREMISE",
        Resources(4, 2, 16),
    )


def workload(count):
    return [
        Job(
            f"J{i:03d}",
            i,
            100,
            Resources(2, 2, 8),
        )
        for i in range(count)
    ]


def test_planning_window_respected():
    simulator = DynamicHybridSimulator(
        workload(20),
        [cluster()],
        policy="rolling",
    )

    simulator.waiting = list(simulator.jobs)
    status, plan = simulator._rolling_plan(20)

    assert status in ("OPTIMAL", "FEASIBLE")
    assert len(plan) <= simulator.planning_window_jobs
    assert len(plan) == 12


def test_all_jobs_eventually_complete():
    simulator = DynamicHybridSimulator(
        workload(20),
        [cluster()],
        policy="rolling",
    )

    results = simulator.run()

    assert len(results) == 20
    assert len({r.job_id for r in results}) == 20


def test_solver_profiling_recorded():
    simulator = DynamicHybridSimulator(
        workload(3),
        [cluster()],
        policy="rolling",
    )

    simulator.run()

    assert simulator.solver_calls >= 1
    assert simulator.solver_wall_s >= 0
    assert simulator.solver_failures >= 0


def test_skip_when_resources_unavailable():
    c = cluster()
    a, b = workload(2)

    simulator = DynamicHybridSimulator(
        [a, b],
        [c],
        policy="rolling",
    )

    simulator._start(a, c, 0)
    simulator.waiting.append(b)

    wakeup = simulator._rolling_dispatch(1)

    assert wakeup is None
    assert simulator.skipped_plans == 1
    assert simulator.solver_calls == 0
    assert len(simulator.waiting) == 1

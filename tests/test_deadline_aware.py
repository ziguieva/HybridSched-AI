"""Regression tests for deadline-aware policies."""

import pytest

from hybridsched.deadline_aware import DeadlineAwareSimulator
from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.resources import Resources


def cluster():
    return ClusterSpec(
        "local", "ON_PREMISE",
        Resources(4, 2, 16),
    )


def job(jid, duration, deadline):
    return Job(
        jid, 0, duration,
        Resources(2, 2, 8),
        deadline_s=deadline,
    )


@pytest.mark.parametrize(
    "policy",
    ["edf", "lst", "resource"],
)
def test_all_policies_complete(policy):
    jobs = [
        job("A", 100, 300),
        job("B", 50, 100),
    ]

    sim = DeadlineAwareSimulator(
        jobs, [cluster()],
        priority_policy=policy,
    )

    results = sim.run()
    assert len(results) == 2
    assert sim.solver_calls == 0


def test_edf_prioritizes_earliest_deadline():
    jobs = [
        job("A", 50, 500),
        job("B", 100, 150),
    ]
    result = DeadlineAwareSimulator(
        jobs, [cluster()],
        priority_policy="edf",
    ).run()

    by_id = {r.job_id: r for r in result}
    assert by_id["B"].start_s == 0
    assert by_id["A"].start_s == 100


def test_lst_accounts_for_duration():
    jobs = [
        job("A", 100, 180),
        job("B", 20, 140),
    ]
    result = DeadlineAwareSimulator(
        jobs, [cluster()],
        priority_policy="lst",
    ).run()

    by_id = {r.job_id: r for r in result}
    assert by_id["A"].start_s == 0


def test_resource_availability_after_completion():
    c = cluster()
    a = job("A", 100, 400)
    b = job("B", 50, 250)

    sim = DeadlineAwareSimulator(
        [a, b], [c],
        priority_policy="resource",
    )

    sim._start(a, c, 0)

    assert sim._earliest_cluster_start(b, c, 10) == 100
    assert sim._projected_finish(b, c, 10) == 150


def test_invalid_policy():
    with pytest.raises(ValueError, match="Unknown"):
        DeadlineAwareSimulator(
            [job("A", 50, 100)],
            [cluster()],
            priority_policy="unknown",
        )


def test_no_resource_overcommit():
    jobs = [
        Job(
            f"J{i}",
            i * 10,
            100,
            Resources(2, 2, 8),
            deadline_s=200 + i * 20,
        )
        for i in range(12)
    ]

    result = DeadlineAwareSimulator(
        jobs, [cluster()],
        priority_policy="resource",
    ).run()

    for t in sorted({
        event
        for r in result
        for event in (r.start_s, r.finish_s)
    }):
        running = [
            r for r in result
            if r.start_s <= t < r.finish_s
        ]
        assert len(running) <= 1

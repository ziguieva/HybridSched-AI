import pytest

from experiments.workload_shift import generate_shifted_workload
from hybridsched.deadline_feasibility import analyze_deadline_feasibility
from hybridsched.hybrid_priority import HybridPrioritySimulator
from hybridsched.infrastructure import ClusterSpec, default_infrastructure
from hybridsched.job import Job
from hybridsched.resources import Resources


def cluster():
    return ClusterSpec(
        "local", "ON_PREMISE",
        Resources(4, 2, 16),
    )


@pytest.mark.parametrize(
    "mode",
    ["COUPLED", "INDEPENDENT", "TIGHT"],
)
def test_reproducible_workload(mode):
    a = generate_shifted_workload(20, 42, mode)
    b = generate_shifted_workload(20, 42, mode)
    assert a == b


def test_distribution_preserves_arrivals():
    a = generate_shifted_workload(20, 42, "COUPLED")
    b = generate_shifted_workload(20, 42, "INDEPENDENT")

    assert [
        (j.arrival_s, j.requested_walltime_s)
        for j in a
    ] == [
        (j.arrival_s, j.requested_walltime_s)
        for j in b
    ]


def test_intrinsically_impossible_deadline():
    jobs = [
        Job(
            "A", 0, 100,
            Resources(2, 2, 8),
            deadline_s=50,
        )
    ]

    result = analyze_deadline_feasibility(
        jobs, [cluster()]
    )

    assert result["intrinsically_late"] == 1
    assert result["minimum_tardiness_s"] == 50


def test_joint_capacity_constraint():
    jobs = [
        Job(jid, 0, 100, Resources(2, 2, 8),
            deadline_s=100)
        for jid in ("A", "B")
    ]

    lower = analyze_deadline_feasibility(
        jobs, [cluster()]
    )

    results = HybridPrioritySimulator(
        jobs, [cluster()]
    ).run()

    assert lower["intrinsically_late"] == 0
    assert sum(r.tardiness_s > 0 for r in results) == 1


def test_hybrid_completes_workload():
    jobs = generate_shifted_workload(
        20, 42, "TIGHT"
    )

    results = HybridPrioritySimulator(
        jobs, default_infrastructure()
    ).run()

    assert len(results) == 20


def test_unknown_distribution():
    with pytest.raises(ValueError):
        generate_shifted_workload(
            10, 42, "INVALID"
        )

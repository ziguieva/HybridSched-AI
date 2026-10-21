"""Tests for hybrid cloud/HPC resource optimization."""

import pytest

from hybridsched.hybrid_optimizer import optimize_hybrid
from hybridsched.infrastructure import (
    ClusterSpec,
    default_infrastructure,
)
from hybridsched.job import Job
from hybridsched.resources import Resources


def test_infrastructure_has_three_clusters():
    clusters = default_infrastructure()

    assert len(clusters) == 3
    assert len({c.name for c in clusters}) == 3


def test_cluster_execution_duration():
    cluster = default_infrastructure()[1]

    assert cluster.execution_time(900) == 740


def test_empty_workload():
    result = optimize_hybrid(
        [],
        default_infrastructure(),
    )

    assert result.status == "OPTIMAL"
    assert result.results == []


def test_single_job_is_allocated():
    jobs = [
        Job("A", 0, 100, Resources(2, 1, 4))
    ]

    result = optimize_hybrid(
        jobs,
        default_infrastructure(),
    )

    assert result.status == "OPTIMAL"
    assert len(result.results) == 1
    assert result.results[0].start_s >= 0


def test_no_cluster_available():
    jobs = [
        Job("A", 0, 100, Resources(100, 20, 500))
    ]

    with pytest.raises(ValueError, match="eligible"):
        optimize_hybrid(
            jobs,
            default_infrastructure(),
        )


def test_duplicate_jobs_rejected():
    job = Job(
        "A", 0, 100, Resources(2, 1, 4)
    )

    with pytest.raises(ValueError, match="Duplicate"):
        optimize_hybrid(
            [job, job],
            default_infrastructure(),
        )


def test_capacity_respected():
    clusters = [
        ClusterSpec(
            "local",
            "ON_PREMISE",
            Resources(4, 2, 16),
        )
    ]

    jobs = [
        Job("A", 0, 100, Resources(2, 2, 8)),
        Job("B", 0, 100, Resources(2, 2, 8)),
    ]

    result = optimize_hybrid(
        jobs,
        clusters,
    )

    assert result.status == "OPTIMAL"
    assert result.makespan_s == 200


def test_hybrid_placement_uses_multiple_clusters():
    clusters = [
        ClusterSpec(
            "local",
            "ON_PREMISE",
            Resources(2, 1, 8),
        ),
        ClusterSpec(
            "cloud",
            "GCP_SIMULATED",
            Resources(2, 1, 8),
        ),
    ]

    jobs = [
        Job("A", 0, 100, Resources(2, 1, 4)),
        Job("B", 0, 100, Resources(2, 1, 4)),
    ]

    result = optimize_hybrid(
        jobs,
        clusters,
    )

    assert result.status == "OPTIMAL"
    assert len(result.results) == 2
    assert result.makespan_s == 100
    assert len({r.cluster for r in result.results}) == 2

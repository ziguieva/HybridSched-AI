"""Deadline and budget optimization tests."""

import pytest

from hybridsched.hybrid_optimizer import optimize_hybrid
from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.resources import Resources


def make_clusters():
    return [
        ClusterSpec(
            "local",
            "ON_PREMISE",
            Resources(4, 2, 16),
            gpu_cost_per_s=1,
        ),
        ClusterSpec(
            "cloud",
            "GCP_SIMULATED",
            Resources(4, 2, 16),
            gpu_cost_per_s=5,
        ),
    ]


def make_jobs():
    return [
        Job(
            "A", 0, 100,
            Resources(2, 2, 8),
            deadline_s=150,
        ),
        Job(
            "B", 0, 100,
            Resources(2, 2, 8),
            deadline_s=150,
        ),
    ]


def test_soft_deadline_calculation():
    result = optimize_hybrid(
        make_jobs(),
        make_clusters()[:1],
        deadline_weight=10,
    )

    assert result.status == "OPTIMAL"
    assert result.deadline_violations == 1
    assert result.total_tardiness_s == 50


def test_hard_deadline_with_cloud():
    result = optimize_hybrid(
        make_jobs(),
        make_clusters(),
        hard_deadlines=True,
    )

    assert result.status == "OPTIMAL"
    assert result.deadline_violations == 0
    assert len({
        r.cluster for r in result.results
    }) == 2


def test_infeasible_hard_deadlines():
    result = optimize_hybrid(
        make_jobs(),
        make_clusters()[:1],
        hard_deadlines=True,
    )

    assert result.status == "INFEASIBLE"
    assert result.results == []


def test_budget_constraint():
    result = optimize_hybrid(
        make_jobs(),
        make_clusters(),
        max_budget_units=400,
    )

    assert result.status == "OPTIMAL"
    assert result.total_cost_units <= 400
    assert result.cloud_cost_units == 0


def test_cloud_budget_zero():
    result = optimize_hybrid(
        make_jobs(),
        make_clusters(),
        max_cloud_budget_units=0,
    )

    assert result.status == "OPTIMAL"
    assert result.cloud_cost_units == 0


def test_invalid_budget_rejected():
    with pytest.raises(ValueError, match="Budgets"):
        optimize_hybrid(
            make_jobs(),
            make_clusters(),
            max_budget_units=-1,
        )


def test_invalid_deadline_weight_rejected():
    with pytest.raises(ValueError, match="Weights"):
        optimize_hybrid(
            make_jobs(),
            make_clusters(),
            deadline_weight=-1,
        )

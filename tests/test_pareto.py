"""Mathematical tests for Pareto dominance."""

import pytest

from hybridsched.hybrid_optimizer import optimize_hybrid
from hybridsched.infrastructure import default_infrastructure
from hybridsched.job import Job
from hybridsched.resources import Resources


def test_dominance_and_tradeoff():
    from experiments.pareto_frontier import dominates

    a = {"cost": 100, "makespan": 200, "tardiness": 0}
    b = {"cost": 120, "makespan": 220, "tardiness": 10}
    c = {"cost": 80, "makespan": 250, "tardiness": 0}

    assert dominates(a, b)
    assert not dominates(b, a)
    assert not dominates(a, c)
    assert not dominates(c, a)


def test_pareto_filter_removes_dominated():
    from experiments.pareto_frontier import pareto_filter

    rows = [
        {"cost": 100, "makespan": 200, "tardiness": 0},
        {"cost": 120, "makespan": 220, "tardiness": 10},
        {"cost": 80, "makespan": 250, "tardiness": 0},
    ]

    frontier = pareto_filter(rows)

    assert len(frontier) == 2
    assert not any(r["cost"] == 120 for r in frontier)


def test_budget_is_respected():
    jobs = [
        Job("A", 0, 100, Resources(2, 1, 4)),
    ]

    result = optimize_hybrid(
        jobs,
        default_infrastructure(),
        max_budget_units=100,
        cost_weight=1,
    )

    assert result.status == "OPTIMAL"
    assert result.total_cost_units <= 100


def test_infeasible_budget():
    jobs = [
        Job("A", 0, 100, Resources(2, 1, 4)),
    ]

    result = optimize_hybrid(
        jobs,
        default_infrastructure(),
        max_budget_units=0,
    )

    assert result.status == "INFEASIBLE"

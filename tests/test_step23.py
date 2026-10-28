"""STEP 23 statistical and deadline sensitivity tests."""

import pytest

from experiments.step23_statistical_evaluation import (
    paired_bootstrap,
)
from experiments.step23_pareto_stability import analyze
from experiments.step23_deadline_sensitivity import (
    apply_deadline_margin,
)
from hybridsched.job import Job
from hybridsched.resources import Resources


def job():
    return Job(
        job_id="A",
        arrival_s=100,
        requested_walltime_s=200,
        resources=Resources(2, 1, 8),
    )


def test_bootstrap_constant_differences():
    avg, low, high = paired_bootstrap([-5.0] * 10)
    assert (avg, low, high) == (-5, -5, -5)


def test_bootstrap_empty_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap([])


def test_deadline_margin_zero():
    result = apply_deadline_margin([job()], 0)
    assert result[0].deadline_s == 300


def test_deadline_margin_600():
    result = apply_deadline_margin([job()], 600)
    assert result[0].deadline_s == 900


def test_negative_margin_rejected():
    with pytest.raises(ValueError):
        apply_deadline_margin([job()], -1)


def test_pareto_stability():
    rows = [
        {
            "scenario": "TEST",
            "seed": "1",
            "policy": "A",
            "late_jobs": "1",
            "tardiness_s": "100",
            "cost_units": "100",
        },
        {
            "scenario": "TEST",
            "seed": "1",
            "policy": "B",
            "late_jobs": "2",
            "tardiness_s": "200",
            "cost_units": "200",
        },
    ]
    result = analyze(rows)
    indexed = {r["policy"]: r for r in result}

    assert indexed["A"]["pareto_count"] == 1
    assert indexed["B"]["pareto_count"] == 0

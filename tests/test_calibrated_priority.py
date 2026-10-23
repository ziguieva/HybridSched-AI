"""Tests for calibrated scheduling priorities."""

import pytest

from experiments.calibrate_priority import dominates
from experiments.evaluate_calibrated import relative_gain_ci
from hybridsched.calibrated_priority import (
    CalibratedPrioritySimulator,
    PriorityWeights,
)
from hybridsched.hybrid_priority import HybridPrioritySimulator
from hybridsched.infrastructure import default_infrastructure
from hybridsched.job import Job
from hybridsched.resources import Resources


def make_jobs():
    return [
        Job(
            f"J{i}",
            i * 10,
            100 + (i % 3) * 50,
            Resources(2, 1, 8),
            deadline_s=250 + i * 20,
        )
        for i in range(8)
    ]


def test_default_weights_match_original():
    jobs = make_jobs()
    clusters = default_infrastructure()

    original = HybridPrioritySimulator(
        jobs, clusters
    ).run()

    calibrated = CalibratedPrioritySimulator(
        jobs,
        clusters,
        weights=PriorityWeights(1, 2, 30),
    ).run()

    assert original == calibrated


def test_invalid_weights():
    with pytest.raises(ValueError):
        PriorityWeights(1, -1, 30)


def test_pareto_dominance():
    better = {
        "late": 5,
        "tardiness": 100,
        "makespan": 900,
        "cost": 3000,
    }

    worse = {
        "late": 6,
        "tardiness": 100,
        "makespan": 950,
        "cost": 3000,
    }

    assert dominates(better, worse)
    assert not dominates(worse, better)


def test_bootstrap_reproducibility():
    pairs = [
        (100 + i, 90 + i)
        for i in range(30)
    ]

    a = relative_gain_ci(pairs, seed=42)
    b = relative_gain_ci(pairs, seed=42)

    assert a == b
    assert a[1] <= a[0] <= a[2]


def test_calibrated_scheduler_completes():
    simulator = CalibratedPrioritySimulator(
        make_jobs(),
        default_infrastructure(),
        weights=PriorityWeights(1, 0.25, 0),
    )

    results = simulator.run()

    assert len(results) == 8
    assert all(
        result.start_s >= result.arrival_s
        for result in results
    )

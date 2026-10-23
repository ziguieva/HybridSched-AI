"""STEP 16 research regression tests."""

import pytest

from experiments.analyze_robustness import analyze
from hybridsched.infrastructure import default_infrastructure
from hybridsched.job import Job
from hybridsched.resources import Resources
from hybridsched.risk_priority import (
    RiskPrioritySimulator,
    RiskWeights,
)


def sample_jobs():
    return [
        Job(
            f"J{i}",
            i * 10,
            120,
            Resources(2, 1, 8),
            deadline_s=250 + i * 20,
        )
        for i in range(6)
    ]


def test_risk_policy_completes():
    sim = RiskPrioritySimulator(
        sample_jobs(),
        default_infrastructure(),
    )
    assert len(sim.run()) == 6


def test_risk_policy_is_online():
    results = RiskPrioritySimulator(
        sample_jobs(),
        default_infrastructure(),
    ).run()
    assert all(r.start_s >= r.arrival_s for r in results)


def test_invalid_risk_weights():
    with pytest.raises(ValueError):
        RiskWeights(1, -0.5, 0)


def test_distribution_analysis_has_all_modes():
    records = []
    for mode in ("COUPLED", "INDEPENDENT", "TIGHT"):
        for policy in ("SJF", "SELECTED"):
            records.append({
                "seed": "1",
                "mode": mode,
                "policy": policy,
                "late": "5",
                "tardiness": "100",
                "makespan": "300",
                "cost": "500",
                "mean_wait": "20",
            })

    rows = analyze(records)
    assert len(rows) == 15
    assert all(row["gain_pct"] == 0 for row in rows)

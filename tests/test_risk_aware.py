"""Risk-aware scheduler unit and integration tests."""

import pytest

from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.resources import Resources
from hybridsched.risk_aware import (
    RiskAwareEASYSimulator,
    RuntimeQuantiles,
)


def make_cluster():
    return Cluster("risk-test", Resources(16, 4, 64))


def make_jobs():
    return [
        Job("A", 0, 100, Resources(4, 4, 16)),
        Job("B", 10, 100, Resources(4, 4, 16)),
    ]


def make_quantiles():
    return {
        "A": RuntimeQuantiles(20, 40, 80),
        "B": RuntimeQuantiles(30, 50, 90),
    }


def test_quantile_order():
    with pytest.raises(ValueError):
        RuntimeQuantiles(40, 20, 80)


def test_negative_quantile():
    with pytest.raises(ValueError):
        RuntimeQuantiles(-1, 20, 80)


def test_missing_quantiles():
    with pytest.raises(ValueError):
        RiskAwareEASYSimulator(
            make_cluster(),
            make_jobs(),
            runtime_quantiles={
                "A": RuntimeQuantiles(20, 40, 80)
            },
        )


def test_actual_runtime_respected():
    cluster = make_cluster()

    results = RiskAwareEASYSimulator(
        cluster,
        make_jobs(),
        runtime_quantiles=make_quantiles(),
        actual_runtimes_s={"A": 30, "B": 40},
    ).run()

    by_id = {r.job_id: r for r in results}

    assert by_id["A"].finish_s == 30
    assert by_id["B"].start_s == 30
    assert by_id["B"].finish_s == 70
    assert cluster.available == cluster.capacity


def test_risk_aware_preserves_head_reservation():
    jobs = [
        Job("A", 0, 100, Resources(2, 2, 8)),
        Job("B", 10, 60, Resources(2, 3, 8)),
        Job("C", 20, 200, Resources(2, 2, 8)),
        Job("D", 30, 40, Resources(2, 1, 8)),
    ]

    quantiles = {
        "A": RuntimeQuantiles(50, 80, 100),
        "B": RuntimeQuantiles(30, 50, 60),
        "C": RuntimeQuantiles(80, 150, 200),
        "D": RuntimeQuantiles(10, 25, 40),
    }

    cluster = make_cluster()

    results = RiskAwareEASYSimulator(
        cluster,
        jobs,
        runtime_quantiles=quantiles,
        actual_runtimes_s={
            "A": 100,
            "B": 60,
            "C": 150,
            "D": 30,
        },
    ).run()

    by_id = {r.job_id: r for r in results}

    assert by_id["B"].start_s == 100
    assert by_id["D"].start_s == 30
    assert cluster.available == cluster.capacity

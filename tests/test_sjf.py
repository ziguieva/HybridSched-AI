"""Tests for resource-aware Shortest Job First scheduling."""

import pytest

from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.metrics import compute_metrics
from hybridsched.resources import Resources
from hybridsched.sjf import SJFSimulator


def make_cluster():
    return Cluster(
        "test-sjf",
        Resources(
            cpu_cores=16,
            gpus=4,
            memory_gib=64,
        ),
    )


def test_sjf_reference_scenario():
    cluster = make_cluster()

    jobs = [
        Job("J001", 0, 900, Resources(4, 2, 16)),
        Job("J002", 60, 1200, Resources(8, 3, 32)),
        Job("J003", 120, 300, Resources(2, 1, 8)),
    ]

    results = SJFSimulator(cluster, jobs).run()
    by_id = {r.job_id: r for r in results}

    assert by_id["J001"].start_s == 0
    assert by_id["J002"].start_s == 900
    assert by_id["J003"].start_s == 120

    assert by_id["J002"].waiting_s == 840
    assert by_id["J003"].waiting_s == 0

    metrics = compute_metrics(
        jobs,
        results,
        cluster.capacity,
    )

    assert metrics.average_waiting_s == 280
    assert metrics.makespan_s == 2100

    assert metrics.gpu_utilization_pct == pytest.approx(
        67.8571428571
    )


def test_shortest_waiting_job_selected_first():
    cluster = make_cluster()

    jobs = [
        Job("A", 0, 100, Resources(8, 4, 32)),
        Job("B", 10, 200, Resources(8, 3, 32)),
        Job("C", 20, 20, Resources(8, 3, 32)),
    ]

    results = SJFSimulator(cluster, jobs).run()
    by_id = {r.job_id: r for r in results}

    assert by_id["A"].start_s == 0
    assert by_id["A"].finish_s == 100

    assert by_id["C"].start_s == 100
    assert by_id["C"].finish_s == 120

    assert by_id["B"].start_s == 120
    assert by_id["B"].finish_s == 320


def test_jobs_never_start_before_arrival():
    cluster = make_cluster()

    jobs = [
        Job("J001", 50, 100, Resources(4, 2, 16)),
        Job("J002", 80, 40, Resources(2, 1, 8)),
        Job("J003", 120, 20, Resources(2, 1, 8)),
    ]

    results = SJFSimulator(cluster, jobs).run()

    for result in results:
        assert result.start_s >= result.arrival_s
        assert result.waiting_s >= 0


def test_resources_released_after_sjf():
    cluster = make_cluster()

    jobs = [
        Job("J001", 0, 100, Resources(4, 2, 16)),
        Job("J002", 10, 50, Resources(4, 2, 16)),
        Job("J003", 20, 20, Resources(2, 1, 8)),
    ]

    results = SJFSimulator(cluster, jobs).run()

    assert len(results) == len(jobs)
    assert cluster.available == cluster.capacity
    assert cluster.running_job_ids == ()


def test_sjf_rejects_impossible_job():
    cluster = make_cluster()

    jobs = [
        Job(
            "J001",
            0,
            100,
            Resources(8, 8, 32),
        ),
    ]

    with pytest.raises(ValueError, match="capacity"):
        SJFSimulator(cluster, jobs)

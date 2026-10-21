"""Tests for HPC scheduling performance metrics."""

import pytest

from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.metrics import compute_metrics
from hybridsched.resources import Resources
from hybridsched.simulation import FCFSSimulator, JobResult


def test_reference_metrics():
    cluster = Cluster(
        "local-hpc",
        Resources(16, 4, 64),
    )

    jobs = [
        Job("J001", 0, 900, Resources(4, 2, 16)),
        Job("J002", 60, 1200, Resources(8, 3, 32)),
        Job("J003", 120, 300, Resources(2, 1, 8)),
    ]

    results = FCFSSimulator(cluster, jobs).run()
    metrics = compute_metrics(jobs, results, cluster.capacity)

    assert metrics.total_jobs == 3
    assert metrics.makespan_s == 2100
    assert metrics.average_waiting_s == 540
    assert metrics.p95_waiting_s == 840

    assert metrics.gpu_utilization_pct == pytest.approx(
        67.8571428571
    )

    assert metrics.cpu_utilization_pct == pytest.approx(
        41.0714285714
    )

    assert metrics.throughput_jobs_hour == pytest.approx(
        5.1428571429
    )


def test_empty_workload_metrics():
    metrics = compute_metrics(
        jobs=[],
        results=[],
        capacity=Resources(8, 2, 32),
    )

    assert metrics.total_jobs == 0
    assert metrics.makespan_s == 0
    assert metrics.average_waiting_s == 0
    assert metrics.p95_waiting_s == 0
    assert metrics.gpu_utilization_pct == 0
    assert metrics.cpu_utilization_pct == 0
    assert metrics.throughput_jobs_hour == 0


def test_cpu_only_metrics():
    cluster = Cluster(
        "cpu-cluster",
        Resources(8, 0, 32),
    )

    jobs = [
        Job("J001", 0, 10, Resources(2, 0, 4)),
    ]

    results = FCFSSimulator(cluster, jobs).run()
    metrics = compute_metrics(jobs, results, cluster.capacity)

    assert metrics.gpu_utilization_pct == 0
    assert metrics.cpu_utilization_pct == 25
    assert metrics.makespan_s == 10
    assert metrics.throughput_jobs_hour == 360


def test_p95_waiting_time():
    waiting_times = [0, 1, 2, 3, 10]

    jobs = [
        Job(
            f"J{i}",
            0,
            10,
            Resources(1, 0, 1),
        )
        for i in range(5)
    ]

    results = [
        JobResult(
            job_id=f"J{i}",
            arrival_s=0,
            start_s=wait,
            finish_s=wait + 10,
            waiting_s=wait,
        )
        for i, wait in enumerate(waiting_times)
    ]

    metrics = compute_metrics(
        jobs,
        results,
        Resources(5, 0, 16),
    )

    assert metrics.average_waiting_s == pytest.approx(3.2)
    assert metrics.p95_waiting_s == 10


def test_missing_job_result_rejected():
    jobs = [
        Job("J001", 0, 10, Resources(1, 0, 1)),
    ]

    with pytest.raises(ValueError, match="match uniquely"):
        compute_metrics(
            jobs,
            [],
            Resources(4, 0, 16),
        )


def test_duplicate_results_rejected():
    jobs = [
        Job("J001", 0, 10, Resources(1, 0, 1)),
    ]

    result = JobResult(
        job_id="J001",
        arrival_s=0,
        start_s=0,
        finish_s=10,
        waiting_s=0,
    )

    with pytest.raises(ValueError, match="match uniquely"):
        compute_metrics(
            jobs,
            [result, result],
            Resources(4, 0, 16),
        )


def test_duplicate_jobs_rejected():
    job = Job(
        "J001",
        0,
        10,
        Resources(1, 0, 1),
    )

    result = JobResult(
        job_id="J001",
        arrival_s=0,
        start_s=0,
        finish_s=10,
        waiting_s=0,
    )

    with pytest.raises(ValueError, match="match uniquely"):
        compute_metrics(
            [job, job],
            [result],
            Resources(4, 0, 16),
        )

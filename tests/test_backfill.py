"""Tests for EASY-style backfilling scheduling."""

import pytest

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.metrics import compute_metrics
from hybridsched.resources import Resources


def make_cluster():
    return Cluster(
        "backfill-test",
        Resources(16, 4, 64),
    )


def make_job(job_id, arrival, duration, gpus):
    return Job(
        job_id=job_id,
        arrival_s=arrival,
        requested_walltime_s=duration,
        resources=Resources(2, gpus, 4),
    )


def test_reference_scenario():
    cluster = make_cluster()

    jobs = [
        make_job("J001", 0, 900, 2),
        make_job("J002", 60, 1200, 3),
        make_job("J003", 120, 300, 1),
    ]

    results = EASYBackfillSimulator(cluster, jobs).run()
    by_id = {r.job_id: r for r in results}

    assert by_id["J001"].start_s == 0
    assert by_id["J002"].start_s == 900
    assert by_id["J003"].start_s == 120

    metrics = compute_metrics(
        jobs, results, cluster.capacity
    )

    assert metrics.average_waiting_s == 280
    assert metrics.makespan_s == 2100
    assert cluster.available == cluster.capacity


def test_backfill_does_not_delay_reserved_job():
    cluster = make_cluster()

    jobs = [
        make_job("A", 0, 100, 2),
        make_job("B", 10, 60, 3),
        make_job("C", 20, 200, 2),
    ]

    results = EASYBackfillSimulator(cluster, jobs).run()
    by_id = {r.job_id: r for r in results}

    assert by_id["A"].finish_s == 100
    assert by_id["B"].start_s == 100
    assert by_id["C"].start_s == 160


def test_safe_long_backfill_allowed():
    cluster = make_cluster()

    jobs = [
        make_job("A", 0, 100, 2),
        make_job("B", 10, 60, 3),
        make_job("C", 20, 200, 1),
    ]

    results = EASYBackfillSimulator(cluster, jobs).run()
    by_id = {r.job_id: r for r in results}

    assert by_id["C"].start_s == 20
    assert by_id["B"].start_s == 100
    assert by_id["C"].finish_s == 220
    assert cluster.available == cluster.capacity


def test_reservation_requires_multiple_completions():
    cluster = make_cluster()

    jobs = [
        make_job("A", 0, 100, 1),
        make_job("D", 0, 200, 2),
        make_job("B", 10, 50, 4),
        make_job("C", 20, 50, 1),
    ]

    results = EASYBackfillSimulator(cluster, jobs).run()
    by_id = {r.job_id: r for r in results}

    assert by_id["C"].start_s == 20
    assert by_id["B"].start_s == 200
    assert cluster.available == cluster.capacity


def test_empty_workload():
    cluster = make_cluster()

    results = EASYBackfillSimulator(cluster, []).run()

    assert results == []
    assert cluster.available == cluster.capacity


def test_impossible_job_rejected():
    cluster = make_cluster()

    jobs = [
        make_job("A", 0, 100, 8),
    ]

    with pytest.raises(ValueError, match="capacity"):
        EASYBackfillSimulator(cluster, jobs)


def test_duplicate_job_ids_rejected():
    cluster = make_cluster()

    jobs = [
        make_job("A", 0, 100, 2),
        make_job("A", 10, 50, 1),
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        EASYBackfillSimulator(cluster, jobs)

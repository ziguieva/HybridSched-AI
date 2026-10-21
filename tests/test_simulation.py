"""Automated tests for the HPC simulation engine."""

import pytest

from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.resources import Resources
from hybridsched.simulation import FCFSSimulator


def make_cluster():
    return Cluster(
        "test-hpc",
        Resources(cpu_cores=16, gpus=4, memory_gib=64),
    )


def test_fcfs_scheduling():
    cluster = make_cluster()

    jobs = [
        Job("J001", 0, 900, Resources(4, 2, 16)),
        Job("J002", 60, 1200, Resources(8, 3, 32)),
        Job("J003", 120, 300, Resources(2, 1, 8)),
    ]

    results = FCFSSimulator(cluster, jobs).run()

    assert len(results) == 3

    assert results[0].start_s == 0
    assert results[0].finish_s == 900

    assert results[1].start_s == 900
    assert results[1].waiting_s == 840

    assert results[2].start_s == 900
    assert results[2].waiting_s == 780

    assert max(r.finish_s for r in results) == 2100

    assert sum(r.waiting_s for r in results) / 3 == 540


def test_parallel_execution():
    cluster = make_cluster()

    jobs = [
        Job("J001", 0, 20, Resources(4, 2, 16)),
        Job("J002", 5, 10, Resources(2, 1, 8)),
    ]

    results = FCFSSimulator(cluster, jobs).run()

    assert results[0].start_s == 0
    assert results[1].start_s == 5
    assert results[1].finish_s == 15


def test_resources_released_after_execution():
    cluster = make_cluster()

    jobs = [
        Job("J001", 0, 100, Resources(4, 2, 16)),
    ]

    FCFSSimulator(cluster, jobs).run()

    assert cluster.available == cluster.capacity
    assert cluster.running_job_ids == ()


def test_empty_workload():
    cluster = make_cluster()

    results = FCFSSimulator(cluster, []).run()

    assert results == []
    assert cluster.available == cluster.capacity


def test_duplicate_job_ids_rejected():
    cluster = make_cluster()

    jobs = [
        Job("J001", 0, 100, Resources(2, 1, 8)),
        Job("J001", 10, 50, Resources(2, 1, 8)),
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        FCFSSimulator(cluster, jobs)


def test_impossible_job_rejected():
    cluster = make_cluster()

    jobs = [
        Job("J001", 0, 100, Resources(4, 8, 16)),
    ]

    with pytest.raises(ValueError, match="capacity"):
        FCFSSimulator(cluster, jobs)


def test_simulation_cannot_run_twice():
    cluster = make_cluster()

    simulator = FCFSSimulator(cluster, [])

    simulator.run()

    with pytest.raises(RuntimeError):
        simulator.run()

"""Tests for HPC resource allocation."""

import pytest

from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.resources import Resources


@pytest.mark.parametrize(
    "cpu,gpu,memory",
    [
        (-1, 0, 1),
        (1, -1, 1),
        (1, 0, -1),
        (True, 0, 1),
        (1, 0, 1.5),
    ],
)
def test_invalid_resources(cpu, gpu, memory):
    with pytest.raises(ValueError):
        Resources(cpu, gpu, memory)


def test_resource_arithmetic_and_capacity():
    a = Resources(4, 1, 8)
    b = Resources(8, 2, 16)

    assert a + b == Resources(12, 3, 24)
    assert b - a == Resources(4, 1, 8)
    assert a.fits_in(b)
    assert not b.fits_in(a)


def test_job_rejects_invalid_deadline():
    with pytest.raises(ValueError):
        Job(
            "J1",
            100,
            40,
            Resources(1, 0, 1),
            deadline_s=99,
        )


def test_job_rejects_missing_cpu():
    with pytest.raises(ValueError):
        Job("J1", 0, 40, Resources(0, 1, 1))


def test_cluster_allocates_and_releases():
    cluster = Cluster("lab", Resources(8, 2, 32))
    job = Job("J1", 0, 60, Resources(4, 1, 8))

    assert cluster.can_allocate(job)

    cluster.allocate(job)

    assert cluster.available == Resources(4, 1, 24)
    assert cluster.running_job_ids == ("J1",)

    cluster.release("J1")

    assert cluster.available == cluster.capacity
    assert cluster.running_job_ids == ()


def test_cluster_rejects_overcommit():
    cluster = Cluster("lab", Resources(8, 2, 32))

    cluster.allocate(
        Job("J1", 0, 60, Resources(4, 2, 16))
    )

    job = Job("J2", 0, 60, Resources(2, 1, 8))

    assert not cluster.can_allocate(job)

    with pytest.raises(ValueError, match="insufficient"):
        cluster.allocate(job)

    assert cluster.available == Resources(4, 0, 16)


def test_cluster_rejects_duplicate_allocation():
    cluster = Cluster("lab", Resources(8, 2, 32))
    job = Job("J1", 0, 60, Resources(4, 1, 8))

    cluster.allocate(job)

    with pytest.raises(ValueError, match="already allocated"):
        cluster.allocate(job)


def test_cluster_rejects_unknown_release():
    cluster = Cluster("lab", Resources(8, 2, 32))

    with pytest.raises(KeyError):
        cluster.release("unknown")


def test_cpu_only_jobs_supported():
    cluster = Cluster("cpu-lab", Resources(8, 0, 32))
    job = Job("J1", 0, 60, Resources(2, 0, 4))

    cluster.allocate(job)

    assert cluster.available == Resources(6, 0, 28)

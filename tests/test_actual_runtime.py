"""Validate separation between requested and actual runtimes."""

import pytest

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.resources import Resources
from hybridsched.simulation import FCFSSimulator
from hybridsched.sjf import SJFSimulator


def make_jobs():
    return [
        Job("A", 0, 100, Resources(4, 4, 16)),
        Job("B", 10, 100, Resources(4, 4, 16)),
    ]


def make_cluster():
    return Cluster("runtime-test", Resources(16, 4, 64))


@pytest.mark.parametrize(
    "scheduler",
    [
        FCFSSimulator,
        SJFSimulator,
        EASYBackfillSimulator,
    ],
)
def test_actual_runtime_execution(scheduler):
    cluster = make_cluster()

    actual = {"A": 30, "B": 40}

    results = scheduler(
        cluster,
        make_jobs(),
        actual_runtimes_s=actual,
    ).run()

    by_id = {r.job_id: r for r in results}

    assert by_id["A"].start_s == 0
    assert by_id["A"].finish_s == 30

    assert by_id["B"].start_s == 30
    assert by_id["B"].finish_s == 70
    assert by_id["B"].waiting_s == 20

    assert cluster.available == cluster.capacity


def test_legacy_execution_unchanged():
    results = FCFSSimulator(
        make_cluster(),
        make_jobs(),
    ).run()

    by_id = {r.job_id: r for r in results}

    assert by_id["A"].finish_s == 100
    assert by_id["B"].start_s == 100
    assert by_id["B"].finish_s == 200


def test_missing_actual_runtime_rejected():
    with pytest.raises(ValueError):
        FCFSSimulator(
            make_cluster(),
            make_jobs(),
            actual_runtimes_s={"A": 30},
        )


def test_invalid_actual_runtime_rejected():
    with pytest.raises(ValueError):
        FCFSSimulator(
            make_cluster(),
            make_jobs(),
            actual_runtimes_s={"A": 0, "B": 40},
        )

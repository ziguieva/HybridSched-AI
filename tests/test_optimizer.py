"""Unit tests for the CP-SAT scheduling optimizer."""

import pytest

from hybridsched.job import Job
from hybridsched.optimizer import optimize_schedule
from hybridsched.resources import Resources


def test_empty_schedule():
    result = optimize_schedule(
        [],
        Resources(4, 2, 16),
    )

    assert result.status == "OPTIMAL"
    assert result.results == []
    assert result.makespan_s == 0


def test_single_job():
    jobs = [
        Job("A", 10, 100, Resources(2, 1, 4)),
    ]

    result = optimize_schedule(
        jobs,
        Resources(4, 2, 16),
    )

    assert result.status == "OPTIMAL"
    assert result.results[0].start_s == 10
    assert result.results[0].finish_s == 110


def test_gpu_capacity():
    jobs = [
        Job("A", 0, 100, Resources(2, 2, 4)),
        Job("B", 0, 100, Resources(2, 2, 4)),
    ]

    result = optimize_schedule(
        jobs,
        Resources(8, 2, 32),
    )

    assert result.status == "OPTIMAL"
    assert result.makespan_s == 200


def test_cpu_capacity():
    jobs = [
        Job("A", 0, 50, Resources(4, 0, 8)),
        Job("B", 0, 50, Resources(4, 0, 8)),
    ]

    result = optimize_schedule(
        jobs,
        Resources(4, 0, 32),
    )

    assert result.status == "OPTIMAL"
    assert result.makespan_s == 100


def test_memory_capacity():
    jobs = [
        Job("A", 0, 50, Resources(1, 0, 16)),
        Job("B", 0, 50, Resources(1, 0, 16)),
    ]

    result = optimize_schedule(
        jobs,
        Resources(8, 0, 16),
    )

    assert result.status == "OPTIMAL"
    assert result.makespan_s == 100


def test_arrival_constraints():
    jobs = [
        Job("A", 100, 20, Resources(1, 0, 2)),
        Job("B", 200, 20, Resources(1, 0, 2)),
    ]

    result = optimize_schedule(
        jobs,
        Resources(4, 0, 16),
    )

    assert result.status == "OPTIMAL"

    by_id = {r.job_id: r for r in result.results}

    assert by_id["A"].start_s >= 100
    assert by_id["B"].start_s >= 200


def test_impossible_job():
    jobs = [
        Job("A", 0, 100, Resources(2, 8, 4)),
    ]

    with pytest.raises(ValueError, match="capacity"):
        optimize_schedule(
            jobs,
            Resources(8, 4, 32),
        )


def test_duplicate_ids():
    jobs = [
        Job("A", 0, 10, Resources(1, 0, 2)),
        Job("A", 10, 20, Resources(1, 0, 2)),
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        optimize_schedule(
            jobs,
            Resources(4, 0, 16),
        )

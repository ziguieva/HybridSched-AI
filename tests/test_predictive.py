"""Tests for prediction-aware scheduling policies."""

import pytest

from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.predictive import (
    PredictedEASYBackfillSimulator,
    PredictedSJFSimulator,
)
from hybridsched.resources import Resources


def make_cluster():
    return Cluster(
        "predictive-test",
        Resources(16, 4, 64),
    )


def make_job(job_id, arrival, duration, gpus):
    return Job(
        job_id,
        arrival,
        duration,
        Resources(2, gpus, 4),
    )


def test_predictive_sjf_uses_estimates():
    cluster = make_cluster()

    jobs = [
        make_job("A", 0, 100, 4),
        make_job("B", 10, 200, 4),
        make_job("C", 20, 20, 4),
    ]

    predictions = {
        "A": 100,
        "B": 10,
        "C": 300,
    }

    actual = {
        "A": 100,
        "B": 60,
        "C": 30,
    }

    results = PredictedSJFSimulator(
        cluster,
        jobs,
        runtime_predictions_s=predictions,
        actual_runtimes_s=actual,
    ).run()

    by_id = {r.job_id: r for r in results}

    assert by_id["A"].start_s == 0
    assert by_id["B"].start_s == 100
    assert by_id["C"].start_s == 160
    assert cluster.available == cluster.capacity


def test_predicted_runtime_is_not_actual_runtime():
    cluster = make_cluster()

    jobs = [
        make_job("A", 0, 100, 4),
    ]

    results = PredictedSJFSimulator(
        cluster,
        jobs,
        runtime_predictions_s={"A": 15},
        actual_runtimes_s={"A": 80},
    ).run()

    assert results[0].finish_s == 80
    assert results[0].finish_s != 15


def test_predictive_easy_protects_reservation():
    cluster = make_cluster()

    jobs = [
        make_job("A", 0, 100, 2),
        make_job("B", 10, 60, 3),
        make_job("C", 20, 200, 2),
        make_job("D", 30, 50, 1),
    ]

    predictions = {
        "A": 100,
        "B": 60,
        "C": 1,
        "D": 1000,
    }

    actual = {
        "A": 100,
        "B": 60,
        "C": 150,
        "D": 40,
    }

    results = PredictedEASYBackfillSimulator(
        cluster,
        jobs,
        runtime_predictions_s=predictions,
        actual_runtimes_s=actual,
    ).run()

    by_id = {r.job_id: r for r in results}

    assert by_id["A"].finish_s == 100
    assert by_id["B"].start_s == 100

    assert by_id["D"].start_s == 30
    assert by_id["D"].finish_s == 70

    assert by_id["C"].start_s >= 100
    assert cluster.available == cluster.capacity


@pytest.mark.parametrize(
    "scheduler",
    [
        PredictedSJFSimulator,
        PredictedEASYBackfillSimulator,
    ],
)
def test_missing_predictions_rejected(scheduler):
    jobs = [
        make_job("A", 0, 100, 1),
        make_job("B", 10, 100, 1),
    ]

    with pytest.raises(ValueError, match="Predictions"):
        scheduler(
            make_cluster(),
            jobs,
            runtime_predictions_s={"A": 100},
        )


@pytest.mark.parametrize(
    "scheduler",
    [
        PredictedSJFSimulator,
        PredictedEASYBackfillSimulator,
    ],
)
@pytest.mark.parametrize("invalid", [0, -10])
def test_nonpositive_predictions_rejected(
    scheduler,
    invalid,
):
    jobs = [
        make_job("A", 0, 100, 1),
    ]

    with pytest.raises(ValueError, match="positive"):
        scheduler(
            make_cluster(),
            jobs,
            runtime_predictions_s={"A": invalid},
        )

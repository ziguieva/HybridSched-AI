"""STEP 21 scheduling regression tests."""

from types import SimpleNamespace

import pytest

from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.resources import Resources
from hybridsched.slack_priority import SlackPrioritySimulator


def cluster():
    return ClusterSpec(
        "local", "ON_PREMISE",
        Resources(4, 2, 16),
    )


def job(jid, arrival=0, deadline=150):
    return Job(
        job_id=jid,
        arrival_s=arrival,
        requested_walltime_s=100,
        resources=Resources(2, 2, 8),
        deadline_s=deadline,
    )


class Predictor:
    def predict_many(self, jobs):
        return {
            j.job_id: SimpleNamespace(
                point_s=50,
                p10_s=20,
                p50_s=40,
                p90_s=100,
            )
            for j in jobs
        }


def test_actual_duration_controls_finish():
    sim = SlackPrioritySimulator(
        [job("A")], [cluster()],
        {"A": 250}, Predictor(),
    )
    result = sim.run()[0]
    assert result.finish_s == 250
    assert result.tardiness_s == 100


def test_arrival_respected():
    sim = SlackPrioritySimulator(
        [job("A", arrival=30)],
        [cluster()],
        {"A": 100}, Predictor(),
    )
    assert sim.run()[0].start_s >= 30


def test_two_jobs_complete():
    jobs = [
        job("A", deadline=100),
        job("B", deadline=500),
    ]
    sim = SlackPrioritySimulator(
        jobs, [cluster()],
        {"A": 100, "B": 100}, Predictor(),
    )
    results = sim.run()
    assert len(results) == 2
    assert len({r.job_id for r in results}) == 2


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        SlackPrioritySimulator(
            [job("A")], [cluster()],
            {"A": 100}, Predictor(),
            slack_weight=-0.1,
        )


def test_missing_runtime_rejected():
    with pytest.raises(ValueError):
        SlackPrioritySimulator(
            [job("A")], [cluster()],
            {}, Predictor(),
        )

"""Scheduler reproducibility and resource capacity tests."""

import random

import pytest

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.predictive import (
    PredictedEASYBackfillSimulator,
    PredictedSJFSimulator,
)
from hybridsched.resources import Resources
from hybridsched.simulation import FCFSSimulator
from hybridsched.sjf import SJFSimulator


CAPACITY = Resources(16, 4, 64)


def make_workload():
    rng = random.Random(2026)
    jobs = []
    actual = {}
    predictions = {}
    arrival = 0

    for i in range(40):
        arrival += rng.randint(0, 15)

        duration = rng.choice([100, 200, 500, 900])
        gpus = rng.choice([0, 1, 2, 3])
        cpus = rng.choice([2, 4, 8])
        memory = 4 + 8 * gpus + cpus

        job = Job(
            job_id=f"J{i:03d}",
            arrival_s=arrival,
            requested_walltime_s=duration,
            resources=Resources(cpus, gpus, memory),
        )

        jobs.append(job)
        actual[job.job_id] = rng.randint(1, duration)
        predictions[job.job_id] = rng.randint(10, 1000)

    return jobs, actual, predictions


@pytest.mark.parametrize(
    "scheduler",
    [
        FCFSSimulator,
        SJFSimulator,
        EASYBackfillSimulator,
        PredictedSJFSimulator,
        PredictedEASYBackfillSimulator,
    ],
)
def test_scheduler_invariants(scheduler):
    jobs, actual, predictions = make_workload()
    job_map = {job.job_id: job for job in jobs}

    def execute():
        cluster = Cluster("test", CAPACITY)

        kwargs = {"actual_runtimes_s": actual}

        if scheduler in (
            PredictedSJFSimulator,
            PredictedEASYBackfillSimulator,
        ):
            kwargs["runtime_predictions_s"] = predictions

        results = scheduler(cluster, jobs, **kwargs).run()

        assert len(results) == len(jobs)
        assert cluster.available == CAPACITY
        assert cluster.running_job_ids == ()

        for result in results:
            assert result.start_s >= result.arrival_s
            assert (
                result.waiting_s
                == result.start_s - result.arrival_s
            )
            assert (
                result.finish_s - result.start_s
                == actual[result.job_id]
            )

        event_times = sorted({
            t
            for result in results
            for t in (result.start_s, result.finish_s)
        })

        for t in event_times:
            active = [
                job_map[r.job_id].resources
                for r in results
                if r.start_s <= t < r.finish_s
            ]

            assert sum(r.cpu_cores for r in active) <= CAPACITY.cpu_cores
            assert sum(r.gpus for r in active) <= CAPACITY.gpus
            assert sum(r.memory_gib for r in active) <= CAPACITY.memory_gib

        return results

    assert execute() == execute()


def test_workload_reproducibility():
    assert make_workload() == make_workload()

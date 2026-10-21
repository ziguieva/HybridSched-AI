"""Discrete-event HPC simulation using SimPy."""

from collections import deque
from dataclasses import dataclass

import simpy

from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.resources import Resources


@dataclass(frozen=True, slots=True)
class JobResult:
    job_id: str
    arrival_s: int
    start_s: int
    finish_s: int
    waiting_s: int


class FCFSSimulator:
    """First-Come, First-Served HPC scheduler."""

    def __init__(
        self,
        cluster: Cluster,
        jobs: list[Job],
        actual_runtimes_s: dict[str, int] | None = None,
    ):
        if cluster.running_job_ids:
            raise ValueError("Cluster must initially be empty")

        ids = [job.job_id for job in jobs]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate job IDs")

        for job in jobs:
            if not job.resources.fits_in(cluster.capacity):
                raise ValueError(
                    f"Job {job.job_id} exceeds cluster capacity"
                )

        if actual_runtimes_s is not None:
            if set(actual_runtimes_s) != set(ids):
                raise ValueError(
                    "Actual runtimes must match all job IDs"
                )

            if any(
                type(value) is not int or value <= 0
                for value in actual_runtimes_s.values()
            ):
                raise ValueError(
                    "Actual runtimes must be positive integers"
                )

        self._actual_runtimes_s = (
            dict(actual_runtimes_s)
            if actual_runtimes_s is not None
            else None
        )

        self.env = simpy.Environment()
        self.cluster = cluster
        self.jobs = sorted(
            jobs,
            key=lambda job: (job.arrival_s, job.job_id),
        )

        self.waiting = deque()
        self.results: list[JobResult] = []
        self._has_run = False

    def _arrival(self, job: Job):
        yield self.env.timeout(job.arrival_s)

        self.waiting.append(job)
        self._dispatch()

    def _dispatch(self) -> None:
        while self.waiting:
            job = self.waiting[0]

            if not self.cluster.can_allocate(job):
                break

            self.waiting.popleft()
            self.cluster.allocate(job)

            self.env.process(self._execute(job))

    def _execute(self, job: Job):
        start_s = int(self.env.now)

        actual_duration = (
            self._actual_runtimes_s[job.job_id]
            if self._actual_runtimes_s is not None
            else job.requested_walltime_s
        )

        yield self.env.timeout(actual_duration)

        finish_s = int(self.env.now)

        result = JobResult(
            job_id=job.job_id,
            arrival_s=job.arrival_s,
            start_s=start_s,
            finish_s=finish_s,
            waiting_s=start_s - job.arrival_s,
        )

        self.results.append(result)
        self.cluster.release(job.job_id)
        self._dispatch()

    def run(self) -> list[JobResult]:
        if self._has_run:
            raise RuntimeError(
                "Simulation can only run once"
            )

        self._has_run = True

        for job in self.jobs:
            self.env.process(self._arrival(job))

        self.env.run()

        return sorted(
            self.results,
            key=lambda result: result.job_id,
        )


def main() -> None:
    cluster = Cluster(
        name="local-hpc",
        capacity=Resources(
            cpu_cores=16,
            gpus=4,
            memory_gib=64,
        ),
    )

    jobs = [
        Job(
            job_id="J001",
            arrival_s=0,
            requested_walltime_s=900,
            resources=Resources(4, 2, 16),
        ),
        Job(
            job_id="J002",
            arrival_s=60,
            requested_walltime_s=1200,
            resources=Resources(8, 3, 32),
        ),
        Job(
            job_id="J003",
            arrival_s=120,
            requested_walltime_s=300,
            resources=Resources(2, 1, 8),
        ),
    ]

    simulator = FCFSSimulator(cluster, jobs)
    results = simulator.run()

    print("=" * 65)
    print("HybridSched-AI | HPC DISCRETE-EVENT SIMULATION")
    print("=" * 65)

    print(f"Cluster: {cluster.name}")
    print(f"GPU capacity: {cluster.capacity.gpus}")
    print(f"Total jobs: {len(jobs)}")

    print("-" * 65)
    print(
        f"{'Job':<8}"
        f"{'Arrival':>10}"
        f"{'Start':>10}"
        f"{'Finish':>10}"
        f"{'Waiting':>10}"
    )
    print("-" * 65)

    for result in results:
        print(
            f"{result.job_id:<8}"
            f"{result.arrival_s:>10}"
            f"{result.start_s:>10}"
            f"{result.finish_s:>10}"
            f"{result.waiting_s:>10}"
        )

    print("-" * 65)

    makespan = max(
        (result.finish_s for result in results),
        default=0,
    )

    average_wait = (
        sum(r.waiting_s for r in results) / len(results)
        if results
        else 0
    )

    print(f"Makespan: {makespan} seconds")
    print(f"Average waiting time: {average_wait:.2f} seconds")
    print(f"Final available resources: {cluster.available}")
    print("=" * 65)


if __name__ == "__main__":
    main()

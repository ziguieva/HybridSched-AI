"""Performance metrics for HPC scheduling simulations."""

from dataclasses import dataclass
from math import ceil

from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.resources import Resources
from hybridsched.simulation import FCFSSimulator, JobResult


@dataclass(frozen=True)
class HPCMetrics:
    total_jobs: int
    makespan_s: int
    average_waiting_s: float
    p95_waiting_s: float
    gpu_utilization_pct: float
    cpu_utilization_pct: float
    throughput_jobs_hour: float


def compute_metrics(
    jobs: list[Job],
    results: list[JobResult],
    capacity: Resources,
) -> HPCMetrics:

    if not jobs and not results:
        return HPCMetrics(0, 0, 0, 0, 0, 0, 0)

    job_map = {job.job_id: job for job in jobs}
    result_ids = [r.job_id for r in results]

    if (
        len(job_map) != len(jobs)
        or len(result_ids) != len(set(result_ids))
        or set(job_map) != set(result_ids)
    ):
        raise ValueError("Jobs and results must match uniquely")

    makespan = max(r.finish_s for r in results)

    if makespan <= 0:
        raise ValueError("Invalid simulation duration")

    waiting_times = sorted(r.waiting_s for r in results)

    average_waiting = (
        sum(waiting_times) / len(waiting_times)
    )

    p95_index = ceil(0.95 * len(waiting_times)) - 1
    p95_waiting = waiting_times[p95_index]

    gpu_seconds = sum(
        job_map[r.job_id].resources.gpus
        * (r.finish_s - r.start_s)
        for r in results
    )

    cpu_seconds = sum(
        job_map[r.job_id].resources.cpu_cores
        * (r.finish_s - r.start_s)
        for r in results
    )

    gpu_utilization = (
        100 * gpu_seconds / (capacity.gpus * makespan)
        if capacity.gpus > 0
        else 0.0
    )

    cpu_utilization = (
        100 * cpu_seconds / (capacity.cpu_cores * makespan)
        if capacity.cpu_cores > 0
        else 0.0
    )

    throughput = len(results) * 3600 / makespan

    return HPCMetrics(
        total_jobs=len(results),
        makespan_s=makespan,
        average_waiting_s=average_waiting,
        p95_waiting_s=p95_waiting,
        gpu_utilization_pct=gpu_utilization,
        cpu_utilization_pct=cpu_utilization,
        throughput_jobs_hour=throughput,
    )


def main() -> None:

    cluster = Cluster(
        "local-hpc",
        Resources(16, 4, 64),
    )

    jobs = [
        Job("J001", 0, 900, Resources(4, 2, 16)),
        Job("J002", 60, 1200, Resources(8, 3, 32)),
        Job("J003", 120, 300, Resources(2, 1, 8)),
    ]

    simulator = FCFSSimulator(cluster, jobs)
    results = simulator.run()

    metrics = compute_metrics(
        jobs,
        results,
        cluster.capacity,
    )

    print("=" * 55)
    print("HybridSched-AI | HPC PERFORMANCE METRICS")
    print("=" * 55)

    print(f"Total jobs:           {metrics.total_jobs}")
    print(f"Makespan:             {metrics.makespan_s} s")
    print(f"Average waiting:      {metrics.average_waiting_s:.2f} s")
    print(f"P95 waiting:          {metrics.p95_waiting_s:.2f} s")
    print(f"GPU utilization:      {metrics.gpu_utilization_pct:.2f}%")
    print(f"CPU utilization:      {metrics.cpu_utilization_pct:.2f}%")
    print(f"Throughput:           {metrics.throughput_jobs_hour:.2f} jobs/h")

    print("=" * 55)


if __name__ == "__main__":
    main()

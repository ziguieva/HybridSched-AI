"""Reproducible HPC scheduling benchmark."""

import random

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.metrics import compute_metrics
from hybridsched.resources import Resources
from hybridsched.simulation import FCFSSimulator
from hybridsched.sjf import SJFSimulator


SEED = 20261021
N_JOBS = 200

CLUSTER_CAPACITY = Resources(
    cpu_cores=32,
    gpus=8,
    memory_gib=128,
)


def generate_workload(
    n_jobs: int,
    seed: int,
) -> list[Job]:
    """Generate deterministic heterogeneous HPC jobs."""

    rng = random.Random(seed)
    jobs = []
    arrival_time = 0

    for i in range(n_jobs):

        arrival_time += rng.randint(5, 75)

        duration = rng.choice([
            120, 300, 600, 900, 1800
        ])

        gpus = rng.choices(
            population=[0, 1, 2, 3, 4],
            weights=[1, 4, 4, 2, 1],
            k=1,
        )[0]

        cpu_cores = rng.choice([2, 4, 8, 12])

        memory_gib = (
            8 + 12 * gpus + cpu_cores
        )

        job = Job(
            job_id=f"J{i + 1:04d}",
            arrival_s=arrival_time,
            requested_walltime_s=duration,
            resources=Resources(
                cpu_cores=cpu_cores,
                gpus=gpus,
                memory_gib=memory_gib,
            ),
            priority=rng.randint(0, 3),
        )

        jobs.append(job)

    return jobs


def main() -> None:

    jobs = generate_workload(
        n_jobs=N_JOBS,
        seed=SEED,
    )

    algorithms = {
        "FCFS": FCFSSimulator,
        "SJF": SJFSimulator,
        "EASY": EASYBackfillSimulator,
    }

    print("=" * 86)
    print("HybridSched-AI | REPRODUCIBLE HPC SCHEDULING BENCHMARK")
    print("=" * 86)

    print(f"Random seed: {SEED}")
    print(f"Number of jobs: {N_JOBS}")
    print(f"CPU capacity: {CLUSTER_CAPACITY.cpu_cores}")
    print(f"GPU capacity: {CLUSTER_CAPACITY.gpus}")
    print(f"Memory capacity: {CLUSTER_CAPACITY.memory_gib} GiB")

    print("-" * 86)

    print(
        f"{'Algorithm':<12}"
        f"{'Makespan':>12}"
        f"{'Avg Wait':>12}"
        f"{'P95 Wait':>12}"
        f"{'GPU Util':>12}"
        f"{'CPU Util':>12}"
        f"{'Jobs/h':>12}"
    )

    print("-" * 86)

    for name, simulator_class in algorithms.items():

        cluster = Cluster(
            name=f"{name.lower()}-cluster",
            capacity=CLUSTER_CAPACITY,
        )

        simulator = simulator_class(
            cluster,
            jobs,
        )

        results = simulator.run()

        assert len(results) == N_JOBS
        assert cluster.available == cluster.capacity

        metrics = compute_metrics(
            jobs,
            results,
            CLUSTER_CAPACITY,
        )

        print(
            f"{name:<12}"
            f"{metrics.makespan_s:>12d}"
            f"{metrics.average_waiting_s:>12.2f}"
            f"{metrics.p95_waiting_s:>12.2f}"
            f"{metrics.gpu_utilization_pct:>11.2f}%"
            f"{metrics.cpu_utilization_pct:>11.2f}%"
            f"{metrics.throughput_jobs_hour:>12.2f}"
        )

    print("=" * 86)
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()

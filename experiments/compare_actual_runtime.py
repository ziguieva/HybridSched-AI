"""Compare requested and actual execution runtimes."""

import csv
from pathlib import Path

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.metrics import compute_metrics
from hybridsched.resources import Resources
from hybridsched.simulation import FCFSSimulator
from hybridsched.sjf import SJFSimulator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"

N_JOBS = 100

CAPACITY = Resources(
    cpu_cores=32,
    gpus=8,
    memory_gib=128,
)


def load_data():
    with (DATA / "jobs.csv").open() as file:
        features = list(csv.DictReader(file))

    with (DATA / "actual_runtime.csv").open() as file:
        runtimes = list(csv.DictReader(file))

    actual_map = {
        row["job_id"]: int(row["actual_runtime_s"])
        for row in runtimes
    }

    jobs = []

    for row in features[:N_JOBS]:
        jobs.append(
            Job(
                job_id=row["job_id"],
                arrival_s=int(row["arrival_s"]),
                requested_walltime_s=int(
                    row["requested_walltime_s"]
                ),
                resources=Resources(
                    cpu_cores=int(row["cpu_cores"]),
                    gpus=int(row["gpus"]),
                    memory_gib=int(row["memory_gib"]),
                ),
                priority=int(row["priority"]),
            )
        )

    selected_runtimes = {
        job.job_id: actual_map[job.job_id]
        for job in jobs
    }

    return jobs, selected_runtimes


def main():
    jobs, actual_runtimes = load_data()

    algorithms = {
        "FCFS": FCFSSimulator,
        "SJF": SJFSimulator,
        "EASY": EASYBackfillSimulator,
    }

    print("=" * 75)
    print("HybridSched-AI | REQUESTED vs ACTUAL RUNTIME")
    print("=" * 75)

    print(f"Number of jobs: {len(jobs)}")

    print("-" * 75)
    print(
        f"{'Algorithm':<12}"
        f"{'Runtime':<14}"
        f"{'Makespan':>12}"
        f"{'Avg Wait':>14}"
        f"{'GPU Util':>12}"
    )
    print("-" * 75)

    for name, scheduler in algorithms.items():

        for mode in ("requested", "actual"):

            cluster = Cluster(
                name=f"{name}-{mode}",
                capacity=CAPACITY,
            )

            runtimes = (
                actual_runtimes
                if mode == "actual"
                else None
            )

            simulator = scheduler(
                cluster,
                jobs,
                actual_runtimes_s=runtimes,
            )

            results = simulator.run()

            assert len(results) == len(jobs)
            assert cluster.available == cluster.capacity

            if mode == "actual":
                for result in results:
                    assert (
                        result.finish_s - result.start_s
                        == actual_runtimes[result.job_id]
                    )

            metrics = compute_metrics(
                jobs,
                results,
                CAPACITY,
            )

            print(
                f"{name:<12}"
                f"{mode:<14}"
                f"{metrics.makespan_s:>12}"
                f"{metrics.average_waiting_s:>14.2f}"
                f"{metrics.gpu_utilization_pct:>11.2f}%"
            )

    print("=" * 75)
    print("Runtime comparison completed successfully.")


if __name__ == "__main__":
    main()

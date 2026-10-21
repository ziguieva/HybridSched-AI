"""Fairness analysis across HPC job durations."""

import csv
from collections import defaultdict
from math import ceil
from pathlib import Path
from statistics import mean

from compare_schedulers import (
    CLUSTER_CAPACITY,
    generate_workload,
)

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.simulation import FCFSSimulator
from hybridsched.sjf import SJFSimulator


N_EXPERIMENTS = 30
N_JOBS = 200
BASE_SEED = 20261021

ALGORITHMS = {
    "FCFS": FCFSSimulator,
    "SJF": SJFSimulator,
    "EASY": EASYBackfillSimulator,
}

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "fairness_by_duration.csv"
)


def percentile95(values):
    ordered = sorted(values)
    index = ceil(0.95 * len(ordered)) - 1
    return ordered[index]


def main():
    observations = defaultdict(list)
    all_waits = defaultdict(list)

    for experiment in range(N_EXPERIMENTS):
        seed = BASE_SEED + experiment

        jobs = generate_workload(N_JOBS, seed)
        job_map = {job.job_id: job for job in jobs}

        for name, scheduler_class in ALGORITHMS.items():
            cluster = Cluster(
                name=f"{name}-{seed}",
                capacity=CLUSTER_CAPACITY,
            )

            results = scheduler_class(
                cluster,
                jobs,
            ).run()

            assert len(results) == len(jobs)
            assert cluster.available == cluster.capacity

            for result in results:
                job = job_map[result.job_id]
                duration = job.requested_walltime_s

                waiting = result.waiting_s
                turnaround = (
                    result.finish_s - result.arrival_s
                )

                slowdown = turnaround / duration

                observations[(name, duration)].append(
                    (waiting, slowdown)
                )

                all_waits[name].append(waiting)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for name in ALGORITHMS:
        for duration in (120, 300, 600, 900, 1800):
            values = observations[(name, duration)]

            waits = [v[0] for v in values]
            slowdowns = [v[1] for v in values]

            rows.append({
                "algorithm": name,
                "duration_s": duration,
                "jobs": len(values),
                "mean_wait_s": round(mean(waits), 2),
                "p95_wait_s": percentile95(waits),
                "mean_slowdown": round(
                    mean(slowdowns), 2
                ),
            })

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 82)
    print("HybridSched-AI | SCHEDULING FAIRNESS ANALYSIS")
    print("=" * 82)

    print(f"Experiments: {N_EXPERIMENTS}")
    print(f"Jobs per experiment: {N_JOBS}")

    print("-" * 82)

    print(
        f"{'Algorithm':<12}"
        f"{'Duration':>12}"
        f"{'Jobs':>9}"
        f"{'Mean Wait':>14}"
        f"{'P95 Wait':>14}"
        f"{'Slowdown':>13}"
    )

    print("-" * 82)

    for row in rows:
        print(
            f"{row['algorithm']:<12}"
            f"{row['duration_s']:>12}"
            f"{row['jobs']:>9}"
            f"{row['mean_wait_s']:>14.2f}"
            f"{row['p95_wait_s']:>14}"
            f"{row['mean_slowdown']:>13.2f}"
        )

    print("-" * 82)
    print("MAXIMUM WAITING TIME")

    for name in ALGORITHMS:
        print(
            f"{name:<12}: "
            f"{max(all_waits[name])} seconds"
        )

    print("=" * 82)
    print(f"CSV saved: {OUTPUT}")
    print("Fairness analysis completed successfully.")


if __name__ == "__main__":
    main()

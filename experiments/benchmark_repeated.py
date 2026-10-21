"""Repeated, paired HPC scheduling experiments."""

import csv
import random
from pathlib import Path
from statistics import mean, stdev

from compare_schedulers import (
    CLUSTER_CAPACITY,
    generate_workload,
)

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.metrics import compute_metrics
from hybridsched.simulation import FCFSSimulator
from hybridsched.sjf import SJFSimulator


N_JOBS = 200
N_EXPERIMENTS = 30
BASE_SEED = 20261021

ALGORITHMS = {
    "FCFS": FCFSSimulator,
    "SJF": SJFSimulator,
    "EASY": EASYBackfillSimulator,
}

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "benchmark_repeated.csv"
)


def bootstrap_ci95(values, seed=42, n_boot=2000):
    """Percentile bootstrap confidence interval for the mean."""

    rng = random.Random(seed)
    n = len(values)

    estimates = sorted(
        mean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(n_boot)
    )

    return (
        estimates[int(0.025 * n_boot)],
        estimates[int(0.975 * n_boot)],
    )


def evaluate_seed(seed):
    """Evaluate all schedulers on the same workload."""

    jobs = generate_workload(
        n_jobs=N_JOBS,
        seed=seed,
    )

    records = []

    for name, scheduler_class in ALGORITHMS.items():

        cluster = Cluster(
            name=f"{name.lower()}-{seed}",
            capacity=CLUSTER_CAPACITY,
        )

        simulator = scheduler_class(cluster, jobs)
        results = simulator.run()

        assert len(results) == len(jobs)
        assert cluster.available == cluster.capacity
        assert cluster.running_job_ids == ()

        metrics = compute_metrics(
            jobs,
            results,
            CLUSTER_CAPACITY,
        )

        records.append({
            "seed": seed,
            "algorithm": name,
            "jobs": metrics.total_jobs,
            "makespan_s": metrics.makespan_s,
            "avg_wait_s": metrics.average_waiting_s,
            "p95_wait_s": metrics.p95_waiting_s,
            "gpu_util_pct": metrics.gpu_utilization_pct,
            "cpu_util_pct": metrics.cpu_utilization_pct,
            "throughput_jobs_h": metrics.throughput_jobs_hour,
        })

    return records


def main():

    records = []

    for i in range(N_EXPERIMENTS):

        seed = BASE_SEED + i

        records.extend(evaluate_seed(seed))

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )

        writer.writeheader()
        writer.writerows(records)

    print("=" * 85)
    print("HybridSched-AI | MULTI-SEED HPC BENCHMARK")
    print("=" * 85)

    print(f"Experiments: {N_EXPERIMENTS}")
    print(f"Jobs per experiment: {N_JOBS}")
    print(f"Total scheduler runs: {len(records)}")

    print("-" * 85)

    print(
        f"{'Algorithm':<12}"
        f"{'Mean Wait':>14}"
        f"{'SD Wait':>14}"
        f"{'Mean Makespan':>17}"
        f"{'Mean GPU %':>14}"
    )

    print("-" * 85)

    by_algorithm = {
        name: [
            row for row in records
            if row["algorithm"] == name
        ]
        for name in ALGORITHMS
    }

    for name, rows in by_algorithm.items():

        waits = [r["avg_wait_s"] for r in rows]
        makespans = [r["makespan_s"] for r in rows]
        gpu = [r["gpu_util_pct"] for r in rows]

        print(
            f"{name:<12}"
            f"{mean(waits):>14.2f}"
            f"{stdev(waits):>14.2f}"
            f"{mean(makespans):>17.2f}"
            f"{mean(gpu):>14.2f}"
        )

    print("-" * 85)
    print("PAIRED IMPROVEMENTS RELATIVE TO FCFS")
    print("-" * 85)

    fcfs = by_algorithm["FCFS"]

    for name in ("SJF", "EASY"):

        candidate = by_algorithm[name]

        wait_gains = [
            100 * (a["avg_wait_s"] - b["avg_wait_s"])
            / a["avg_wait_s"]
            for a, b in zip(fcfs, candidate)
        ]

        makespan_gains = [
            100 * (a["makespan_s"] - b["makespan_s"])
            / a["makespan_s"]
            for a, b in zip(fcfs, candidate)
        ]

        low_w, high_w = bootstrap_ci95(wait_gains)
        low_m, high_m = bootstrap_ci95(
            makespan_gains,
            seed=43,
        )

        print(f"\n{name} vs FCFS")

        print(
            f"Waiting improvement: "
            f"{mean(wait_gains):.2f}% "
            f"[95% CI: {low_w:.2f}%, {high_w:.2f}%]"
        )

        print(
            f"Makespan improvement: "
            f"{mean(makespan_gains):.2f}% "
            f"[95% CI: {low_m:.2f}%, {high_m:.2f}%]"
        )

    print("\n" + "=" * 85)
    print(f"CSV saved: {OUTPUT}")
    print("Benchmark completed successfully.")
    print("=" * 85)


if __name__ == "__main__":
    main()

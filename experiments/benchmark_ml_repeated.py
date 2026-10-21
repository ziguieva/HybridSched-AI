"""Repeated out-of-sample ML scheduling benchmark."""

import csv
import random
from dataclasses import replace
from statistics import mean, median, stdev

from xgboost import XGBRegressor

from benchmark_ml_scheduling import (
    CAPACITY,
    DATA,
    MODEL_PATH,
    RESULTS,
)
from compare_schedulers import generate_workload
from create_runtime_dataset import generate_actual_runtime

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.metrics import compute_metrics
from hybridsched.predictive import (
    PredictedEASYBackfillSimulator,
    PredictedSJFSimulator,
)
from hybridsched.simulation import FCFSSimulator
from hybridsched.sjf import SJFSimulator


N_SCENARIOS = 30
N_JOBS = 200

WORKLOAD_SEED = 20271000
RUNTIME_SEED = 20281000
FUTURE_OFFSET_S = 42000

OUTPUT = RESULTS / "benchmark_ml_repeated.csv"

POLICIES = {
    "FCFS": (FCFSSimulator, None),
    "SJF": (SJFSimulator, None),
    "EASY": (EASYBackfillSimulator, None),
    "SJF-Ratio": (PredictedSJFSimulator, "ratio"),
    "SJF-XGB": (PredictedSJFSimulator, "xgb"),
    "EASY-XGB": (
        PredictedEASYBackfillSimulator,
        "xgb",
    ),
}


def bootstrap_ci(values, seed=42, rounds=2000):
    rng = random.Random(seed)
    n = len(values)

    estimates = sorted(
        mean(
            values[rng.randrange(n)]
            for _ in range(n)
        )
        for _ in range(rounds)
    )

    return (
        estimates[int(rounds * 0.025)],
        estimates[int(rounds * 0.975)],
    )


def main():
    with (DATA / "train.csv").open(newline="") as file:
        train = list(csv.DictReader(file))

    historical_ratio = median(
        float(row["actual_runtime_s"])
        / float(row["requested_walltime_s"])
        for row in train
    )

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    records = []

    errors = {
        "Requested": [],
        "Ratio": [],
        "XGBoost": [],
    }

    for index in range(N_SCENARIOS):
        raw_jobs = generate_workload(
            N_JOBS,
            WORKLOAD_SEED + index,
        )

        rng = random.Random(RUNTIME_SEED + index)

        actual = {
            job.job_id: generate_actual_runtime(job, rng)
            for job in raw_jobs
        }

        origin = min(
            job.arrival_s for job in raw_jobs
        )

        jobs = [
            replace(
                job,
                arrival_s=job.arrival_s - origin,
            )
            for job in raw_jobs
        ]

        # Future timestamps for ML input.
        # Simulation arrival times remain relative.
        features = [
            [
                job.requested_walltime_s,
                job.resources.cpu_cores,
                job.resources.gpus,
                job.resources.memory_gib,
                job.priority,
                FUTURE_OFFSET_S + job.arrival_s,
            ]
            for job in raw_jobs
        ]

        raw_predictions = model.predict(features)

        xgb_predictions = {
            job.job_id: max(1, round(float(value)))
            for job, value in zip(
                raw_jobs,
                raw_predictions,
            )
        }

        ratio_predictions = {
            job.job_id: max(
                1,
                round(
                    job.requested_walltime_s
                    * historical_ratio
                ),
            )
            for job in raw_jobs
        }

        predictions = {
            "ratio": ratio_predictions,
            "xgb": xgb_predictions,
        }

        for job in raw_jobs:
            truth = actual[job.job_id]

            errors["Requested"].append(
                abs(job.requested_walltime_s - truth)
            )

            errors["Ratio"].append(
                abs(ratio_predictions[job.job_id] - truth)
            )

            errors["XGBoost"].append(
                abs(xgb_predictions[job.job_id] - truth)
            )

        for name, (scheduler, prediction_type) in POLICIES.items():
            cluster = Cluster(
                f"{name}-{index}",
                CAPACITY,
            )

            if prediction_type is None:
                simulator = scheduler(
                    cluster,
                    jobs,
                    actual_runtimes_s=actual,
                )
            else:
                simulator = scheduler(
                    cluster,
                    jobs,
                    runtime_predictions_s=predictions[
                        prediction_type
                    ],
                    actual_runtimes_s=actual,
                )

            results = simulator.run()

            assert len(results) == len(jobs)
            assert cluster.available == cluster.capacity

            for result in results:
                assert (
                    result.finish_s - result.start_s
                    == actual[result.job_id]
                )

            metrics = compute_metrics(
                jobs,
                results,
                CAPACITY,
            )

            records.append({
                "scenario": index + 1,
                "algorithm": name,
                "makespan_s": metrics.makespan_s,
                "avg_wait_s": metrics.average_waiting_s,
                "p95_wait_s": metrics.p95_waiting_s,
                "gpu_util_pct": metrics.gpu_utilization_pct,
            })

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )
        writer.writeheader()
        writer.writerows(records)

    print("=" * 79)
    print("HybridSched-AI | MULTI-SEED ML SCHEDULING BENCHMARK")
    print("=" * 79)

    print(f"Scenarios: {N_SCENARIOS}")
    print(f"Jobs per scenario: {N_JOBS}")
    print(f"Total scheduler runs: {len(records)}")

    print("-" * 79)

    print(
        f"{'Algorithm':<13}"
        f"{'Mean Wait':>15}"
        f"{'SD Wait':>13}"
        f"{'Makespan':>16}"
        f"{'GPU %':>12}"
    )

    print("-" * 79)

    grouped = {
        name: [
            row for row in records
            if row["algorithm"] == name
        ]
        for name in POLICIES
    }

    for name, rows in grouped.items():
        print(
            f"{name:<13}"
            f"{mean(r['avg_wait_s'] for r in rows):>15.2f}"
            f"{stdev(r['avg_wait_s'] for r in rows):>13.2f}"
            f"{mean(r['makespan_s'] for r in rows):>16.2f}"
            f"{mean(r['gpu_util_pct'] for r in rows):>12.2f}"
        )

    print("-" * 79)
    print("PAIRED IMPROVEMENTS: WAITING TIME")
    print("-" * 79)

    comparisons = [
        ("SJF-XGB", "SJF"),
        ("SJF-XGB", "SJF-Ratio"),
        ("EASY-XGB", "EASY"),
    ]

    for candidate, baseline in comparisons:
        gains = [
            100 * (
                a["avg_wait_s"] - b["avg_wait_s"]
            ) / a["avg_wait_s"]
            for a, b in zip(
                grouped[baseline],
                grouped[candidate],
            )
        ]

        low, high = bootstrap_ci(gains)
        wins = sum(gain > 0 for gain in gains)

        print(f"\n{candidate} vs {baseline}")
        print(f"Mean improvement: {mean(gains):+.2f}%")
        print(f"95% CI: [{low:+.2f}%, {high:+.2f}%]")
        print(f"Improved scenarios: {wins}/{N_SCENARIOS}")

    print("-" * 79)
    print("RUNTIME PREDICTION MAE")
    print("-" * 79)

    for name, values in errors.items():
        print(f"{name:<13}: {mean(values):.2f} s")

    print("-" * 79)
    print(f"CSV saved: {OUTPUT}")
    print("Benchmark completed successfully.")
    print("=" * 79)


if __name__ == "__main__":
    main()

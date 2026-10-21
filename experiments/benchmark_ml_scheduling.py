"""Evaluate ML-assisted scheduling on held-out HPC jobs."""

import csv
from pathlib import Path
from statistics import median

from xgboost import XGBRegressor

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.metrics import compute_metrics
from hybridsched.predictive import (
    PredictedEASYBackfillSimulator,
    PredictedSJFSimulator,
)
from hybridsched.resources import Resources
from hybridsched.simulation import FCFSSimulator
from hybridsched.sjf import SJFSimulator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

MODEL_PATH = RESULTS / "runtime_xgboost.json"
OUTPUT = RESULTS / "benchmark_ml_scheduling.csv"

CAPACITY = Resources(
    cpu_cores=32,
    gpus=8,
    memory_gib=128,
)

FEATURES = [
    "requested_walltime_s",
    "cpu_cores",
    "gpus",
    "memory_gib",
    "priority",
    "arrival_s",
]


def load_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def prepare_test_workload():
    rows = load_csv(DATA / "test.csv")

    if not rows:
        raise ValueError("Test dataset is empty")

    # Rebase time to avoid counting the training
    # period as idle time in benchmark metrics.
    origin = min(int(row["arrival_s"]) for row in rows)

    jobs = []
    actual = {}

    for row in rows:
        job_id = row["job_id"]

        jobs.append(
            Job(
                job_id=job_id,
                arrival_s=int(row["arrival_s"]) - origin,
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

        actual[job_id] = int(
            row["actual_runtime_s"]
        )

    return rows, jobs, actual


def prepare_predictions(rows):
    train = load_csv(DATA / "train.csv")

    # Historical ratio estimated using training only.
    ratio = median(
        float(row["actual_runtime_s"])
        / float(row["requested_walltime_s"])
        for row in train
    )

    historical = {
        row["job_id"]: max(
            1,
            round(
                float(row["requested_walltime_s"])
                * ratio
            ),
        )
        for row in rows
    }

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    X_test = [
        [
            float(row[column])
            for column in FEATURES
        ]
        for row in rows
    ]

    raw_predictions = model.predict(X_test)

    xgboost_predictions = {
        row["job_id"]: max(1, round(float(pred)))
        for row, pred in zip(rows, raw_predictions)
    }

    return historical, xgboost_predictions


def main():
    rows, jobs, actual = prepare_test_workload()

    historical, xgboost_predictions = (
        prepare_predictions(rows)
    )

    policies = [
        ("FCFS", FCFSSimulator, None),
        ("SJF", SJFSimulator, None),
        ("EASY", EASYBackfillSimulator, None),
        (
            "SJF-Ratio",
            PredictedSJFSimulator,
            historical,
        ),
        (
            "SJF-XGB",
            PredictedSJFSimulator,
            xgboost_predictions,
        ),
        (
            "EASY-XGB",
            PredictedEASYBackfillSimulator,
            xgboost_predictions,
        ),
    ]

    records = []
    metrics_by_name = {}

    print("=" * 92)
    print("HybridSched-AI | ML-ASSISTED SCHEDULING BENCHMARK")
    print("=" * 92)

    print(f"Test jobs: {len(jobs)}")
    print("Runtime source: held-out synthetic observations")
    print("Prediction model: trained XGBoost")

    print("-" * 92)
    print(
        f"{'Algorithm':<14}"
        f"{'Makespan':>12}"
        f"{'Avg Wait':>13}"
        f"{'P95 Wait':>13}"
        f"{'GPU Util':>13}"
        f"{'Jobs/h':>13}"
    )
    print("-" * 92)

    for name, scheduler_class, predictions in policies:
        cluster = Cluster(
            name=f"{name.lower()}-test",
            capacity=CAPACITY,
        )

        if predictions is None:
            simulator = scheduler_class(
                cluster,
                jobs,
                actual_runtimes_s=actual,
            )
        else:
            simulator = scheduler_class(
                cluster,
                jobs,
                runtime_predictions_s=predictions,
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

        metrics_by_name[name] = metrics

        records.append({
            "algorithm": name,
            "jobs": metrics.total_jobs,
            "makespan_s": metrics.makespan_s,
            "mean_wait_s": metrics.average_waiting_s,
            "p95_wait_s": metrics.p95_waiting_s,
            "gpu_util_pct": metrics.gpu_utilization_pct,
            "throughput_jobs_h":
                metrics.throughput_jobs_hour,
        })

        print(
            f"{name:<14}"
            f"{metrics.makespan_s:>12d}"
            f"{metrics.average_waiting_s:>13.2f}"
            f"{metrics.p95_waiting_s:>13.2f}"
            f"{metrics.gpu_utilization_pct:>12.2f}%"
            f"{metrics.throughput_jobs_hour:>13.2f}"
        )

    print("-" * 92)
    print("COMPARISON AGAINST MATCHING BASELINES")

    comparisons = [
        ("SJF-XGB", "SJF"),
        ("SJF-XGB", "SJF-Ratio"),
        ("EASY-XGB", "EASY"),
    ]

    for candidate_name, baseline_name in comparisons:
        candidate = metrics_by_name[candidate_name]
        baseline = metrics_by_name[baseline_name]

        improvement = (
            100
            * (
                baseline.average_waiting_s
                - candidate.average_waiting_s
            )
            / baseline.average_waiting_s
        )

        print(
            f"{candidate_name} vs {baseline_name}: "
            f"{improvement:+.2f}% waiting reduction"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )

        writer.writeheader()
        writer.writerows(records)

    print("-" * 92)
    print(f"Results saved: {OUTPUT}")
    print("ML scheduling benchmark completed.")
    print("=" * 92)


if __name__ == "__main__":
    main()

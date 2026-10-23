"""Held-out online scheduling with real synthetic runtimes."""

import csv
from pathlib import Path
from statistics import mean

from hybridsched.infrastructure import default_infrastructure
from hybridsched.job import Job
from hybridsched.resources import Resources
from hybridsched.ml_predictor import MLRuntimePredictor
from hybridsched.ml_dynamic import MLDynamicSimulator


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_ml_dynamic.csv"

MODES = ("requested", "point", "p50", "p90")


def load_test_data():
    with (ROOT / "data/synthetic/jobs.csv").open(
        newline=""
    ) as f:
        jobs_raw = list(csv.DictReader(f))

    with (ROOT / "data/synthetic/actual_runtime.csv").open(
        newline=""
    ) as f:
        runtime_raw = list(csv.DictReader(f))

    actual = {
        row["job_id"]: float(row["actual_runtime_s"])
        for row in runtime_raw
    }

    # Chronological 70/15/15 split established at training.
    # Never use train or validation rows for test metrics.
    if len(jobs_raw) != 1000:
        raise ValueError("Expected 1000 dataset jobs")

    test_rows = jobs_raw[850:]

    if len(test_rows) != 150:
        raise ValueError("Unexpected test size")

    jobs = []

    for row in test_rows:
        jobs.append(
            Job(
                job_id=row["job_id"],
                arrival_s=int(float(row["arrival_s"])),
                requested_walltime_s=int(
                    float(row["requested_walltime_s"])
                ),
                resources=Resources(
                    cpu_cores=int(float(row["cpu_cores"])),
                    gpus=int(float(row["gpus"])),
                    memory_gib=int(float(row["memory_gib"])),
                ),
                priority=int(float(row["priority"])),
            )
        )

    if any(j.job_id not in actual for j in jobs):
        raise ValueError("Missing actual runtime")

    return jobs, actual


def main():
    jobs, actual = load_test_data()
    predictor = MLRuntimePredictor()

    # Predictions are evaluated on the same held-out jobs.
    estimates = predictor.predict_many(jobs)

    print("=" * 90)
    print("HybridSched-AI | ML RUNTIME PREDICTION")
    print("=" * 90)

    for label in MODES:
        predicted = [
            (
                j.requested_walltime_s
                if label == "requested"
                else getattr(estimates[j.job_id], label + "_s")
                if label != "point"
                else estimates[j.job_id].point_s
            )
            for j in jobs
        ]

        mae = mean(
            abs(pred - actual[j.job_id])
            for pred, j in zip(predicted, jobs)
        )

        print(f"{label:<12} MAE={mae:.2f}s")

    rows = []

    print()
    print("=" * 90)
    print("HybridSched-AI | ML-AWARE DYNAMIC BENCHMARK")
    print("=" * 90)

    print(
        f"{'Mode':<14}"
        f"{'Makespan':>12}"
        f"{'Mean Wait':>13}"
        f"{'Cost':>14}"
        f"{'Cloud Jobs':>13}"
    )
    print("-" * 90)

    for mode in MODES:
        simulator = MLDynamicSimulator(
            jobs,
            default_infrastructure(),
            actual_runtimes=actual,
            predictor=predictor,
            mode=mode,
        )

        results = simulator.run()

        if len(results) != len(jobs):
            raise RuntimeError("Incomplete run")

        row = {
            "mode": mode,
            "makespan_s": max(
                r.finish_s for r in results
            ),
            "mean_wait_s": round(
                mean(r.waiting_s for r in results), 2
            ),
            "cost_units": sum(
                r.cost_units for r in results
            ),
            "cloud_jobs": sum(
                r.cluster == "gcp-gpu"
                for r in results
            ),
        }

        rows.append(row)

        print(
            f"{mode:<14}"
            f"{row['makespan_s']:>12}"
            f"{row['mean_wait_s']:>13.2f}"
            f"{row['cost_units']:>14}"
            f"{row['cloud_jobs']:>13}"
        )

    OUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 90)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

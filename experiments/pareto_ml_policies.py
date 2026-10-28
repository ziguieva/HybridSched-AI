"""Pareto analysis of ML deadline scheduling policies."""

import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean

from experiments.benchmark_ml_dynamic import load_test_data
from hybridsched.conditional_calibration import (
    CalibratedRuntimePredictor,
)
from hybridsched.infrastructure import default_infrastructure
from hybridsched.ml_deadline import MLDeadlineSimulator
from hybridsched.ml_predictor import MLRuntimePredictor


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = ("cost_units", "tardiness_s", "makespan_s")


def dominates(a, b):
    return (
        all(a[k] <= b[k] for k in METRICS)
        and any(a[k] < b[k] for k in METRICS)
    )


def main():
    source = RESULTS / "uncertainty_aware_benchmark.csv"

    with source.open(newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for field in (
            "makespan_s",
            "mean_wait_s",
            "late_jobs",
            "tardiness_s",
            "cost_units",
            "cloud_jobs",
        ):
            row[field] = float(row[field])

    with (
        RESULTS / "conditional_p90_corrections.json"
    ).open() as f:
        corrections = json.load(f)["corrections_s"]

    predictor = CalibratedRuntimePredictor(
        MLRuntimePredictor(), corrections
    )

    original, actual = load_test_data()
    jobs = [
        replace(
            job,
            deadline_s=(
                job.arrival_s
                + job.requested_walltime_s
                + 300
            ),
        )
        for job in original
    ]

    sim = MLDeadlineSimulator(
        jobs,
        default_infrastructure(),
        actual_runtimes=actual,
        predictor=predictor,
        mode="p90",
    )
    results = sim.run()

    rows.append({
        "policy": "CAL_P90",
        "makespan_s": max(r.finish_s for r in results),
        "mean_wait_s": mean(r.waiting_s for r in results),
        "late_jobs": sum(r.tardiness_s > 0 for r in results),
        "tardiness_s": sum(r.tardiness_s for r in results),
        "cost_units": sum(r.cost_units for r in results),
        "cloud_jobs": sum(
            r.cluster == "gcp-gpu" for r in results
        ),
    })

    for row in rows:
        row["pareto_optimal"] = not any(
            dominates(other, row)
            for other in rows
            if other is not row
        )

    out = RESULTS / "ml_pareto.csv"

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=== STEP 19.2 | ML PARETO FRONT ===")
    print(
        f"{'Policy':<15}{'Cost':>12}"
        f"{'Tardiness':>13}{'Makespan':>12}"
        f"{'Late':>8}{'Pareto':>10}"
    )

    for row in rows:
        print(
            f"{row['policy']:<15}"
            f"{row['cost_units']:>12.0f}"
            f"{row['tardiness_s']:>13.0f}"
            f"{row['makespan_s']:>12.0f}"
            f"{row['late_jobs']:>8.0f}"
            f"{str(row['pareto_optimal']):>10}"
        )

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

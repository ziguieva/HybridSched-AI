"""Deadline-margin sensitivity for established scheduling policies."""

import csv
from dataclasses import replace
from pathlib import Path
from statistics import mean

from experiments.benchmark_ml_dynamic import load_test_data
from experiments.benchmark_multiobjective import run_policy
from experiments.decision_shift_benchmark import (
    SCENARIOS,
    transform_workload,
)
from hybridsched.ml_predictor import MLRuntimePredictor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "step23_deadline_sensitivity.csv"

POLICIES = ("REQUESTED", "P90", "SLACK", "MULTI")
MARGINS = (0, 150, 300, 600)
SEEDS = range(23001, 23006)


def apply_deadline_margin(jobs, margin):
    if margin < 0:
        raise ValueError("Negative deadline margin")

    return [
        replace(
            job,
            deadline_s=(
                job.arrival_s
                + job.requested_walltime_s
                + margin
            ),
        )
        for job in jobs
    ]


def main():
    original, actual = load_test_data()
    predictor = MLRuntimePredictor()
    rows = []

    for scenario in SCENARIOS:
        for seed in SEEDS:
            shifted, disturbed = transform_workload(
                original, actual, scenario, seed
            )

            for margin in MARGINS:
                jobs = apply_deadline_margin(
                    shifted, margin
                )

                for policy in POLICIES:
                    result = run_policy(
                        jobs,
                        disturbed,
                        predictor,
                        policy,
                    )

                    rows.append({
                        "scenario": scenario,
                        "seed": seed,
                        "deadline_margin_s": margin,
                        "policy": policy,
                        **result,
                    })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=== STEP 23.3 | DEADLINE SENSITIVITY ===")

    for scenario in SCENARIOS:
        print(f"\n{scenario}")

        for margin in MARGINS:
            print(f"\nMargin: +{margin}s")
            print(
                f"{'Policy':<13}{'Late':>9}"
                f"{'Tardiness':>15}{'Cost':>15}"
            )

            for policy in POLICIES:
                subset = [
                    r for r in rows
                    if r["scenario"] == scenario
                    and r["deadline_margin_s"] == margin
                    and r["policy"] == policy
                ]

                print(
                    f"{policy:<13}"
                    f"{mean(r['late_jobs'] for r in subset):>9.2f}"
                    f"{mean(r['tardiness_s'] for r in subset):>15.1f}"
                    f"{mean(r['cost_units'] for r in subset):>15.1f}"
                )

    print(f"\nSaved: {OUTPUT}")
    print(f"Simulations: {len(rows)}")


if __name__ == "__main__":
    main()

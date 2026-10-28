"""Paired 30-scenario synthetic runtime stress-test."""

import csv
import json
import random
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

from experiments.benchmark_ml_dynamic import load_test_data
from hybridsched.conditional_calibration import (
    CalibratedRuntimePredictor,
)
from hybridsched.infrastructure import default_infrastructure
from hybridsched.ml_deadline import MLDeadlineSimulator
from hybridsched.ml_predictor import MLRuntimePredictor


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = range(19001, 19031)
POLICIES = ("REQUESTED", "P50", "P90", "CAL_P90")
METRICS = ("tardiness_s", "cost_units", "makespan_s", "late_jobs")


def evaluate(jobs, actual, predictor, mode):
    sim = MLDeadlineSimulator(
        jobs,
        default_infrastructure(),
        actual_runtimes=actual,
        predictor=predictor,
        mode=mode,
    )
    outputs = sim.run()

    if len(outputs) != len(jobs):
        raise RuntimeError("Incomplete simulation")

    return {
        "makespan_s": max(r.finish_s for r in outputs),
        "mean_wait_s": mean(r.waiting_s for r in outputs),
        "late_jobs": sum(r.tardiness_s > 0 for r in outputs),
        "tardiness_s": sum(r.tardiness_s for r in outputs),
        "cost_units": sum(r.cost_units for r in outputs),
    }


def mean_ci(values):
    average = mean(values)
    if len(values) < 2:
        return average, average, average

    half_width = 1.96 * stdev(values) / (len(values) ** 0.5)
    return average, average - half_width, average + half_width


def main():
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

    with (
        RESULTS / "conditional_p90_corrections.json"
    ).open() as f:
        corrections = json.load(f)["corrections_s"]

    base = MLRuntimePredictor()
    calibrated = CalibratedRuntimePredictor(base, corrections)

    configs = {
        "REQUESTED": (None, "requested"),
        "P50": (base, "p50"),
        "P90": (base, "p90"),
        "CAL_P90": (calibrated, "p90"),
    }

    rows = []

    for seed in SEEDS:
        rng = random.Random(seed)
        disturbed = {
            job.job_id: max(
                1.0,
                actual[job.job_id] * rng.uniform(0.8, 1.2),
            )
            for job in jobs
        }

        for policy, (predictor, mode) in configs.items():
            metrics = evaluate(
                jobs, disturbed, predictor, mode
            )
            rows.append({
                "seed": seed,
                "policy": policy,
                **metrics,
            })

    out = RESULTS / "ml_stress_30seeds.csv"

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=== STEP 19.3 | 30-SCENARIO ML STRESS TEST ===")
    print(
        f"{'Policy':<15}{'Late':>10}"
        f"{'Tardiness':>15}{'Cost':>15}"
        f"{'Makespan':>13}"
    )

    for policy in POLICIES:
        group = [r for r in rows if r["policy"] == policy]
        print(
            f"{policy:<15}"
            f"{mean(r['late_jobs'] for r in group):>10.2f}"
            f"{mean(r['tardiness_s'] for r in group):>15.1f}"
            f"{mean(r['cost_units'] for r in group):>15.1f}"
            f"{mean(r['makespan_s'] for r in group):>13.1f}"
        )

    print("\nPAIRED DIFFERENCES VS P90 (95% normal-approx CI)")
    by_key = {
        (r["seed"], r["policy"]): r for r in rows
    }

    for policy in ("REQUESTED", "P50", "CAL_P90"):
        print(f"\n{policy} - P90")

        for metric in METRICS:
            differences = [
                by_key[(seed, policy)][metric]
                - by_key[(seed, "P90")][metric]
                for seed in SEEDS
            ]
            avg, low, high = mean_ci(differences)
            print(
                f"{metric:<15}"
                f"mean={avg:>11.2f} "
                f"CI=[{low:.2f}, {high:.2f}]"
            )

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

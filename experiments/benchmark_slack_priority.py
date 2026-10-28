"""Paired comparison of deadline priority policies."""

import csv
from pathlib import Path
from statistics import mean

from experiments.benchmark_ml_dynamic import load_test_data
from experiments.decision_shift_benchmark import (
    SCENARIOS,
    transform_workload,
)
from hybridsched.decision_focused import DecisionFocusedSimulator
from hybridsched.infrastructure import default_infrastructure
from hybridsched.ml_deadline import MLDeadlineSimulator
from hybridsched.ml_predictor import MLRuntimePredictor
from hybridsched.slack_priority import SlackPrioritySimulator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "slack_priority_benchmark.csv"

POLICIES = ("REQUESTED", "P90", "DECISION", "SLACK")


def run_policy(jobs, actual, policy, predictor):
    clusters = default_infrastructure()

    if policy == "REQUESTED":
        sim = MLDeadlineSimulator(
            jobs, clusters,
            actual_runtimes=actual,
            mode="requested",
        )
    elif policy == "P90":
        sim = MLDeadlineSimulator(
            jobs, clusters,
            actual_runtimes=actual,
            predictor=predictor,
            mode="p90",
        )
    elif policy == "DECISION":
        sim = DecisionFocusedSimulator(
            jobs, clusters, actual, predictor
        )
    elif policy == "SLACK":
        sim = SlackPrioritySimulator(
            jobs, clusters, actual, predictor
        )
    else:
        raise ValueError(policy)

    results = sim.run()

    if len(results) != len(jobs):
        raise RuntimeError("Incomplete simulation")

    return {
        "late_jobs": sum(r.tardiness_s > 0 for r in results),
        "tardiness_s": sum(r.tardiness_s for r in results),
        "mean_wait_s": mean(r.waiting_s for r in results),
        "makespan_s": max(r.finish_s for r in results),
        "cost_units": sum(r.cost_units for r in results),
    }


def main():
    jobs, actual = load_test_data()
    predictor = MLRuntimePredictor()
    rows = []

    for scenario in SCENARIOS:
        for seed in range(20001, 20011):
            shifted, actual_shifted = transform_workload(
                jobs, actual, scenario, seed
            )

            for policy in POLICIES:
                metrics = run_policy(
                    shifted, actual_shifted,
                    policy, predictor
                )
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "policy": policy,
                    **metrics,
                })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=== STEP 21.3 | SLACK PRIORITY BENCHMARK ===")

    for scenario in SCENARIOS:
        print(f"\n{scenario}")
        print(
            f"{'Policy':<14}{'Late':>9}"
            f"{'Tardiness':>15}{'Wait':>12}"
            f"{'Cost':>15}{'Makespan':>13}"
        )

        for policy in POLICIES:
            subset = [
                row for row in rows
                if row["scenario"] == scenario
                and row["policy"] == policy
            ]

            print(
                f"{policy:<14}"
                f"{mean(r['late_jobs'] for r in subset):>9.2f}"
                f"{mean(r['tardiness_s'] for r in subset):>15.1f}"
                f"{mean(r['mean_wait_s'] for r in subset):>12.1f}"
                f"{mean(r['cost_units'] for r in subset):>15.1f}"
                f"{mean(r['makespan_s'] for r in subset):>13.1f}"
            )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()

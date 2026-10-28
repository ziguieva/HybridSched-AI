"""Paired multiobjective scheduler benchmark and Pareto analysis."""

import csv
from pathlib import Path
from statistics import mean

from experiments.benchmark_ml_dynamic import load_test_data
from experiments.decision_shift_benchmark import (
    SCENARIOS,
    transform_workload,
)
from hybridsched.infrastructure import default_infrastructure
from hybridsched.ml_deadline import MLDeadlineSimulator
from hybridsched.ml_predictor import MLRuntimePredictor
from hybridsched.slack_priority import SlackPrioritySimulator
from hybridsched.multiobjective_deadline import (
    MultiobjectiveDeadlineSimulator,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "multiobjective_benchmark.csv"
PARETO_OUTPUT = ROOT / "results" / "multiobjective_pareto.csv"

POLICIES = ("REQUESTED", "P90", "SLACK", "MULTI")
OBJECTIVES = ("late_jobs", "tardiness_s", "cost_units")


def run_policy(jobs, actual, predictor, policy):
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
    elif policy == "SLACK":
        sim = SlackPrioritySimulator(
            jobs, clusters, actual, predictor,
        )
    elif policy == "MULTI":
        sim = MultiobjectiveDeadlineSimulator(
            jobs, clusters, actual, predictor,
        )
    else:
        raise ValueError(policy)

    results = sim.run()

    if len(results) != len(jobs):
        raise RuntimeError("Incomplete scheduling")

    return {
        "late_jobs": sum(r.tardiness_s > 0 for r in results),
        "tardiness_s": sum(r.tardiness_s for r in results),
        "mean_wait_s": mean(r.waiting_s for r in results),
        "cost_units": sum(r.cost_units for r in results),
        "makespan_s": max(r.finish_s for r in results),
    }


def dominates(a, b):
    return (
        all(a[k] <= b[k] for k in OBJECTIVES)
        and any(a[k] < b[k] for k in OBJECTIVES)
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    base_jobs, base_actual = load_test_data()
    predictor = MLRuntimePredictor()
    rows = []
    summary = []

    for scenario in SCENARIOS:
        for seed in range(20001, 20011):
            jobs, actual = transform_workload(
                base_jobs, base_actual, scenario, seed
            )

            for policy in POLICIES:
                metrics = run_policy(
                    jobs, actual, predictor, policy
                )
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "policy": policy,
                    **metrics,
                })

        for policy in POLICIES:
            group = [
                r for r in rows
                if r["scenario"] == scenario
                and r["policy"] == policy
            ]

            summary.append({
                "scenario": scenario,
                "policy": policy,
                **{
                    key: mean(r[key] for r in group)
                    for key in (
                        "late_jobs",
                        "tardiness_s",
                        "mean_wait_s",
                        "cost_units",
                        "makespan_s",
                    )
                },
            })

    for item in summary:
        peers = [
            x for x in summary
            if x["scenario"] == item["scenario"]
            and x is not item
        ]
        item["pareto"] = not any(
            dominates(other, item)
            for other in peers
        )

    write_csv(OUTPUT, rows)
    write_csv(PARETO_OUTPUT, summary)

    print("=== STEP 22.3 | MULTIOBJECTIVE BENCHMARK ===")

    for scenario in SCENARIOS:
        print(f"\n{scenario}")
        print(
            f"{'Policy':<13}{'Late':>8}"
            f"{'Tardiness':>14}{'Wait':>11}"
            f"{'Cost':>13}{'Makespan':>12}"
            f"{'Pareto':>9}"
        )

        for r in summary:
            if r["scenario"] != scenario:
                continue

            print(
                f"{r['policy']:<13}"
                f"{r['late_jobs']:>8.2f}"
                f"{r['tardiness_s']:>14.1f}"
                f"{r['mean_wait_s']:>11.1f}"
                f"{r['cost_units']:>13.1f}"
                f"{r['makespan_s']:>12.1f}"
                f"{str(r['pareto']):>9}"
            )

    print(f"\nSaved: {OUTPUT}")
    print(f"Saved: {PARETO_OUTPUT}")


if __name__ == "__main__":
    main()

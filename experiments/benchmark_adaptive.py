"""Reproducible adaptive scheduling evaluation."""

import csv
import time
from pathlib import Path
from statistics import mean

from stress_test_dynamic import generate_workload

from hybridsched.adaptive import AdaptiveHybridSimulator
from hybridsched.dynamic import DynamicHybridSimulator
from hybridsched.infrastructure import default_infrastructure


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "benchmark_adaptive.csv"

SEEDS = (42, 123, 2026)
SIZES = (50, 100, 200)


def evaluate(jobs, policy):
    if policy == "ADAPTIVE":
        simulator = AdaptiveHybridSimulator(
            jobs,
            default_infrastructure(),
            time_limit_s=0.2,
            cp_threshold=8,
            max_cp_calls=4,
        )
    else:
        simulator = DynamicHybridSimulator(
            jobs,
            default_infrastructure(),
            policy=policy.lower(),
            time_limit_s=0.2,
        )

    start = time.perf_counter()
    results = simulator.run()
    wall = time.perf_counter() - start

    if len(results) != len(jobs):
        raise AssertionError("Incomplete scheduling")

    return {
        "makespan": max(r.finish_s for r in results),
        "mean_wait": round(
            mean(r.waiting_s for r in results), 2
        ),
        "late_jobs": sum(
            r.tardiness_s > 0 for r in results
        ),
        "total_tardiness": sum(
            r.tardiness_s for r in results
        ),
        "cost_units": sum(
            r.cost_units for r in results
        ),
        "cloud_jobs": sum(
            r.cluster == "gcp-gpu" for r in results
        ),
        "wall_s": round(wall, 3),
        "cp_calls": simulator.solver_calls,
        "cp_solver_s": round(
            simulator.solver_wall_s, 3
        ),
    }


def main():
    rows = []

    print("=" * 108)
    print("HybridSched-AI | ADAPTIVE SCHEDULING")
    print("=" * 108)

    print(
        f"{'Jobs':>6}"
        f"{'Seed':>8}"
        f"{'Policy':<13}"
        f"{'Makespan':>12}"
        f"{'Wait':>12}"
        f"{'Late':>8}"
        f"{'Cost':>13}"
        f"{'CP calls':>11}"
        f"{'Wall(s)':>12}"
    )
    print("-" * 108)

    for count in SIZES:
        for seed in SEEDS:
            jobs = generate_workload(count, seed)

            policies = ["FCFS", "SJF", "ADAPTIVE"]

            # Rolling-CP is expensive: one separate
            # reference run, not a full repeated study.
            if count == 50 and seed == SEEDS[0]:
                policies.append("ROLLING")

            for policy in policies:
                metrics = evaluate(jobs, policy)

                row = {
                    "jobs": count,
                    "seed": seed,
                    "policy": policy,
                    **metrics,
                }

                rows.append(row)

                print(
                    f"{count:>6}"
                    f"{seed:>8}"
                    f"{policy:<13}"
                    f"{row['makespan']:>12}"
                    f"{row['mean_wait']:>12.2f}"
                    f"{row['late_jobs']:>8}"
                    f"{row['cost_units']:>13}"
                    f"{row['cp_calls']:>11}"
                    f"{row['wall_s']:>12.3f}",
                    flush=True,
                )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 108)
    print(f"Saved: {OUTPUT}")

    print()
    print("MEAN PERFORMANCE BY WORKLOAD")
    print("-" * 84)
    print(
        f"{'Jobs':>6}"
        f"{'Policy':<13}"
        f"{'Mean Makespan':>17}"
        f"{'Mean Late':>13}"
        f"{'Mean Cost':>16}"
        f"{'Mean Wall':>14}"
    )

    for count in SIZES:
        for policy in ("FCFS", "SJF", "ADAPTIVE"):
            selected = [
                row for row in rows
                if row["jobs"] == count
                and row["policy"] == policy
            ]

            print(
                f"{count:>6}"
                f"{policy:<13}"
                f"{mean(r['makespan'] for r in selected):>17.1f}"
                f"{mean(r['late_jobs'] for r in selected):>13.1f}"
                f"{mean(r['cost_units'] for r in selected):>16.1f}"
                f"{mean(r['wall_s'] for r in selected):>14.3f}"
            )

    print("Adaptive experiment completed.")


if __name__ == "__main__":
    main()

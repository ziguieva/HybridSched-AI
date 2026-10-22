"""Multi-seed comparison of deadline-aware policies."""

import csv
import time
from pathlib import Path
from statistics import mean

from stress_test_dynamic import generate_workload

from hybridsched.adaptive import AdaptiveHybridSimulator
from hybridsched.deadline_aware import DeadlineAwareSimulator
from hybridsched.dynamic import DynamicHybridSimulator
from hybridsched.infrastructure import default_infrastructure


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "benchmark_deadline_aware.csv"

SIZES = (50, 100, 200)
SEEDS = (42, 123, 2026)

POLICIES = (
    "SJF",
    "ADAPTIVE",
    "EDF",
    "LST",
    "RESOURCE",
)


def build_simulator(policy, jobs):
    clusters = default_infrastructure()

    if policy == "SJF":
        return DynamicHybridSimulator(
            jobs, clusters, policy="sjf"
        )

    if policy == "ADAPTIVE":
        return AdaptiveHybridSimulator(
            jobs,
            clusters,
            time_limit_s=0.2,
            cp_threshold=8,
            max_cp_calls=4,
        )

    return DeadlineAwareSimulator(
        jobs,
        clusters,
        priority_policy=policy.lower(),
    )


def evaluate(policy, jobs):
    simulator = build_simulator(policy, jobs)

    started = time.perf_counter()
    results = simulator.run()
    elapsed = time.perf_counter() - started

    assert len(results) == len(jobs)

    total_late = sum(
        r.tardiness_s for r in results
    )
    late = sum(
        r.tardiness_s > 0 for r in results
    )

    return {
        "policy": policy,
        "makespan_s": max(r.finish_s for r in results),
        "mean_wait_s": mean(
            r.waiting_s for r in results
        ),
        "late_jobs": late,
        "total_tardiness_s": total_late,
        "mean_late_s": (
            total_late / late if late else 0
        ),
        "cost_units": sum(
            r.cost_units for r in results
        ),
        "cloud_jobs": sum(
            r.cluster == "gcp-gpu"
            for r in results
        ),
        "solver_calls": simulator.solver_calls,
        "wall_s": elapsed,
    }


def main():
    rows = []

    print("=" * 125)
    print("HybridSched-AI | DEADLINE-AWARE BENCHMARK")
    print("=" * 125)

    print(
        f"{'Jobs':>6}"
        f"{'Seed':>8}"
        f"{'Policy':<13}"
        f"{'Makespan':>11}"
        f"{'Wait':>11}"
        f"{'Late':>8}"
        f"{'Tardiness':>13}"
        f"{'Cost':>13}"
        f"{'CP':>6}"
        f"{'Wall(s)':>11}"
    )
    print("-" * 125)

    for size in SIZES:
        for seed in SEEDS:
            jobs = generate_workload(size, seed)

            for policy in POLICIES:
                metrics = evaluate(policy, jobs)

                record = {
                    "jobs": size,
                    "seed": seed,
                    **metrics,
                }
                rows.append(record)

                print(
                    f"{size:>6}"
                    f"{seed:>8}"
                    f"{policy:<13}"
                    f"{metrics['makespan_s']:>11}"
                    f"{metrics['mean_wait_s']:>11.2f}"
                    f"{metrics['late_jobs']:>8}"
                    f"{metrics['total_tardiness_s']:>13}"
                    f"{metrics['cost_units']:>13}"
                    f"{metrics['solver_calls']:>6}"
                    f"{metrics['wall_s']:>11.3f}",
                    flush=True,
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("MEAN PERFORMANCE ACROSS SEEDS")
    print("-" * 108)
    print(
        f"{'Jobs':>6}"
        f"{'Policy':<13}"
        f"{'Makespan':>12}"
        f"{'Wait':>12}"
        f"{'Late':>9}"
        f"{'Tardiness':>15}"
        f"{'Cost':>14}"
        f"{'Wall(s)':>12}"
    )

    for size in SIZES:
        for policy in POLICIES:
            subset = [
                r for r in rows
                if r["jobs"] == size
                and r["policy"] == policy
            ]

            avg = lambda key: mean(
                r[key] for r in subset
            )

            print(
                f"{size:>6}"
                f"{policy:<13}"
                f"{avg('makespan_s'):>12.1f}"
                f"{avg('mean_wait_s'):>12.1f}"
                f"{avg('late_jobs'):>9.1f}"
                f"{avg('total_tardiness_s'):>15.1f}"
                f"{avg('cost_units'):>14.1f}"
                f"{avg('wall_s'):>12.3f}"
            )

    print("-" * 108)
    print(f"CSV: {OUTPUT}")
    print(f"Completed: {len(rows)} simulations")


if __name__ == "__main__":
    main()

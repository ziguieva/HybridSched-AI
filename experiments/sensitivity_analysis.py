"""Sensitivity analysis for hybrid cloud scheduling."""

import csv
from dataclasses import replace
from pathlib import Path

from benchmark_hybrid import JOBS

from hybridsched.hybrid_optimizer import optimize_hybrid
from hybridsched.infrastructure import default_infrastructure
from hybridsched.resources import Resources


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "sensitivity_analysis.csv"


def evaluate(label, clusters, jobs):
    solution = optimize_hybrid(
        jobs,
        clusters,
        time_limit_s=5,
        wait_weight=2,
        makespan_weight=5,
        cost_weight=1,
        deadline_weight=20,
    )

    if not solution.results:
        return {
            "scenario": label,
            "status": solution.status,
            "makespan": "",
            "cost": "",
            "late_jobs": "",
            "cloud_jobs": "",
        }

    return {
        "scenario": label,
        "status": solution.status,
        "makespan": solution.makespan_s,
        "cost": solution.total_cost_units,
        "late_jobs": solution.deadline_violations,
        "cloud_jobs": sum(
            r.cluster == "gcp-gpu"
            for r in solution.results
        ),
    }


def main():
    base = default_infrastructure()
    rows = []

    # Experiment A: cloud price.
    for multiplier in (1, 2, 4):
        clusters = [
            replace(
                cluster,
                gpu_cost_per_s=(
                    cluster.gpu_cost_per_s * multiplier
                ),
                cpu_cost_per_s=(
                    cluster.cpu_cost_per_s * multiplier
                ),
            )
            if cluster.name == "gcp-gpu"
            else cluster
            for cluster in base
        ]

        rows.append(
            evaluate(
                f"CLOUD_PRICE_X{multiplier}",
                clusters,
                JOBS,
            )
        )

    # Experiment B: available cloud GPU capacity.
    for gpu_capacity in (2, 4, 6):
        clusters = [
            replace(
                cluster,
                capacity=Resources(
                    cluster.capacity.cpu_cores,
                    gpu_capacity,
                    cluster.capacity.memory_gib,
                ),
            )
            if cluster.name == "gcp-gpu"
            else cluster
            for cluster in base
        ]

        rows.append(
            evaluate(
                f"CLOUD_GPU_{gpu_capacity}",
                clusters,
                JOBS,
            )
        )

    # Experiment C: deadline tightness.
    for offset in (500, 800, 1200):
        jobs = [
            replace(
                job,
                deadline_s=job.arrival_s + offset,
            )
            for job in JOBS
        ]

        rows.append(
            evaluate(
                f"DEADLINE_{offset}",
                base,
                jobs,
            )
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

    print("=" * 85)
    print("HybridSched-AI | SENSITIVITY ANALYSIS")
    print("=" * 85)
    print(
        f"{'Scenario':<21}"
        f"{'Status':<13}"
        f"{'Makespan':>12}"
        f"{'Cost':>12}"
        f"{'Late':>10}"
        f"{'Cloud':>10}"
    )
    print("-" * 85)

    for row in rows:
        print(
            f"{row['scenario']:<21}"
            f"{row['status']:<13}"
            f"{str(row['makespan']):>12}"
            f"{str(row['cost']):>12}"
            f"{str(row['late_jobs']):>10}"
            f"{str(row['cloud_jobs']):>10}"
        )

    print("=" * 85)
    print(f"Results: {OUTPUT}")
    print("Sensitivity analysis completed.")


if __name__ == "__main__":
    main()

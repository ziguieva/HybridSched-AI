"""Budget-constrained multiobjective HPC experiments."""

import csv
from dataclasses import replace
from pathlib import Path

from benchmark_hybrid import JOBS

from hybridsched.hybrid_optimizer import optimize_hybrid
from hybridsched.infrastructure import default_infrastructure


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "multiobjective.csv"

# Introduce explicit deadlines for this experiment.
DEADLINE_OFFSETS = [
    1100, 900, 450, 400,
    800, 400, 850, 250,
]

JOBS_WITH_DEADLINES = [
    replace(
        job,
        deadline_s=job.arrival_s + offset,
    )
    for job, offset in zip(JOBS, DEADLINE_OFFSETS)
]


def solve_budget(budget):
    return optimize_hybrid(
        jobs=JOBS_WITH_DEADLINES,
        clusters=default_infrastructure(),
        time_limit_s=5,
        wait_weight=2,
        makespan_weight=5,
        cost_weight=0,
        deadline_weight=20,
        max_budget_units=budget,
    )


def run_experiment():
    clusters = default_infrastructure()

    # Find a low-cost feasible reference.
    minimum_cost_solution = optimize_hybrid(
        JOBS_WITH_DEADLINES,
        clusters,
        time_limit_s=10,
        wait_weight=0,
        makespan_weight=0,
        cost_weight=1,
        deadline_weight=0,
    )

    if not minimum_cost_solution.results:
        raise RuntimeError(
            "No feasible minimum-cost solution"
        )

    minimum_budget = (
        minimum_cost_solution.total_cost_units
    )

    # Upper bound: assign every job to its most
    # expensive compatible cluster.
    maximum_budget = sum(
        max(
            cluster.execution_cost(
                job.resources,
                cluster.execution_time(
                    job.requested_walltime_s
                ),
            )
            for cluster in clusters
            if job.resources.fits_in(cluster.capacity)
        )
        for job in JOBS_WITH_DEADLINES
    )

    budgets = sorted({
        minimum_budget
        + (maximum_budget - minimum_budget) * i // 7
        for i in range(8)
    })

    rows = []

    for budget in budgets:
        solution = solve_budget(budget)

        if not solution.results:
            rows.append({
                "budget": budget,
                "status": solution.status,
                "cost": "",
                "makespan": "",
                "avg_wait": "",
                "late_jobs": "",
                "tardiness": "",
                "cloud_jobs": "",
            })
            continue

        rows.append({
            "budget": budget,
            "status": solution.status,
            "cost": solution.total_cost_units,
            "makespan": solution.makespan_s,
            "avg_wait": round(
                sum(r.waiting_s for r in solution.results)
                / len(solution.results), 2
            ),
            "late_jobs": solution.deadline_violations,
            "tardiness": solution.total_tardiness_s,
            "cloud_jobs": sum(
                r.cluster == "gcp-gpu"
                for r in solution.results
            ),
        })

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
    print("HybridSched-AI | MULTIOBJECTIVE OPTIMIZATION")
    print("=" * 85)

    print(
        f"Minimum-cost reference: {minimum_budget} units "
        f"({minimum_cost_solution.status})"
    )

    print("-" * 85)
    print(
        f"{'Budget':>9}"
        f"{'Status':>13}"
        f"{'Cost':>11}"
        f"{'Makespan':>12}"
        f"{'Tardiness':>13}"
        f"{'Cloud':>9}"
    )
    print("-" * 85)

    for row in rows:
        print(
            f"{row['budget']:>9}"
            f"{row['status']:>13}"
            f"{str(row['cost']):>11}"
            f"{str(row['makespan']):>12}"
            f"{str(row['tardiness']):>13}"
            f"{str(row['cloud_jobs']):>9}"
        )

    print("=" * 85)
    print(f"CSV saved: {OUTPUT}")
    print("Multiobjective experiment completed.")

    return rows


if __name__ == "__main__":
    run_experiment()

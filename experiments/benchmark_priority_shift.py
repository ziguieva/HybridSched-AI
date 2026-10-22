"""Evaluate scheduling robustness under deadline shifts."""

import csv
import time
from pathlib import Path
from statistics import mean

from experiments.workload_shift import (
    MODES,
    generate_shifted_workload,
)

from hybridsched.deadline_feasibility import (
    analyze_deadline_feasibility,
)
from hybridsched.deadline_aware import DeadlineAwareSimulator
from hybridsched.dynamic import DynamicHybridSimulator
from hybridsched.hybrid_priority import HybridPrioritySimulator
from hybridsched.infrastructure import default_infrastructure


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "benchmark_priority_shift.csv"
)

POLICIES = ("SJF", "EDF", "LST", "RESOURCE", "HYBRID")
SIZES = (100, 200)
SEEDS = (42, 123, 2026)


def make_simulator(policy, jobs, clusters):
    if policy == "SJF":
        return DynamicHybridSimulator(
            jobs, clusters, policy="sjf"
        )

    if policy == "HYBRID":
        return HybridPrioritySimulator(
            jobs, clusters
        )

    return DeadlineAwareSimulator(
        jobs,
        clusters,
        priority_policy=policy.lower(),
    )


def run_experiments():
    rows = []
    clusters = default_infrastructure()

    for size in SIZES:
        for seed in SEEDS:
            for mode in MODES:
                jobs = generate_shifted_workload(
                    size, seed, mode
                )

                feasibility = analyze_deadline_feasibility(
                    jobs, clusters
                )

                for policy in POLICIES:
                    sim = make_simulator(
                        policy, jobs, clusters
                    )

                    start = time.perf_counter()
                    results = sim.run()
                    wall = time.perf_counter() - start

                    assert len(results) == size

                    late = sum(
                        r.tardiness_s > 0
                        for r in results
                    )
                    tardiness = sum(
                        r.tardiness_s
                        for r in results
                    )

                    assert late >= feasibility[
                        "intrinsically_late"
                    ]
                    assert tardiness >= feasibility[
                        "minimum_tardiness_s"
                    ]

                    rows.append({
                        "jobs": size,
                        "seed": seed,
                        "mode": mode,
                        "policy": policy,
                        "makespan": max(
                            r.finish_s for r in results
                        ),
                        "mean_wait": mean(
                            r.waiting_s for r in results
                        ),
                        "late": late,
                        "unavoidable_late": feasibility[
                            "intrinsically_late"
                        ],
                        "tardiness": tardiness,
                        "minimum_tardiness": feasibility[
                            "minimum_tardiness_s"
                        ],
                        "excess_tardiness_lower_bound": (
                            tardiness - feasibility[
                                "minimum_tardiness_s"
                            ]
                        ),
                        "cost": sum(
                            r.cost_units for r in results
                        ),
                        "wall": wall,
                    })

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 112)
    print("HybridSched-AI | PRIORITY ROBUSTNESS")
    print("=" * 112)
    print(
        f"{'Jobs':>5} "
        f"{'Mode':<13}"
        f"{'Policy':<11}"
        f"{'Late':>9}"
        f"{'Lower':>9}"
        f"{'Tardiness':>14}"
        f"{'Makespan':>13}"
        f"{'Cost':>14}"
        f"{'Wall(s)':>12}"
    )
    print("-" * 112)

    for size in SIZES:
        for mode in MODES:
            for policy in POLICIES:
                subset = [
                    r for r in rows
                    if r["jobs"] == size
                    and r["mode"] == mode
                    and r["policy"] == policy
                ]

                avg = lambda name: mean(
                    r[name] for r in subset
                )

                print(
                    f"{size:>5} "
                    f"{mode:<13}"
                    f"{policy:<11}"
                    f"{avg('late'):>9.1f}"
                    f"{avg('unavoidable_late'):>9.1f}"
                    f"{avg('tardiness'):>14.0f}"
                    f"{avg('makespan'):>13.0f}"
                    f"{avg('cost'):>14.0f}"
                    f"{avg('wall'):>12.3f}"
                )

    print("=" * 112)
    print(f"Simulations: {len(rows)}")
    print(f"Results: {OUTPUT}")


if __name__ == "__main__":
    run_experiments()

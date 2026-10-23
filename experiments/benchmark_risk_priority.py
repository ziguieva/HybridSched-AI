"""Exploratory risk-priority benchmark on validation seeds."""

import csv
from pathlib import Path
from statistics import mean

from experiments.workload_shift import (
    MODES,
    generate_shifted_workload,
)
from hybridsched.dynamic import DynamicHybridSimulator
from hybridsched.infrastructure import default_infrastructure
from hybridsched.risk_priority import (
    RiskPrioritySimulator,
    RiskWeights,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "risk_priority_validation.csv"

SEEDS = range(201, 209)
WEIGHTS = {
    "RISK_A": RiskWeights(1, 0.25, 0),
    "RISK_B": RiskWeights(1, 0.5, 0),
    "RISK_C": RiskWeights(1, 1, 0),
    "RISK_D": RiskWeights(1, 1, 0.25),
}


def evaluate(jobs, weights):
    if weights is None:
        sim = DynamicHybridSimulator(
            jobs,
            default_infrastructure(),
            policy="sjf",
        )
    else:
        sim = RiskPrioritySimulator(
            jobs,
            default_infrastructure(),
            weights=weights,
        )

    results = sim.run()
    return {
        "late": sum(r.tardiness_s > 0 for r in results),
        "tardiness": sum(r.tardiness_s for r in results),
        "makespan": max(r.finish_s for r in results),
        "cost": sum(r.cost_units for r in results),
    }


def main():
    rows = []

    for seed in SEEDS:
        for mode in MODES:
            jobs = generate_shifted_workload(
                200, seed, mode
            )

            for name, weights in {
                "SJF": None,
                **WEIGHTS,
            }.items():
                metrics = evaluate(jobs, weights)
                rows.append({
                    "seed": seed,
                    "mode": mode,
                    "policy": name,
                    **metrics,
                })

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 88)
    print("HybridSched-AI | RISK PRIORITY VALIDATION")
    print("=" * 88)
    print(
        f"{'Policy':<12}{'Late':>12}"
        f"{'Tardiness':>16}"
        f"{'Makespan':>14}{'Cost':>16}"
    )

    for policy in ("SJF", *WEIGHTS):
        subset = [
            r for r in rows
            if r["policy"] == policy
        ]
        print(
            f"{policy:<12}"
            f"{mean(r['late'] for r in subset):>12.2f}"
            f"{mean(r['tardiness'] for r in subset):>16.0f}"
            f"{mean(r['makespan'] for r in subset):>14.0f}"
            f"{mean(r['cost'] for r in subset):>16.0f}"
        )

    print("=" * 88)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

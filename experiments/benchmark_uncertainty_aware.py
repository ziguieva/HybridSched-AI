"""Compare fixed quantiles with urgency-dependent predictions."""

import csv
from dataclasses import replace
from pathlib import Path
from statistics import mean

from experiments.benchmark_ml_dynamic import load_test_data
from hybridsched.infrastructure import default_infrastructure
from hybridsched.ml_deadline import MLDeadlineSimulator
from hybridsched.ml_predictor import MLRuntimePredictor
from hybridsched.uncertainty_aware import UncertaintyAwareSimulator


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results" / "uncertainty_aware_benchmark.csv"
)


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

    predictor = MLRuntimePredictor()

    policies = (
        "REQUESTED",
        "P50",
        "P90",
        "UNCERTAINTY",
    )

    rows = []

    print("=" * 105)
    print("HybridSched-AI | UNCERTAINTY-AWARE BENCHMARK")
    print("=" * 105)
    print(
        f"{'Policy':<16}{'Makespan':>12}"
        f"{'Wait':>12}{'Late':>9}"
        f"{'Tardiness':>15}{'Cost':>14}"
        f"{'Cloud':>10}"
    )
    print("-" * 105)

    for policy in policies:
        if policy == "UNCERTAINTY":
            sim = UncertaintyAwareSimulator(
                jobs, default_infrastructure(),
                actual, predictor,
            )
        else:
            mode = policy.lower()

            sim = MLDeadlineSimulator(
                jobs, default_infrastructure(),
                actual_runtimes=actual,
                predictor=(
                    None if mode == "requested"
                    else predictor
                ),
                mode=mode,
            )

        results = sim.run()

        row = {
            "policy": policy,
            "makespan_s": max(
                r.finish_s for r in results
            ),
            "mean_wait_s": round(
                mean(r.waiting_s for r in results), 2
            ),
            "late_jobs": sum(
                r.tardiness_s > 0 for r in results
            ),
            "tardiness_s": sum(
                r.tardiness_s for r in results
            ),
            "cost_units": sum(
                r.cost_units for r in results
            ),
            "cloud_jobs": sum(
                r.cluster == "gcp-gpu"
                for r in results
            ),
        }
        rows.append(row)

        print(
            f"{policy:<16}"
            f"{row['makespan_s']:>12}"
            f"{row['mean_wait_s']:>12.2f}"
            f"{row['late_jobs']:>9}"
            f"{row['tardiness_s']:>15}"
            f"{row['cost_units']:>14}"
            f"{row['cloud_jobs']:>10}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 105)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

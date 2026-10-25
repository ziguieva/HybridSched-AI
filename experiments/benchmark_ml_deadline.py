"""Paired held-out ML deadline scheduling experiment."""

import csv
from dataclasses import replace
from pathlib import Path
from statistics import mean

from experiments.benchmark_ml_dynamic import load_test_data
from hybridsched.infrastructure import default_infrastructure
from hybridsched.ml_deadline import MLDeadlineSimulator
from hybridsched.ml_predictor import MLRuntimePredictor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "benchmark_ml_deadline.csv"

MODES = ("requested", "point", "p50", "p90")


def main():
    jobs, actual = load_test_data()

    jobs = [
        replace(
            job,
            deadline_s=(
                job.arrival_s
                + job.requested_walltime_s
                + 300
            ),
        )
        for job in jobs
    ]

    predictor = MLRuntimePredictor()
    rows = []

    print("=" * 100)
    print("HybridSched-AI | ML DEADLINE-AWARE BENCHMARK")
    print("=" * 100)

    print(
        f"{'Mode':<13}"
        f"{'Makespan':>12}"
        f"{'Mean Wait':>13}"
        f"{'Late':>9}"
        f"{'Tardiness':>14}"
        f"{'Cost':>14}"
        f"{'Cloud':>10}"
    )
    print("-" * 100)

    for mode in MODES:
        simulation = MLDeadlineSimulator(
            jobs,
            default_infrastructure(),
            actual_runtimes=actual,
            predictor=(
                predictor if mode != "requested"
                else None
            ),
            mode=mode,
        )

        results = simulation.run()

        if len(results) != len(jobs):
            raise RuntimeError("Incomplete simulation")

        row = {
            "mode": mode,
            "makespan_s": max(
                r.finish_s for r in results
            ),
            "mean_wait_s": mean(
                r.waiting_s for r in results
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
            f"{mode:<13}"
            f"{row['makespan_s']:>12}"
            f"{row['mean_wait_s']:>13.2f}"
            f"{row['late_jobs']:>9}"
            f"{row['tardiness_s']:>14}"
            f"{row['cost_units']:>14}"
            f"{row['cloud_jobs']:>10}"
        )

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print(f"Saved: {OUTPUT}")
    print("ML deadline experiment completed.")


if __name__ == "__main__":
    main()

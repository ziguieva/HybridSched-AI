"""Nine ranking/placement combinations on identical jobs."""

import csv
from pathlib import Path
from statistics import mean

from experiments.benchmark_ml_dynamic import load_test_data
from hybridsched.infrastructure import default_infrastructure
from hybridsched.ml_ablation import MLAblationSimulator
from hybridsched.ml_predictor import MLRuntimePredictor


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results" / "ml_ablation.csv"
)

MODES = ("requested", "p50", "p90")


def main():
    jobs, actual = load_test_data()
    predictor = MLRuntimePredictor()

    rows = []

    print("=" * 100)
    print("HybridSched-AI | ML SCHEDULING ABLATION")
    print("=" * 100)
    print(
        f"{'Ranking':<14}{'Placement':<14}"
        f"{'Makespan':>12}{'Mean Wait':>14}"
        f"{'Cost':>14}{'Cloud':>10}"
    )
    print("-" * 100)

    for ranking in MODES:
        for placement in MODES:
            sim = MLAblationSimulator(
                jobs,
                default_infrastructure(),
                actual,
                predictor,
                ranking_mode=ranking,
                placement_mode=placement,
            )

            results = sim.run()

            row = {
                "ranking": ranking,
                "placement": placement,
                "makespan_s": max(
                    r.finish_s for r in results
                ),
                "mean_wait_s": round(
                    mean(r.waiting_s for r in results), 2
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
                f"{ranking:<14}{placement:<14}"
                f"{row['makespan_s']:>12}"
                f"{row['mean_wait_s']:>14.2f}"
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

    print("=" * 100)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

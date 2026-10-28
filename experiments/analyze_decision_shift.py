"""Paired bootstrap analysis of decision-focused scheduling."""

import csv
import random
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "decision_shift_benchmark.csv"
)

METRICS = (
    "late_jobs",
    "tardiness_s",
    "mean_wait_s",
    "cost_units",
    "makespan_s",
)


def bootstrap_ci(values, seed=202620):
    if not values:
        raise ValueError("No paired observations")

    rng = random.Random(seed)
    n = len(values)
    estimates = []

    for _ in range(3000):
        sample = [
            values[rng.randrange(n)]
            for _ in range(n)
        ]
        estimates.append(mean(sample))

    estimates.sort()

    return (
        mean(values),
        estimates[int(0.025 * len(estimates))],
        estimates[int(0.975 * len(estimates))],
    )


def main():
    with INPUT.open(newline="") as file:
        rows = list(csv.DictReader(file))

    indexed = {
        (
            row["scenario"],
            int(row["seed"]),
            row["policy"],
        ): row
        for row in rows
    }

    scenarios = sorted({
        r["scenario"] for r in rows
    })

    print("=" * 90)
    print("STEP 20.3 | PAIRED DECISION-FOCUSED COMPARISON")
    print("Differences = DECISION - P90")
    print("=" * 90)

    for scenario in scenarios:
        print(f"\n{scenario}")

        seeds = sorted({
            int(r["seed"])
            for r in rows
            if r["scenario"] == scenario
        })

        for metric in METRICS:
            differences = []

            for seed in seeds:
                decision = indexed[
                    (scenario, seed, "DECISION")
                ]
                baseline = indexed[
                    (scenario, seed, "P90")
                ]

                differences.append(
                    float(decision[metric])
                    - float(baseline[metric])
                )

            avg, low, high = bootstrap_ci(
                differences
            )

            direction = (
                "BETTER" if high < 0
                else "WORSE" if low > 0
                else "INCONCLUSIVE"
            )

            print(
                f"{metric:<15}"
                f"delta={avg:>11.2f} "
                f"CI95=[{low:>10.2f}, "
                f"{high:>10.2f}] "
                f"{direction}"
            )

    print("\nInterpretation limited to these synthetic shifts.")


if __name__ == "__main__":
    main()

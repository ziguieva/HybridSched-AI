"""Paired statistical evaluation of MULTI versus baselines."""

import csv
import random
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "multiobjective_benchmark.csv"
OUTPUT = ROOT / "results" / "step23_statistical.csv"

METRICS = (
    "late_jobs",
    "tardiness_s",
    "cost_units",
    "mean_wait_s",
    "makespan_s",
)

BASELINES = ("REQUESTED", "P90", "SLACK")


def paired_bootstrap(values, seed=23001, repeats=5000):
    if not values:
        raise ValueError("Empty paired sample")

    rng = random.Random(seed)
    n = len(values)
    estimates = []

    for _ in range(repeats):
        estimates.append(mean(
            values[rng.randrange(n)]
            for _ in range(n)
        ))

    estimates.sort()

    return (
        mean(values),
        estimates[int(0.025 * (repeats - 1))],
        estimates[int(0.975 * (repeats - 1))],
    )


def analyze(rows):
    indexed = {
        (r["scenario"], int(r["seed"]), r["policy"]): r
        for r in rows
    }

    if len(indexed) != len(rows):
        raise ValueError("Duplicate scenario/seed/policy rows")

    scenarios = sorted(set(r["scenario"] for r in rows))
    output = []

    for scenario in scenarios:
        seeds = sorted(set(
            int(r["seed"]) for r in rows
            if r["scenario"] == scenario
        ))

        for baseline in BASELINES:
            for metric in METRICS:
                differences = []

                for seed in seeds:
                    multi = indexed[(scenario, seed, "MULTI")]
                    reference = indexed[(scenario, seed, baseline)]

                    differences.append(
                        float(multi[metric])
                        - float(reference[metric])
                    )

                avg, low, high = paired_bootstrap(
                    differences, seed=23001
                )

                verdict = (
                    "BETTER" if high < 0
                    else "WORSE" if low > 0
                    else "INCONCLUSIVE"
                )

                output.append({
                    "scenario": scenario,
                    "baseline": baseline,
                    "metric": metric,
                    "n_seeds": len(seeds),
                    "mean_difference": round(avg, 3),
                    "ci95_low": round(low, 3),
                    "ci95_high": round(high, 3),
                    "verdict": verdict,
                })

    return output


def main():
    with INPUT.open(newline="") as f:
        rows = list(csv.DictReader(f))

    results = analyze(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    print("=== STEP 23.1 | PAIRED STATISTICAL EVALUATION ===")
    print("Negative differences favor MULTI.")

    for row in results:
        if row["metric"] not in ("late_jobs", "tardiness_s", "cost_units"):
            continue

        print(
            f"{row['scenario']:<11}"
            f"{row['baseline']:<11}"
            f"{row['metric']:<15}"
            f"delta={row['mean_difference']:>11.1f} "
            f"CI=[{row['ci95_low']:>10.1f}, "
            f"{row['ci95_high']:>10.1f}] "
            f"{row['verdict']}"
        )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()

"""Per-seed Pareto stability analysis."""

import csv
from collections import defaultdict
from pathlib import Path

from experiments.benchmark_multiobjective import dominates


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "multiobjective_benchmark.csv"
OUTPUT = ROOT / "results" / "step23_pareto_stability.csv"

OBJECTIVES = ("late_jobs", "tardiness_s", "cost_units")


def analyze(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (row["scenario"], int(row["seed"]))
        groups[key].append(row)

    counts = defaultdict(lambda: [0, 0])

    for (scenario, seed), group in groups.items():
        policies = [r["policy"] for r in group]
        if len(set(policies)) != len(policies):
            raise ValueError(f"Duplicate policy: {scenario}/{seed}")

        for row in group:
            point = {k: float(row[k]) for k in OBJECTIVES}

            nondominated = not any(
                dominates(
                    {k: float(other[k]) for k in OBJECTIVES},
                    point,
                )
                for other in group
                if other is not row
            )

            key = (scenario, row["policy"])
            counts[key][0] += int(nondominated)
            counts[key][1] += 1

    return [
        {
            "scenario": scenario,
            "policy": policy,
            "pareto_count": count,
            "total_seeds": total,
            "pareto_frequency_pct": round(
                100 * count / total, 2
            ),
        }
        for (scenario, policy), (count, total)
        in sorted(counts.items())
    ]


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

    print("=== STEP 23.2 | PARETO STABILITY ===")
    print(
        f"{'Scenario':<13}{'Policy':<13}"
        f"{'Non-dominated':>17}{'Frequency':>14}"
    )

    for row in results:
        print(
            f"{row['scenario']:<13}"
            f"{row['policy']:<13}"
            f"{row['pareto_count']:>11}/"
            f"{row['total_seeds']:<5}"
            f"{row['pareto_frequency_pct']:>12.2f}%"
        )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()

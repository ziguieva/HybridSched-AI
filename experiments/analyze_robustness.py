"""Distribution-specific held-out robustness analysis."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "calibrated_test.csv"
OUTPUT = ROOT / "results" / "robustness_by_distribution.csv"

METRICS = ("late", "tardiness", "makespan", "cost", "mean_wait")
MODES = ("COUPLED", "INDEPENDENT", "TIGHT")


def analyze(records):
    indexed = {
        (int(r["seed"]), r["mode"], r["policy"]): r
        for r in records
    }
    seeds = sorted({int(r["seed"]) for r in records})
    rows = []

    for mode in MODES:
        for metric in METRICS:
            pairs = []

            for seed in seeds:
                baseline = indexed[(seed, mode, "SJF")]
                candidate = indexed[(seed, mode, "SELECTED")]

                pairs.append((
                    float(baseline[metric]),
                    float(candidate[metric]),
                ))

            baseline_mean = mean(x for x, _ in pairs)
            selected_mean = mean(y for _, y in pairs)

            gain = (
                100 * (baseline_mean - selected_mean)
                / baseline_mean
                if baseline_mean else 0
            )

            improvements = sum(
                candidate < baseline
                for baseline, candidate in pairs
            )

            rows.append({
                "mode": mode,
                "metric": metric,
                "seeds": len(pairs),
                "sjf_mean": round(baseline_mean, 3),
                "selected_mean": round(selected_mean, 3),
                "gain_pct": round(gain, 3),
                "improved_seeds": improvements,
            })

    return rows


def main():
    with SOURCE.open(newline="") as f:
        records = list(csv.DictReader(f))

    expected = 30 * 3 * 4
    if len(records) != expected:
        raise ValueError(
            f"Expected {expected} records, got {len(records)}"
        )

    rows = analyze(records)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 94)
    print("HybridSched-AI | DISTRIBUTION ROBUSTNESS")
    print("=" * 94)
    print(
        f"{'Mode':<16}{'Metric':<14}"
        f"{'SJF':>13}{'Selected':>13}"
        f"{'Gain %':>12}{'Better seeds':>15}"
    )

    for row in rows:
        print(
            f"{row['mode']:<16}"
            f"{row['metric']:<14}"
            f"{row['sjf_mean']:>13.1f}"
            f"{row['selected_mean']:>13.1f}"
            f"{row['gain_pct']:>12.2f}"
            f"{row['improved_seeds']:>12}/30"
        )

    print("=" * 94)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

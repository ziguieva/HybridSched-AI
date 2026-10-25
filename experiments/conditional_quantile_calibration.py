"""Conditional quantile calibration on existing test data."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from experiments.benchmark_ml_dynamic import load_test_data
from hybridsched.ml_predictor import MLRuntimePredictor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "conditional_quantile_calibration.csv"


def duration_group(seconds):
    if seconds <= 300:
        return "SHORT"
    if seconds <= 900:
        return "MEDIUM"
    return "LONG"


def gpu_group(gpus):
    if gpus == 1:
        return "GPU_1"
    if gpus == 2:
        return "GPU_2"
    return "GPU_3_PLUS"


def summarize(jobs, actual, predictions):
    groups = defaultdict(list)

    for job in jobs:
        prediction = predictions[job.job_id]
        runtime = actual[job.job_id]

        values = {
            "actual": runtime,
            "p10": prediction.p10_s,
            "p50": prediction.p50_s,
            "p90": prediction.p90_s,
        }

        for dimension, group in (
            (
                "requested_duration",
                duration_group(job.requested_walltime_s),
            ),
            ("gpu_demand", gpu_group(job.resources.gpus)),
        ):
            groups[(dimension, group)].append(values)

    rows = []

    for (dimension, group), items in sorted(groups.items()):
        count = len(items)

        rows.append({
            "dimension": dimension,
            "group": group,
            "count": count,
            "p10_coverage_pct": round(
                100 * mean(
                    x["actual"] <= x["p10"]
                    for x in items
                ), 2
            ),
            "p50_coverage_pct": round(
                100 * mean(
                    x["actual"] <= x["p50"]
                    for x in items
                ), 2
            ),
            "p90_coverage_pct": round(
                100 * mean(
                    x["actual"] <= x["p90"]
                    for x in items
                ), 2
            ),
            "interval_coverage_pct": round(
                100 * mean(
                    x["p10"] <= x["actual"] <= x["p90"]
                    for x in items
                ), 2
            ),
            "mean_width_s": round(
                mean(x["p90"] - x["p10"] for x in items),
                2,
            ),
        })

    return rows


def main():
    jobs, actual = load_test_data()
    predictor = MLRuntimePredictor()
    predictions = predictor.predict_many(jobs)

    rows = summarize(jobs, actual, predictions)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print("HybridSched-AI | CONDITIONAL QUANTILE CALIBRATION")
    print("=" * 100)
    print(
        f"{'Dimension':<23}{'Group':<14}"
        f"{'N':>7}{'P90 coverage':>17}"
        f"{'P10-P90':>15}{'Width(s)':>13}"
    )
    print("-" * 100)

    for row in rows:
        print(
            f"{row['dimension']:<23}"
            f"{row['group']:<14}"
            f"{row['count']:>7}"
            f"{row['p90_coverage_pct']:>16.2f}%"
            f"{row['interval_coverage_pct']:>14.2f}%"
            f"{row['mean_width_s']:>13.2f}"
        )

    print("=" * 100)
    print(f"Saved: {OUTPUT}")
    print("Small groups require cautious interpretation.")


if __name__ == "__main__":
    main()

"""Fit on validation only, inspect the previously used test set."""

import csv
import json
from pathlib import Path

from experiments.benchmark_ml_dynamic import load_test_data
from hybridsched.conditional_calibration import (
    CalibratedRuntimePredictor,
    duration_group,
    fit_corrections,
)
from hybridsched.job import Job
from hybridsched.ml_predictor import MLRuntimePredictor
from hybridsched.resources import Resources


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "conditional_p90_corrections.json"


def load_validation():
    with (ROOT / "data/synthetic/jobs.csv").open(
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    with (ROOT / "data/synthetic/actual_runtime.csv").open(
        newline=""
    ) as f:
        actual = {
            row["job_id"]: float(row["actual_runtime_s"])
            for row in csv.DictReader(f)
        }

    if len(rows) != 1000:
        raise ValueError("Expected chronological 1000-job dataset")

    jobs = []
    for row in rows[700:850]:
        jobs.append(Job(
            job_id=row["job_id"],
            arrival_s=int(float(row["arrival_s"])),
            requested_walltime_s=int(
                float(row["requested_walltime_s"])
            ),
            resources=Resources(
                cpu_cores=int(float(row["cpu_cores"])),
                gpus=int(float(row["gpus"])),
                memory_gib=int(float(row["memory_gib"])),
            ),
            priority=int(float(row["priority"])),
        ))

    return jobs, actual


def coverage(jobs, actual, predicted):
    groups = {}
    for job in jobs:
        group = duration_group(job.requested_walltime_s)
        groups.setdefault(group, []).append(
            actual[job.job_id] <= predicted[job.job_id].p90_s
        )
    return {
        group: (len(values), 100 * sum(values) / len(values))
        for group, values in groups.items()
    }


def main():
    validation, validation_actual = load_validation()
    test, test_actual = load_test_data()

    predictor = MLRuntimePredictor()
    corrections = fit_corrections(
        validation, validation_actual, predictor
    )
    calibrated = CalibratedRuntimePredictor(
        predictor, corrections
    )

    baseline = predictor.predict_many(test)
    adjusted = calibrated.predict_many(test)

    report = {
        "fitted_on": "validation_rows_700_849",
        "corrections_s": corrections,
        "test_coverage_original": coverage(
            test, test_actual, baseline
        ),
        "test_coverage_calibrated": coverage(
            test, test_actual, adjusted
        ),
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print("=== STEP 19.1 | P90 CONDITIONAL CALIBRATION ===")
    for group in sorted(corrections):
        n, before = report["test_coverage_original"][group]
        _, after = report["test_coverage_calibrated"][group]
        print(
            f"{group:<8} N={n:<3} "
            f"Correction={corrections[group]:>7.2f}s "
            f"Coverage={before:>6.2f}% -> {after:>6.2f}%"
        )

    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

"""Quantile calibration on the chronological held-out test split."""

import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from experiments.benchmark_ml_dynamic import load_test_data
from hybridsched.ml_predictor import MLRuntimePredictor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "ml_quantile_validation.json"


def main():
    jobs, actual = load_test_data()
    predictor = MLRuntimePredictor()

    config = json.loads(
        predictor.quantile_model.save_config()
    )

    objective = config["learner"]["objective"]
    print("Model objective:", objective["name"])

    params = objective.get("quantile_loss_param", {})
    print("Quantile parameters:", params)

    if objective["name"] != "reg:quantileerror":
        raise ValueError("Unexpected quantile objective")

    alpha = json.loads(params["quantile_alpha"])
    expected_alpha = [0.1, 0.5, 0.9]

    if not np.allclose(alpha, expected_alpha, atol=1e-6):
        raise ValueError(
            f"Unexpected quantile order: {alpha}"
        )

    estimates = predictor.predict_many(jobs)

    y = np.asarray(
        [actual[j.job_id] for j in jobs],
        dtype=float,
    )

    q10 = np.asarray(
        [estimates[j.job_id].p10_s for j in jobs]
    )
    q50 = np.asarray(
        [estimates[j.job_id].p50_s for j in jobs]
    )
    q90 = np.asarray(
        [estimates[j.job_id].p90_s for j in jobs]
    )

    coverage = {
        "p10": float(np.mean(y <= q10)),
        "p50": float(np.mean(y <= q50)),
        "p90": float(np.mean(y <= q90)),
    }

    interval_coverage = float(
        np.mean((y >= q10) & (y <= q90))
    )

    interval_width = float(
        np.mean(q90 - q10)
    )

    pinball = {}

    for name, tau, predicted in (
        ("p10", 0.1, q10),
        ("p50", 0.5, q50),
        ("p90", 0.9, q90),
    ):
        error = y - predicted
        pinball[name] = float(
            np.mean(
                np.maximum(
                    tau * error,
                    (tau - 1) * error,
                )
            )
        )

    report = {
        "test_jobs": len(jobs),
        "expected_quantiles": expected_alpha,
        "coverage": coverage,
        "interval_coverage_p10_p90": interval_coverage,
        "mean_interval_width_s": interval_width,
        "pinball_loss": pinball,
        "p50_mae_s": float(
            mean(abs(a - b) for a, b in zip(y, q50))
        ),
        "quantile_crossing_after_adapter": int(
            np.sum((q10 > q50) | (q50 > q90))
        ),
    }

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("=" * 75)
    print("HybridSched-AI | QUANTILE CALIBRATION")
    print("=" * 75)

    for name, value in coverage.items():
        print(f"{name.upper():<12} coverage={value * 100:.2f}%")

    print(
        f"P10-P90 interval coverage: "
        f"{interval_coverage * 100:.2f}%"
    )
    print(f"Mean interval width: {interval_width:.2f}s")

    for name, loss in pinball.items():
        print(f"Pinball {name}: {loss:.2f}s")

    print(f"P50 MAE: {report['p50_mae_s']:.2f}s")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

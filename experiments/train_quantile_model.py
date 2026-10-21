"""Quantile runtime prediction for HybridSched-AI."""

import csv
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "requested_walltime_s",
    "cpu_cores",
    "gpus",
    "memory_gib",
    "priority",
    "arrival_s",
]

QUANTILES = (0.10, 0.50, 0.90)


def load_split(name):
    with (DATA / f"{name}.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    X = np.array(
        [
            [float(row[col]) for col in FEATURES]
            for row in rows
        ],
        dtype=np.float32,
    )

    y = np.array(
        [float(row["actual_runtime_s"]) for row in rows],
        dtype=np.float32,
    )

    return rows, X, y


def ordered_predictions(model, X):
    predictions = np.asarray(
        model.predict(X),
        dtype=float,
    ).reshape(-1, 3)

    predictions = np.maximum(predictions, 1.0)

    # Prevent crossing of predicted quantiles.
    return np.sort(predictions, axis=1)


def evaluate(y, predictions):
    p10 = predictions[:, 0]
    p50 = predictions[:, 1]
    p90 = predictions[:, 2]

    return {
        "mae_p50_s": float(np.mean(np.abs(y - p50))),
        "p10_coverage": float(np.mean(y <= p10)),
        "p90_coverage": float(np.mean(y <= p90)),
        "interval_coverage": float(
            np.mean((y >= p10) & (y <= p90))
        ),
        "mean_interval_width_s": float(
            np.mean(p90 - p10)
        ),
    }


def main():
    train_rows, X_train, y_train = load_split("train")
    val_rows, X_val, y_val = load_split("validation")
    test_rows, X_test, y_test = load_split("test")

    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=list(QUANTILES),
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        reg_lambda=5.0,
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )

    model.fit(X_train, y_train)

    print("=" * 70)
    print("HybridSched-AI | QUANTILE RUNTIME PREDICTION")
    print("=" * 70)

    print(f"Training samples:   {len(train_rows)}")
    print(f"Validation samples: {len(val_rows)}")
    print(f"Test samples:       {len(test_rows)}")

    reports = {}

    for name, X, y in (
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ):
        predictions = ordered_predictions(model, X)
        metrics = evaluate(y, predictions)
        reports[name] = metrics

        print(f"\n{name.upper()}")
        print("-" * 70)
        print(f"P50 MAE:           {metrics['mae_p50_s']:.2f} s")
        print(f"P10 coverage:      {metrics['p10_coverage']:.2%}")
        print(f"P90 coverage:      {metrics['p90_coverage']:.2%}")
        print(f"P10-P90 coverage:  {metrics['interval_coverage']:.2%}")
        print(
            "Mean interval width: "
            f"{metrics['mean_interval_width_s']:.2f} s"
        )

    model_path = OUTPUT / "runtime_quantiles.json"
    model.save_model(str(model_path))

    predictions = ordered_predictions(model, X_test)

    csv_path = OUTPUT / "runtime_quantiles_test.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "job_id", "actual_runtime_s",
            "p10_s", "p50_s", "p90_s",
        ])

        for row, y, pred in zip(
            test_rows, y_test, predictions
        ):
            writer.writerow([
                row["job_id"],
                float(y),
                *[float(value) for value in pred],
            ])

    with (OUTPUT / "quantile_report.json").open("w") as f:
        json.dump(reports, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Model: {model_path}")
    print(f"Predictions: {csv_path}")
    print("Quantile training completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()

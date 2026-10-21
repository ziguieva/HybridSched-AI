"""Train and evaluate HPC runtime prediction models."""

import csv
from pathlib import Path
from statistics import median

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "results"

FEATURES = [
    "requested_walltime_s",
    "cpu_cores",
    "gpus",
    "memory_gib",
    "priority",
    "arrival_s",
]


def load_split(name):
    path = DATA / f"{name}.csv"

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    X = [
        [float(row[column]) for column in FEATURES]
        for row in rows
    ]

    y = [
        float(row["actual_runtime_s"])
        for row in rows
    ]

    return rows, X, y


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return mae, rmse


def print_metrics(name, y_true, predictions):
    mae, rmse = evaluate(y_true, predictions)

    print(
        f"{name:<22}"
        f"{mae:>14.2f}"
        f"{rmse:>14.2f}"
    )

    return mae


def predict_baselines(rows, ratio):
    requested = [
        float(row["requested_walltime_s"])
        for row in rows
    ]

    return {
        "Requested walltime": requested,
        "Historical ratio": [
            max(1.0, value * ratio)
            for value in requested
        ],
    }


def main():
    train_rows, X_train, y_train = load_split("train")
    val_rows, X_val, y_val = load_split("validation")
    test_rows, X_test, y_test = load_split("test")

    print("=" * 72)
    print("HybridSched-AI | MACHINE LEARNING RUNTIME PREDICTION")
    print("=" * 72)

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"Features:           {len(FEATURES)}")

    # Baseline fitted only on training observations.
    historical_ratio = median(
        actual / float(row["requested_walltime_s"])
        for row, actual in zip(train_rows, y_train)
    )

    print(f"Historical runtime ratio: {historical_ratio:.4f}")

    # Gradient-boosted decision tree regressor.
    model = XGBRegressor(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=1.0,
        reg_lambda=5.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )

    model.fit(X_train, y_train)

    val_predictions = {
        **predict_baselines(val_rows, historical_ratio),
        "XGBoost": [
            max(1.0, float(value))
            for value in model.predict(X_val)
        ],
    }

    test_predictions = {
        **predict_baselines(test_rows, historical_ratio),
        "XGBoost": [
            max(1.0, float(value))
            for value in model.predict(X_test)
        ],
    }

    print("-" * 72)
    print("VALIDATION RESULTS")
    print("-" * 72)

    print(
        f"{'Model':<22}"
        f"{'MAE (s)':>14}"
        f"{'RMSE (s)':>14}"
    )

    validation_maes = {}

    for name, predictions in val_predictions.items():
        validation_maes[name] = print_metrics(
            name, y_val, predictions
        )

    selected = min(
        validation_maes,
        key=validation_maes.get,
    )

    print("-" * 72)
    print(f"Best validation MAE: {selected}")

    print("-" * 72)
    print("TEST RESULTS (held-out chronological data)")
    print("-" * 72)

    print(
        f"{'Model':<22}"
        f"{'MAE (s)':>14}"
        f"{'RMSE (s)':>14}"
    )

    test_maes = {}

    for name, predictions in test_predictions.items():
        test_maes[name] = print_metrics(
            name, y_test, predictions
        )

    baseline_mae = test_maes["Requested walltime"]
    xgb_mae = test_maes["XGBoost"]

    improvement = (
        100 * (baseline_mae - xgb_mae) / baseline_mae
    )

    print("-" * 72)
    print(
        f"XGBoost MAE improvement vs requested: "
        f"{improvement:.2f}%"
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT / "runtime_xgboost.json"
    model.save_model(str(model_path))

    predictions_path = OUTPUT / "runtime_predictions.csv"

    with predictions_path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "job_id",
            "actual_runtime_s",
            "requested_walltime_s",
            "historical_ratio_prediction_s",
            "xgboost_prediction_s",
        ])

        for i, row in enumerate(test_rows):
            writer.writerow([
                row["job_id"],
                y_test[i],
                test_predictions["Requested walltime"][i],
                test_predictions["Historical ratio"][i],
                test_predictions["XGBoost"][i],
            ])

    print(f"Model saved: {model_path}")
    print(f"Predictions saved: {predictions_path}")
    print("=" * 72)
    print("Model training completed successfully.")


if __name__ == "__main__":
    main()

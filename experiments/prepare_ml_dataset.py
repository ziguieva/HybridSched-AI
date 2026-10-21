"""Prepare chronological ML datasets without data leakage."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "synthetic"
OUTPUT = ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "job_id",
    "arrival_s",
    "requested_walltime_s",
    "cpu_cores",
    "gpus",
    "memory_gib",
    "priority",
]


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def save_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    features = read_csv(SOURCE / "jobs.csv")
    actuals = read_csv(SOURCE / "actual_runtime.csv")

    if not features or not actuals:
        raise ValueError("Input datasets are empty")

    feature_ids = [r["job_id"] for r in features]
    actual_ids = [r["job_id"] for r in actuals]

    if (
        len(feature_ids) != len(set(feature_ids))
        or len(actual_ids) != len(set(actual_ids))
        or set(feature_ids) != set(actual_ids)
    ):
        raise ValueError("Invalid or duplicate job IDs")

    actual_map = {
        row["job_id"]: int(row["actual_runtime_s"])
        for row in actuals
    }

    dataset = []

    for row in features:
        record = {
            column: row[column]
            for column in FEATURE_COLUMNS
        }

        record["actual_runtime_s"] = actual_map[
            row["job_id"]
        ]

        if record["actual_runtime_s"] <= 0:
            raise ValueError("Invalid actual runtime")

        dataset.append(record)

    dataset.sort(
        key=lambda row: (
            int(row["arrival_s"]),
            row["job_id"],
        )
    )

    n = len(dataset)

    if n < 10:
        raise ValueError("Not enough observations")

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    splits = {
        "train": dataset[:train_end],
        "validation": dataset[
            train_end:validation_end
        ],
        "test": dataset[validation_end:],
    }

    all_ids = []

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HybridSched-AI | ML DATA PREPARATION")
    print("=" * 60)

    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty split: {name}")

        path = OUTPUT / f"{name}.csv"
        save_csv(path, rows)

        all_ids.extend(row["job_id"] for row in rows)

        print(f"{name:<12}: {len(rows)} jobs")
        print(f"  First arrival: {rows[0]['arrival_s']} s")
        print(f"  Last arrival:  {rows[-1]['arrival_s']} s")

    assert len(all_ids) == len(set(all_ids))
    assert len(all_ids) == n

    print("-" * 60)
    print(f"Total observations: {n}")
    print("Split strategy: chronological 70/15/15")
    print("Duplicate job IDs: 0")
    print("Target excluded from model features: YES")
    print("Dataset preparation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()

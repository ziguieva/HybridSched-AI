"""Train/validation calibration without test-set leakage."""

import csv
import json
from pathlib import Path
from statistics import mean

from experiments.workload_shift import (
    MODES,
    generate_shifted_workload,
)
from hybridsched.calibrated_priority import (
    CalibratedPrioritySimulator,
    PriorityWeights,
)
from hybridsched.dynamic import DynamicHybridSimulator
from hybridsched.infrastructure import default_infrastructure


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
TRAIN_SEEDS = tuple(range(101, 109))
VAL_SEEDS = tuple(range(201, 209))
TEST_SEEDS = tuple(range(1001, 1031))
N_JOBS = 200

CANDIDATES = [
    PriorityWeights(d, s, g)
    for d, s, g in [
        (1, 0, 0),
        (1, 0, 15),
        (1, 0, 30),
        (1, 0.25, 0),
        (1, 0.25, 15),
        (1, 0.5, 0),
        (1, 0.5, 15),
        (1, 1, 0),
        (1, 1, 15),
        (1, 2, 0),
        (1, 2, 30),
        (1, 4, 30),
    ]
]


def evaluate(jobs, weights=None):
    clusters = default_infrastructure()

    if weights is None:
        sim = DynamicHybridSimulator(
            jobs, clusters, policy="sjf"
        )
    else:
        sim = CalibratedPrioritySimulator(
            jobs, clusters, weights=weights
        )

    results = sim.run()

    return {
        "late": sum(r.tardiness_s > 0 for r in results),
        "tardiness": sum(r.tardiness_s for r in results),
        "makespan": max(r.finish_s for r in results),
        "cost": sum(r.cost_units for r in results),
    }


def workload_set(seeds):
    return [
        generate_shifted_workload(
            N_JOBS, seed, mode
        )
        for seed in seeds
        for mode in MODES
    ]


def aggregate(workloads, weights):
    measurements = [
        evaluate(jobs, weights)
        for jobs in workloads
    ]

    return {
        key: mean(r[key] for r in measurements)
        for key in (
            "late", "tardiness", "makespan", "cost"
        )
    }


def dominates(a, b):
    keys = ("late", "tardiness", "makespan", "cost")
    return (
        all(a[k] <= b[k] for k in keys)
        and any(a[k] < b[k] for k in keys)
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train = workload_set(TRAIN_SEEDS)
    validation = workload_set(VAL_SEEDS)

    sjf_train = aggregate(train, None)
    sjf_val = aggregate(validation, None)

    records = []

    print("=" * 92)
    print("HybridSched-AI | PRIORITY CALIBRATION")
    print("=" * 92)
    print(f"Train workloads: {len(train)}")
    print(f"Validation workloads: {len(validation)}")
    print(f"Candidates: {len(CANDIDATES)}")

    for index, weights in enumerate(CANDIDATES):
        train_metrics = aggregate(train, weights)

        record = {
            "id": index,
            "duration": weights.duration,
            "negative_slack": weights.negative_slack,
            "gpu": weights.gpu,
            **{
                f"train_{k}": value
                for k, value in train_metrics.items()
            },
        }
        records.append(record)

        print(
            f"Candidate {index:02d} "
            f"weights=({weights.duration},"
            f"{weights.negative_slack},{weights.gpu}) "
            f"late={train_metrics['late']:.2f} "
            f"tardiness={train_metrics['tardiness']:.0f}",
            flush=True,
        )

    objective_keys = (
        "late", "tardiness", "makespan", "cost"
    )

    def train_objectives(row):
        return {
            k: row[f"train_{k}"]
            for k in objective_keys
        }

    frontier = [
        row for row in records
        if not any(
            dominates(
                train_objectives(other),
                train_objectives(row),
            )
            for other in records
            if other["id"] != row["id"]
        )
    ]

    for row in frontier:
        weights = CANDIDATES[row["id"]]
        val_metrics = aggregate(validation, weights)

        row.update({
            f"val_{k}": value
            for k, value in val_metrics.items()
        })

    eligible = [
        row for row in frontier
        if row["val_tardiness"]
        <= 1.05 * sjf_val["tardiness"]
    ]

    # If the predeclared constraint is not met by any
    # candidate, retain SJF rather than forcing a claim.
    selected = None

    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                row["val_late"],
                row["val_tardiness"],
                row["val_makespan"],
                row["val_cost"],
                row["id"],
            ),
        )

    with (OUT / "calibration_candidates.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)

    config = {
        "status": (
            "CALIBRATED" if selected else "SJF_FALLBACK"
        ),
        "weights": (
            {
                "duration": selected["duration"],
                "negative_slack": selected["negative_slack"],
                "gpu": selected["gpu"],
            }
            if selected else None
        ),
        "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VAL_SEEDS),
        "test_seeds": list(TEST_SEEDS),
        "n_jobs": N_JOBS,
        "train_pareto_candidates": [
            r["id"] for r in frontier
        ],
        "sjf_validation": sjf_val,
        "selected_validation": (
            {
                k: selected[f"val_{k}"]
                for k in objective_keys
            }
            if selected else sjf_val
        ),
    }

    path = OUT / "selected_priority.json"
    path.write_text(
        json.dumps(config, indent=2) + "\n"
    )

    print("-" * 92)
    print(f"Train Pareto candidates: {len(frontier)}")
    print(
        f"SJF validation: late={sjf_val['late']:.2f}, "
        f"tardiness={sjf_val['tardiness']:.0f}"
    )
    print("Selection:", config["status"])
    print("Weights:", config["weights"])
    print("Saved:", path)


if __name__ == "__main__":
    main()

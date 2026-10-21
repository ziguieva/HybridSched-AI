"""Generate reproducible HPC runtime prediction datasets."""

import csv
import random
from pathlib import Path
from statistics import mean

from compare_schedulers import generate_workload


SEED = 20261021
RUNTIME_SEED = 20261022
N_JOBS = 1000

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic"
)


def generate_actual_runtime(job, rng):
    """Simulate execution time without exposing it to schedulers."""

    resources = job.resources

    # Synthetic relationship between workload and runtime.
    base_ratio = (
        0.46
        + 0.045 * resources.gpus
        + 0.009 * resources.cpu_cores
        + 0.002 * resources.memory_gib
    )

    noise = rng.gauss(0, 0.08)

    runtime_ratio = min(
        0.98,
        max(0.25, base_ratio + noise),
    )

    actual_runtime = round(
        job.requested_walltime_s * runtime_ratio
    )

    return max(1, actual_runtime)


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    rng = random.Random(RUNTIME_SEED)

    jobs = generate_workload(
        n_jobs=N_JOBS,
        seed=SEED,
    )

    features = []
    actuals = []

    for job in jobs:
        actual_runtime = generate_actual_runtime(
            job,
            rng,
        )

        features.append({
            "job_id": job.job_id,
            "arrival_s": job.arrival_s,
            "requested_walltime_s": job.requested_walltime_s,
            "cpu_cores": job.resources.cpu_cores,
            "gpus": job.resources.gpus,
            "memory_gib": job.resources.memory_gib,
            "priority": job.priority,
        })

        actuals.append({
            "job_id": job.job_id,
            "actual_runtime_s": actual_runtime,
        })

    assert len(features) == N_JOBS
    assert len(actuals) == N_JOBS

    assert len({
        row["job_id"] for row in features
    }) == N_JOBS

    for feature, actual in zip(features, actuals):
        assert feature["job_id"] == actual["job_id"]

        assert (
            0 < actual["actual_runtime_s"]
            <= feature["requested_walltime_s"]
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_path = OUTPUT_DIR / "jobs.csv"
    actuals_path = OUTPUT_DIR / "actual_runtime.csv"

    write_csv(features_path, features)
    write_csv(actuals_path, actuals)

    requested = [
        row["requested_walltime_s"]
        for row in features
    ]

    actual = [
        row["actual_runtime_s"]
        for row in actuals
    ]

    print("=" * 60)
    print("HybridSched-AI | RUNTIME DATASET GENERATION")
    print("=" * 60)

    print(f"Total jobs:             {N_JOBS}")
    print(f"Workload seed:          {SEED}")
    print(f"Runtime seed:           {RUNTIME_SEED}")
    print(f"Average requested:      {mean(requested):.2f} s")
    print(f"Average actual:         {mean(actual):.2f} s")

    print(f"Features:               {features_path}")
    print(f"Actual runtimes:        {actuals_path}")

    print("=" * 60)
    print("Dataset generated successfully.")


if __name__ == "__main__":
    main()

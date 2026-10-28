"""Frozen, independently generated synthetic HPC evaluation."""

import csv
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

from experiments.benchmark_multiobjective import run_policy
from hybridsched.job import Job
from hybridsched.resources import Resources
from hybridsched.ml_predictor import MLRuntimePredictor


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/config/step24_protocol.json"
OUTPUT = ROOT / "results/step24_independent_benchmark.csv"
MANIFEST = ROOT / "results/step24_manifest.json"


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_workload(seed, config):
    """Generate new jobs without reading previous test workloads."""
    rng = random.Random(seed)
    settings = config["workload"]

    jobs = []
    actual = {}
    current_arrival = 0

    for index in range(config["jobs_per_scenario"]):
        gap = rng.expovariate(
            1 / settings["interarrival_mean_s"]
        )
        current_arrival += max(1, round(gap))

        requested = rng.randint(
            settings["requested_walltime_min_s"],
            settings["requested_walltime_max_s"],
        )

        gpus = rng.choice([1, 1, 2, 2, 3, 4])
        cpu = rng.choice([2, 4, 6, 8, 12])
        memory = rng.choice([8, 16, 24, 32, 48])
        priority = rng.randint(0, 2)

        margin = rng.randint(
            settings["deadline_margin_min_s"],
            settings["deadline_margin_max_s"],
        )

        job_id = f"NEW_{seed}_{index:04d}"

        job = Job(
            job_id=job_id,
            arrival_s=current_arrival,
            requested_walltime_s=requested,
            resources=Resources(
                cpu_cores=cpu,
                gpus=gpus,
                memory_gib=memory,
            ),
            priority=priority,
            deadline_s=current_arrival + requested + margin,
        )

        # Synthetic execution model; never supplied to the scheduler.
        ratio = rng.uniform(
            settings["runtime_ratio_min"],
            settings["runtime_ratio_max"],
        )
        gpu_factor = 1.0 + 0.03 * (gpus - 1)
        runtime = max(1, round(requested * ratio * gpu_factor))

        jobs.append(job)
        actual[job_id] = runtime

    return jobs, actual


def main():
    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)

    if config["status"] != "frozen_before_evaluation":
        raise ValueError("Protocol is not frozen")

    policies = config["policies"]
    predictor = MLRuntimePredictor()
    rows = []

    start = config["seed_start"]
    stop = start + config["seed_count"]

    for seed in range(start, stop):
        jobs, actual = generate_workload(seed, config)

        for policy in policies:
            metrics = run_policy(
                jobs, actual, predictor, policy
            )

            rows.append({
                "seed": seed,
                "policy": policy,
                **metrics,
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    model_paths = [
        ROOT / "results/runtime_xgboost.json",
        ROOT / "results/runtime_quantiles.json",
    ]

    manifest = {
        "experiment": config["experiment"],
        "protocol_sha256": hashlib.sha256(
            config_bytes
        ).hexdigest(),
        "config": config,
        "models_sha256": {
            path.name: sha256_file(path)
            for path in model_paths
        },
        "results_sha256": sha256_file(OUTPUT),
        "total_simulations": len(rows),
        "evaluation_type": "independent synthetic generation",
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("=" * 90)
    print("STEP 24.2 | INDEPENDENT SYNTHETIC BENCHMARK")
    print("=" * 90)
    print(
        f"{'Policy':<14}{'Late':>10}"
        f"{'Tardiness':>15}{'Cost':>15}"
        f"{'Wait':>12}{'Makespan':>13}"
    )

    for policy in policies:
        group = [r for r in rows if r["policy"] == policy]

        print(
            f"{policy:<14}"
            f"{mean(r['late_jobs'] for r in group):>10.2f}"
            f"{mean(r['tardiness_s'] for r in group):>15.1f}"
            f"{mean(r['cost_units'] for r in group):>15.1f}"
            f"{mean(r['mean_wait_s'] for r in group):>12.1f}"
            f"{mean(r['makespan_s'] for r in group):>13.1f}"
        )

    print(f"\nSimulations: {len(rows)}")
    print(f"Saved: {OUTPUT}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()

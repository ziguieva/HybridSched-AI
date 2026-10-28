"""Decision-focused scheduling under synthetic workload shifts."""

import csv
import random
from dataclasses import replace
from pathlib import Path
from statistics import mean

from experiments.benchmark_ml_dynamic import load_test_data
from hybridsched.decision_focused import DecisionFocusedSimulator
from hybridsched.infrastructure import default_infrastructure
from hybridsched.ml_deadline import MLDeadlineSimulator
from hybridsched.ml_predictor import MLRuntimePredictor
from hybridsched.resources import Resources


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "decision_shift_benchmark.csv"

SCENARIOS = ("BURST", "GPU_HEAVY", "COMBINED")
POLICIES = ("REQUESTED", "P50", "P90", "DECISION")


def transform_workload(
    jobs, actual, scenario, seed
):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown workload shift")

    rng = random.Random(seed)
    transformed = []
    disturbed_actual = {}

    first_arrival = min(j.arrival_s for j in jobs)

    burst = scenario in ("BURST", "COMBINED")
    gpu_heavy = scenario in ("GPU_HEAVY", "COMBINED")

    for job in jobs:
        arrival = job.arrival_s

        if burst:
            arrival = (
                first_arrival
                + round(
                    0.70 * (arrival - first_arrival)
                )
                + rng.randint(0, 20)
            )

        gpus = job.resources.gpus

        if gpu_heavy and rng.random() < 0.30:
            gpus = min(4, gpus + 1)

        resources = Resources(
            cpu_cores=job.resources.cpu_cores,
            gpus=gpus,
            memory_gib=job.resources.memory_gib,
        )

        transformed_job = replace(
            job,
            arrival_s=arrival,
            resources=resources,
            deadline_s=(
                arrival
                + job.requested_walltime_s
                + 300
            ),
        )

        transformed.append(transformed_job)

        disturbed_actual[job.job_id] = max(
            1.0,
            actual[job.job_id]
            * rng.uniform(0.85, 1.15),
        )

    return transformed, disturbed_actual


def evaluate(jobs, actual, policy, predictor):
    if policy == "DECISION":
        simulator = DecisionFocusedSimulator(
            jobs,
            default_infrastructure(),
            actual,
            predictor,
        )
    else:
        mode = policy.lower()

        simulator = MLDeadlineSimulator(
            jobs,
            default_infrastructure(),
            actual_runtimes=actual,
            predictor=(
                predictor
                if policy != "REQUESTED"
                else None
            ),
            mode=mode,
        )

    results = simulator.run()

    if len(results) != len(jobs):
        raise RuntimeError("Incomplete simulation")

    if len({r.job_id for r in results}) != len(jobs):
        raise RuntimeError("Duplicate job result")

    return {
        "late_jobs": sum(
            r.tardiness_s > 0 for r in results
        ),
        "tardiness_s": sum(
            r.tardiness_s for r in results
        ),
        "mean_wait_s": mean(
            r.waiting_s for r in results
        ),
        "makespan_s": max(
            r.finish_s for r in results
        ),
        "cost_units": sum(
            r.cost_units for r in results
        ),
        "cloud_jobs": sum(
            r.cluster == "gcp-gpu"
            for r in results
        ),
    }


def main():
    jobs, actual = load_test_data()
    predictor = MLRuntimePredictor()

    rows = []

    for scenario in SCENARIOS:
        for index in range(10):
            seed = 20001 + index

            shifted, actual_shifted = (
                transform_workload(
                    jobs, actual, scenario, seed
                )
            )

            for policy in POLICIES:
                result = evaluate(
                    shifted,
                    actual_shifted,
                    policy,
                    predictor,
                )
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "policy": policy,
                    **result,
                })

    OUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print("STEP 20.2 | DISTRIBUTION-SHIFT BENCHMARK")
    print("=" * 100)

    for scenario in SCENARIOS:
        print(f"\n{scenario}")
        print(
            f"{'Policy':<14}"
            f"{'Late':>9}"
            f"{'Tardiness':>15}"
            f"{'Wait':>12}"
            f"{'Cost':>15}"
            f"{'Makespan':>13}"
        )

        for policy in POLICIES:
            group = [
                r for r in rows
                if (
                    r["scenario"] == scenario
                    and r["policy"] == policy
                )
            ]

            print(
                f"{policy:<14}"
                f"{mean(r['late_jobs'] for r in group):>9.2f}"
                f"{mean(r['tardiness_s'] for r in group):>15.1f}"
                f"{mean(r['mean_wait_s'] for r in group):>12.1f}"
                f"{mean(r['cost_units'] for r in group):>15.1f}"
                f"{mean(r['makespan_s'] for r in group):>13.1f}"
            )

    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()

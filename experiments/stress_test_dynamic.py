"""Reproducible dynamic HPC stress testing."""

import csv
import random
import time
from pathlib import Path
from statistics import mean

from hybridsched.dynamic import DynamicHybridSimulator
from hybridsched.infrastructure import default_infrastructure
from hybridsched.job import Job
from hybridsched.resources import Resources


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "stress_test_dynamic.csv"

SEED = 20261021
WORKLOAD_SIZES = (50, 100, 200)

POLICIES = {
    "FCFS": "fcfs",
    "SJF": "sjf",
    "ROLLING-CP": "rolling",
}


def generate_workload(count, seed):
    rng = random.Random(seed)

    jobs = []
    arrival = 0

    for index in range(count):
        # Dense arrivals create resource contention.
        arrival += rng.randint(1, 15)

        duration = rng.choice(
            [120, 300, 600, 900, 1200]
        )

        gpus = rng.choice([1, 2, 3, 4])
        cpus = rng.choice([2, 4, 8])

        memory = 4 + 8 * gpus + cpus

        deadline = (
            arrival
            + duration
            + rng.randint(100, 600)
        )

        jobs.append(
            Job(
                job_id=f"J{index:04d}",
                arrival_s=arrival,
                requested_walltime_s=duration,
                resources=Resources(
                    cpu_cores=cpus,
                    gpus=gpus,
                    memory_gib=memory,
                ),
                priority=rng.randint(0, 3),
                deadline_s=deadline,
            )
        )

    return jobs


def evaluate(count, policy_name, policy):
    jobs = generate_workload(
        count,
        SEED + count,
    )

    simulator = DynamicHybridSimulator(
        jobs=jobs,
        clusters=default_infrastructure(),
        policy=policy,
        time_limit_s=0.5,
        cost_weight=1,
        deadline_weight=10,
    )

    start = time.perf_counter()
    results = simulator.run()
    elapsed = time.perf_counter() - start

    assert len(results) == count

    job_ids = {job.job_id for job in jobs}
    result_ids = {result.job_id for result in results}

    assert job_ids == result_ids

    makespan = max(
        result.finish_s for result in results
    )

    return {
        "jobs": count,
        "policy": policy_name,
        "makespan_s": makespan,
        "mean_wait_s": round(
            mean(r.waiting_s for r in results), 2
        ),
        "late_jobs": sum(
            r.tardiness_s > 0 for r in results
        ),
        "total_tardiness_s": sum(
            r.tardiness_s for r in results
        ),
        "cost_units": sum(
            r.cost_units for r in results
        ),
        "cloud_jobs": sum(
            r.cluster == "gcp-gpu"
            for r in results
        ),
        "runtime_wall_s": round(elapsed, 3),
    }


def main():
    records = []

    print("=" * 115)
    print("HybridSched-AI | DYNAMIC HPC STRESS TEST")
    print("=" * 115)
    print(f"Seed: {SEED}")
    print("Cluster configuration: LOCAL + HPC + GCP")
    print("CP-SAT time limit: 0.5 s per decision")
    print("-" * 115)

    print(
        f"{'Jobs':>6}"
        f"{'Policy':<14}"
        f"{'Makespan':>12}"
        f"{'Mean Wait':>13}"
        f"{'Late':>9}"
        f"{'Cost':>13}"
        f"{'Cloud':>10}"
        f"{'Wall(s)':>12}"
    )

    print("-" * 115)

    for count in WORKLOAD_SIZES:
        for name, policy in POLICIES.items():

            row = evaluate(
                count,
                name,
                policy,
            )

            records.append(row)

            print(
                f"{row['jobs']:>6}"
                f"{row['policy']:<14}"
                f"{row['makespan_s']:>12}"
                f"{row['mean_wait_s']:>13.2f}"
                f"{row['late_jobs']:>9}"
                f"{row['cost_units']:>13}"
                f"{row['cloud_jobs']:>10}"
                f"{row['runtime_wall_s']:>12.3f}"
            )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )

        writer.writeheader()
        writer.writerows(records)

    print("=" * 115)
    print(f"Results saved: {OUTPUT}")
    print("Stress testing completed successfully.")


if __name__ == "__main__":
    main()

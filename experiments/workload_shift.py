"""Controlled workload distribution shifts."""

import random
from dataclasses import replace

from experiments.stress_test_dynamic import generate_workload


MODES = ("COUPLED", "INDEPENDENT", "TIGHT")


def generate_shifted_workload(
    count: int,
    seed: int,
    mode: str,
):
    if mode not in MODES:
        raise ValueError(f"Unknown workload mode: {mode}")

    base_jobs = generate_workload(count, seed)
    rng = random.Random(seed + 100_003)

    jobs = []

    for job in base_jobs:
        duration = job.requested_walltime_s

        if mode == "COUPLED":
            relative_deadline = (
                duration + rng.randint(100, 600)
            )

        elif mode == "INDEPENDENT":
            relative_deadline = rng.choice(
                [300, 500, 700, 900, 1200, 1600]
            )

        else:
            relative_deadline = (
                duration + rng.randint(0, 150)
            )

        jobs.append(
            replace(
                job,
                deadline_s=(
                    job.arrival_s + relative_deadline
                ),
            )
        )

    return jobs


if __name__ == "__main__":
    print("HybridSched-AI | DISTRIBUTION SHIFT")
    print("-" * 65)

    for mode in MODES:
        jobs = generate_shifted_workload(
            count=100,
            seed=42,
            mode=mode,
        )

        relative = [
            job.deadline_s - job.arrival_s
            for job in jobs
        ]

        print(
            f"{mode:<14}"
            f"Jobs={len(jobs):>4}  "
            f"Mean relative deadline="
            f"{sum(relative) / len(relative):>8.1f}s"
        )

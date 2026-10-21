"""Benchmark online dynamic hybrid HPC schedulers."""

import csv
from pathlib import Path
from statistics import mean

from hybridsched.dynamic import DynamicHybridSimulator
from hybridsched.infrastructure import default_infrastructure
from hybridsched.job import Job
from hybridsched.resources import Resources


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "benchmark_dynamic.csv"


def make_workload():
    specifications = [
        ("A", 0, 900, 2, 4, 1100),
        ("B", 40, 1200, 3, 8, 1500),
        ("C", 80, 300, 1, 2, 500),
        ("D", 120, 200, 2, 4, 550),
        ("E", 160, 600, 2, 4, 950),
        ("F", 200, 180, 1, 2, 500),
        ("G", 240, 750, 2, 6, 1200),
        ("H", 280, 120, 1, 2, 550),
        ("I", 320, 800, 3, 8, 1400),
        ("J", 360, 250, 2, 4, 800),
        ("K", 400, 500, 2, 4, 1100),
        ("L", 440, 350, 1, 2, 950),
    ]

    return [
        Job(
            job_id=jid,
            arrival_s=arrival,
            requested_walltime_s=duration,
            resources=Resources(
                cpu,
                gpu,
                4 + 8 * gpu + cpu,
            ),
            deadline_s=deadline,
        )
        for jid, arrival, duration, gpu, cpu, deadline
        in specifications
    ]


def main():
    jobs = make_workload()
    clusters = default_infrastructure()

    configurations = [
        ("FCFS-HYBRID", "fcfs"),
        ("SJF-HYBRID", "sjf"),
        ("ROLLING-CP", "rolling"),
    ]

    rows = []

    print("=" * 98)
    print("HybridSched-AI | DYNAMIC HYBRID SCHEDULING")
    print("=" * 98)
    print(f"Total jobs: {len(jobs)}")
    print("All policies use identical clusters and arrivals.")
    print("-" * 98)

    print(
        f"{'Policy':<17}"
        f"{'Makespan':>12}"
        f"{'Avg Wait':>13}"
        f"{'Late Jobs':>12}"
        f"{'Cost':>13}"
        f"{'Cloud Jobs':>13}"
    )

    print("-" * 98)

    for name, policy in configurations:
        simulator = DynamicHybridSimulator(
            jobs=jobs,
            clusters=clusters,
            policy=policy,
            time_limit_s=2,
            cost_weight=1,
            deadline_weight=10,
        )

        results = simulator.run()

        assert len(results) == len(jobs)

        makespan = max(
            r.finish_s for r in results
        )

        avg_wait = mean(
            r.waiting_s for r in results
        )

        late = sum(
            r.tardiness_s > 0 for r in results
        )

        cost = sum(
            r.cost_units for r in results
        )

        cloud_jobs = sum(
            r.cluster == "gcp-gpu"
            for r in results
        )

        record = {
            "policy": name,
            "jobs": len(results),
            "makespan_s": makespan,
            "avg_wait_s": round(avg_wait, 2),
            "late_jobs": late,
            "total_tardiness_s": sum(
                r.tardiness_s for r in results
            ),
            "cost_units": cost,
            "cloud_jobs": cloud_jobs,
        }

        rows.append(record)

        print(
            f"{name:<17}"
            f"{makespan:>12}"
            f"{avg_wait:>13.2f}"
            f"{late:>12}"
            f"{cost:>13}"
            f"{cloud_jobs:>13}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 98)
    print(f"CSV saved: {OUTPUT}")
    print("Dynamic benchmark completed successfully.")


if __name__ == "__main__":
    main()

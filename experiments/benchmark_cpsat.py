"""Compare offline CP-SAT planning to online HPC policies."""

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.metrics import compute_metrics
from hybridsched.optimizer import optimize_schedule
from hybridsched.resources import Resources
from hybridsched.simulation import FCFSSimulator
from hybridsched.sjf import SJFSimulator


CAPACITY = Resources(16, 4, 64)

JOBS = [
    Job("J001", 0, 900, Resources(4, 2, 16)),
    Job("J002", 60, 1200, Resources(8, 3, 32)),
    Job("J003", 120, 300, Resources(2, 1, 8)),
    Job("J004", 180, 200, Resources(4, 2, 16)),
    Job("J005", 240, 600, Resources(4, 2, 16)),
    Job("J006", 300, 180, Resources(2, 1, 8)),
    Job("J007", 360, 750, Resources(6, 2, 24)),
    Job("J008", 420, 120, Resources(2, 1, 8)),
]


def main():
    print("=" * 82)
    print("HybridSched-AI | CP-SAT OPTIMIZATION BENCHMARK")
    print("=" * 82)

    policies = {
        "FCFS": FCFSSimulator,
        "SJF": SJFSimulator,
        "EASY": EASYBackfillSimulator,
    }

    comparisons = []

    for name, cls in policies.items():
        cluster = Cluster(name, CAPACITY)
        results = cls(cluster, JOBS).run()
        metrics = compute_metrics(
            JOBS, results, CAPACITY
        )

        comparisons.append(
            (name, metrics, "ONLINE")
        )

    optimized = optimize_schedule(
        JOBS,
        CAPACITY,
        time_limit_s=10,
    )

    print(f"CP-SAT status: {optimized.status}")

    if not optimized.results:
        print("No feasible CP-SAT schedule returned.")
        return

    cp_metrics = compute_metrics(
        JOBS,
        optimized.results,
        CAPACITY,
    )

    comparisons.append(
        ("CP-SAT", cp_metrics, "OFFLINE")
    )

    print("-" * 82)
    print(
        f"{'Algorithm':<13}"
        f"{'Mode':<11}"
        f"{'Makespan':>12}"
        f"{'Avg Wait':>13}"
        f"{'GPU Util':>13}"
    )
    print("-" * 82)

    for name, m, mode in comparisons:
        print(
            f"{name:<13}"
            f"{mode:<11}"
            f"{m.makespan_s:>12}"
            f"{m.average_waiting_s:>13.2f}"
            f"{m.gpu_utilization_pct:>12.2f}%"
        )

    print("-" * 82)
    print("CP-SAT JOB SCHEDULE")

    for result in optimized.results:
        print(
            f"{result.job_id}: "
            f"start={result.start_s}s, "
            f"finish={result.finish_s}s, "
            f"wait={result.waiting_s}s"
        )

    print("=" * 82)


if __name__ == "__main__":
    main()

"""Compare local-only and hybrid HPC optimization."""

from statistics import mean

from hybridsched.hybrid_optimizer import optimize_hybrid
from hybridsched.infrastructure import default_infrastructure
from hybridsched.job import Job
from hybridsched.resources import Resources


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
    infrastructure = default_infrastructure()

    scenarios = {
        "LOCAL ONLY": infrastructure[:1],
        "HYBRID": infrastructure,
    }

    print("=" * 86)
    print("HybridSched-AI | HYBRID CLOUD/HPC BENCHMARK")
    print("=" * 86)

    print(
        f"{'Scenario':<16}"
        f"{'Status':<12}"
        f"{'Makespan':>12}"
        f"{'Avg Wait':>13}"
        f"{'Cost Units':>15}"
    )
    print("-" * 86)

    for name, clusters in scenarios.items():
        result = optimize_hybrid(
            jobs=JOBS,
            clusters=clusters,
            time_limit_s=15,
        )

        if not result.results:
            print(f"{name}: {result.status}")
            continue

        average_wait = mean(
            r.waiting_s for r in result.results
        )

        print(
            f"{name:<16}"
            f"{result.status:<12}"
            f"{result.makespan_s:>12}"
            f"{average_wait:>13.2f}"
            f"{result.total_cost_units:>15}"
        )

        print(f"\n{name} — PLACEMENT")

        for job in result.results:
            print(
                f"{job.job_id} -> {job.cluster:<12}"
                f" start={job.start_s:>5}s"
                f" finish={job.finish_s:>5}s"
                f" cost={job.cost_units}"
            )

        print("-" * 86)

    print("Hybrid benchmark completed.")
    print("=" * 86)


if __name__ == "__main__":
    main()

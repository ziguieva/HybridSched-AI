"""Cloud bursting experiment under increasing workloads."""

from hybridsched.hybrid_optimizer import optimize_hybrid
from hybridsched.infrastructure import default_infrastructure
from hybridsched.job import Job
from hybridsched.resources import Resources


def generate_jobs(count):
    return [
        Job(
            job_id=f"J{i:03d}",
            arrival_s=0,
            requested_walltime_s=600,
            resources=Resources(4, 2, 16),
            deadline_s=1000,
        )
        for i in range(count)
    ]


def main():
    clusters = default_infrastructure()

    print("=" * 88)
    print("HybridSched-AI | CLOUD BURSTING EXPERIMENT")
    print("=" * 88)

    print(
        f"{'Jobs':>7}"
        f"{'Mode':<15}"
        f"{'Status':<12}"
        f"{'Makespan':>12}"
        f"{'Late':>8}"
        f"{'Cloud Jobs':>13}"
        f"{'Cost':>13}"
    )
    print("-" * 88)

    for count in (4, 8, 12):
        jobs = generate_jobs(count)

        for name, available_clusters in (
            ("ON-PREMISE", clusters[:2]),
            ("HYBRID", clusters),
        ):
            result = optimize_hybrid(
                jobs,
                available_clusters,
                time_limit_s=15,
                wait_weight=2,
                makespan_weight=5,
                cost_weight=0,
                deadline_weight=30,
            )

            if not result.results:
                print(
                    f"{count:>7}"
                    f"{name:<15}"
                    f"{result.status:<12}"
                )
                continue

            cloud_jobs = sum(
                r.cluster == "gcp-gpu"
                for r in result.results
            )

            print(
                f"{count:>7}"
                f"{name:<15}"
                f"{result.status:<12}"
                f"{result.makespan_s:>12}"
                f"{result.deadline_violations:>8}"
                f"{cloud_jobs:>13}"
                f"{result.total_cost_units:>13}"
            )

    print("=" * 88)
    print("Cloud bursting benchmark completed.")


if __name__ == "__main__":
    main()

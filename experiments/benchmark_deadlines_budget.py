"""Evaluate deadlines and cost constraints."""

from statistics import mean

from hybridsched.hybrid_optimizer import optimize_hybrid
from hybridsched.infrastructure import default_infrastructure
from hybridsched.job import Job
from hybridsched.resources import Resources


JOBS = [
    Job("A", 0, 900, Resources(4, 2, 16),
        deadline_s=1200),
    Job("B", 0, 900, Resources(4, 2, 16),
        deadline_s=1200),
    Job("C", 0, 600, Resources(4, 2, 16),
        deadline_s=900),
    Job("D", 100, 600, Resources(4, 2, 16),
        deadline_s=1000),
    Job("E", 200, 300, Resources(2, 1, 8),
        deadline_s=700),
    Job("F", 300, 500, Resources(4, 2, 16),
        deadline_s=1000),
]


def main():
    clusters = default_infrastructure()

    scenarios = [
        (
            "BASELINE",
            {},
        ),
        (
            "DEADLINE-AWARE",
            {"deadline_weight": 30},
        ),
        (
            "BUDGET-LIMITED",
            {
                "deadline_weight": 30,
                "max_budget_units": 13000,
                "max_cloud_budget_units": 0,
            },
        ),
        (
            "HARD-DEADLINES",
            {
                "hard_deadlines": True,
                "max_budget_units": 25000,
            },
        ),
    ]

    print("=" * 93)
    print("HybridSched-AI | DEADLINES & BUDGET BENCHMARK")
    print("=" * 93)

    print(
        f"{'Scenario':<19}"
        f"{'Status':<12}"
        f"{'Makespan':>12}"
        f"{'Avg Wait':>12}"
        f"{'Late Jobs':>12}"
        f"{'Cost':>12}"
        f"{'Cloud Cost':>14}"
    )
    print("-" * 93)

    for name, kwargs in scenarios:
        result = optimize_hybrid(
            JOBS,
            clusters,
            time_limit_s=15,
            **kwargs,
        )

        if not result.results:
            print(f"{name:<19}{result.status:<12}")
            continue

        avg_wait = mean(
            r.waiting_s for r in result.results
        )

        print(
            f"{name:<19}"
            f"{result.status:<12}"
            f"{result.makespan_s:>12}"
            f"{avg_wait:>12.2f}"
            f"{result.deadline_violations:>12}"
            f"{result.total_cost_units:>12}"
            f"{result.cloud_cost_units:>14}"
        )

        if kwargs.get("max_budget_units") is not None:
            assert (
                result.total_cost_units
                <= kwargs["max_budget_units"]
            )

        if kwargs.get("max_cloud_budget_units") is not None:
            assert (
                result.cloud_cost_units
                <= kwargs["max_cloud_budget_units"]
            )

        if kwargs.get("hard_deadlines"):
            assert result.deadline_violations == 0

    print("=" * 93)
    print("Deadline and budget experiments completed.")


if __name__ == "__main__":
    main()

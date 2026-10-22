"""Feasibility study across workload distributions."""

from experiments.workload_shift import (
    MODES,
    generate_shifted_workload,
)

from hybridsched.deadline_feasibility import (
    analyze_deadline_feasibility,
)
from hybridsched.infrastructure import default_infrastructure


def main():
    clusters = default_infrastructure()

    print("=" * 86)
    print("HybridSched-AI | DEADLINE FEASIBILITY")
    print("=" * 86)
    print(
        f"{'Jobs':>6}"
        f"{'Seed':>8}"
        f"{'Mode':<16}"
        f"{'Unavoidable late':>20}"
        f"{'Minimum tardiness':>22}"
    )
    print("-" * 86)

    for count in (100, 200):
        for seed in (42, 123, 2026):
            for mode in MODES:
                jobs = generate_shifted_workload(
                    count, seed, mode
                )

                analysis = analyze_deadline_feasibility(
                    jobs, clusters
                )

                print(
                    f"{count:>6}"
                    f"{seed:>8}"
                    f"{mode:<16}"
                    f"{analysis['intrinsically_late']:>20}"
                    f"{analysis['minimum_tardiness_s']:>22}"
                )

    print("=" * 86)


if __name__ == "__main__":
    main()

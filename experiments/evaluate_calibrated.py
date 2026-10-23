"""Held-out paired evaluation of scheduling priorities."""

import csv
import json
import random
from pathlib import Path
from statistics import mean

from experiments.workload_shift import (
    MODES,
    generate_shifted_workload,
)
from hybridsched.calibrated_priority import (
    CalibratedPrioritySimulator,
    PriorityWeights,
)
from hybridsched.deadline_aware import (
    DeadlineAwareSimulator,
)
from hybridsched.dynamic import DynamicHybridSimulator
from hybridsched.hybrid_priority import (
    HybridPrioritySimulator,
)
from hybridsched.infrastructure import (
    default_infrastructure,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
CONFIG = OUT / "selected_priority.json"

METRICS = (
    "late",
    "tardiness",
    "makespan",
    "cost",
    "mean_wait",
)


def run_policy(policy, jobs, weights):
    clusters = default_infrastructure()

    if policy == "SJF":
        sim = DynamicHybridSimulator(
            jobs, clusters, policy="sjf"
        )
    elif policy == "EDF":
        sim = DeadlineAwareSimulator(
            jobs, clusters, priority_policy="edf"
        )
    elif policy == "HYBRID_ORIGINAL":
        sim = HybridPrioritySimulator(jobs, clusters)
    elif policy == "SELECTED":
        if weights is None:
            sim = DynamicHybridSimulator(
                jobs, clusters, policy="sjf"
            )
        else:
            sim = CalibratedPrioritySimulator(
                jobs, clusters, weights=weights
            )
    else:
        raise ValueError(policy)

    results = sim.run()

    if len(results) != len(jobs):
        raise RuntimeError("Incomplete experiment")

    return {
        "late": sum(r.tardiness_s > 0 for r in results),
        "tardiness": sum(r.tardiness_s for r in results),
        "makespan": max(r.finish_s for r in results),
        "cost": sum(r.cost_units for r in results),
        "mean_wait": mean(
            r.waiting_s for r in results
        ),
    }


def relative_gain_ci(pairs, seed=20261023):
    """Paired bootstrap percentile confidence interval.

    Each pair is a (baseline, candidate) observation
    aggregated at the random-seed level.
    """
    if not pairs:
        raise ValueError("Empty paired sample")

    rng = random.Random(seed)
    n = len(pairs)

    baseline = mean(x for x, _ in pairs)
    candidate = mean(y for _, y in pairs)

    if baseline <= 0:
        raise ValueError("Nonpositive baseline mean")

    observed = 100 * (baseline - candidate) / baseline

    samples = []
    for _ in range(3000):
        chosen = [
            pairs[rng.randrange(n)]
            for _ in range(n)
        ]

        b = mean(x for x, _ in chosen)
        c = mean(y for _, y in chosen)

        if b > 0:
            samples.append(100 * (b - c) / b)

    samples.sort()
    lower = samples[int(0.025 * (len(samples) - 1))]
    upper = samples[int(0.975 * (len(samples) - 1))]

    return observed, lower, upper


def main():
    config = json.loads(CONFIG.read_text())

    expected_seeds = list(range(1001, 1031))
    seeds = config["test_seeds"]

    if seeds != expected_seeds:
        raise ValueError("Unexpected test seed configuration")

    weights = (
        PriorityWeights(**config["weights"])
        if config["weights"] is not None
        else None
    )

    policies = (
        "SJF",
        "EDF",
        "HYBRID_ORIGINAL",
        "SELECTED",
    )

    records = []

    print("=" * 105)
    print("HybridSched-AI | HELD-OUT STATISTICAL EVALUATION")
    print("=" * 105)
    print("Selection:", config["status"])
    print("Weights:", config["weights"])
    print("Independent test seeds:", len(seeds))

    for seed in seeds:
        for mode in MODES:
            jobs = generate_shifted_workload(
                config["n_jobs"], seed, mode
            )

            for policy in policies:
                metrics = run_policy(
                    policy, jobs, weights
                )

                records.append({
                    "seed": seed,
                    "mode": mode,
                    "policy": policy,
                    **metrics,
                })

        print(f"Completed seed {seed}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT / "calibrated_test.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)

    # Aggregate by seed. The three deadline regimes
    # remain grouped within each bootstrap observation.
    seed_metrics = {}

    for seed in seeds:
        for policy in policies:
            group = [
                row for row in records
                if row["seed"] == seed
                and row["policy"] == policy
            ]

            seed_metrics[(seed, policy)] = {
                metric: mean(
                    row[metric] for row in group
                )
                for metric in METRICS
            }

    summary = []

    for policy in policies:
        for metric in METRICS:
            pairs = [
                (
                    seed_metrics[(seed, "SJF")][metric],
                    seed_metrics[(seed, policy)][metric],
                )
                for seed in seeds
            ]

            gain, lower, upper = relative_gain_ci(
                pairs
            )

            summary.append({
                "policy": policy,
                "metric": metric,
                "sjf_mean": mean(
                    x for x, _ in pairs
                ),
                "candidate_mean": mean(
                    y for _, y in pairs
                ),
                "gain_pct": gain,
                "ci95_lower_pct": lower,
                "ci95_upper_pct": upper,
            })

    with (OUT / "calibrated_summary.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary[0])
        )
        writer.writeheader()
        writer.writerows(summary)

    print("-" * 105)
    print(
        f"{'Policy':<18}"
        f"{'Metric':<14}"
        f"{'SJF mean':>13}"
        f"{'Candidate':>13}"
        f"{'Gain %':>12}"
        f"{'95% CI':>27}"
    )

    for row in summary:
        print(
            f"{row['policy']:<18}"
            f"{row['metric']:<14}"
            f"{row['sjf_mean']:>13.1f}"
            f"{row['candidate_mean']:>13.1f}"
            f"{row['gain_pct']:>12.2f}"
            f"  [{row['ci95_lower_pct']:.2f}, "
            f"{row['ci95_upper_pct']:.2f}]"
        )

    print("=" * 105)
    print(f"Test simulations: {len(records)}")
    print("Raw data: results/calibrated_test.csv")
    print("Summary: results/calibrated_summary.csv")


if __name__ == "__main__":
    main()

"""Generate a reproducible Markdown research report."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "results/STEP24_RESEARCH_REPORT.md"

METRICS = (
    "late_jobs",
    "tardiness_s",
    "cost_units",
    "mean_wait_s",
    "makespan_s",
)


def paired_ci(values, seed=24024, repeats=3000):
    rng = random.Random(seed)
    n = len(values)
    if n < 2:
        raise ValueError("At least two scenarios required")

    bootstrap = sorted(
        mean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(repeats)
    )

    return (
        mean(values),
        bootstrap[int(0.025 * (repeats - 1))],
        bootstrap[int(0.975 * (repeats - 1))],
    )


def main():
    with (
        RESULTS / "step24_independent_benchmark.csv"
    ).open(newline="") as f:
        rows = list(csv.DictReader(f))

    manifest = json.loads(
        (RESULTS / "step24_manifest.json").read_text()
    )

    config = manifest["config"]
    policies = config["policies"]
    seeds = list(range(
        config["seed_start"],
        config["seed_start"] + config["seed_count"],
    ))

    indexed = {
        (int(r["seed"]), r["policy"]): r
        for r in rows
    }

    if len(indexed) != len(seeds) * len(policies):
        raise ValueError("Incomplete or duplicate experimental data")

    lines = [
        "# HybridSched-AI — STEP 24 Research Report",
        "",
        "## Research question",
        "",
        "How robust are Requested, P90, Slack and MULTI "
        "on newly generated synthetic HPC workloads?",
        "",
        "## Experimental protocol",
        "",
        f"- Protocol version: {config['protocol_version']}",
        f"- Scenarios: {len(seeds)}",
        f"- Jobs per scenario: {config['jobs_per_scenario']}",
        f"- Simulations: {len(rows)}",
        f"- Protocol SHA256: `{manifest['protocol_sha256']}`",
        "",
        "The models were not retrained. Policies and "
        "coefficients were fixed before this evaluation.",
        "",
        "## Mean performance",
        "",
        "| Policy | Late jobs | Tardiness (s) | "
        "Cost units | Mean wait (s) | Makespan (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for policy in policies:
        group = [indexed[(seed, policy)] for seed in seeds]

        values = [
            mean(float(r[metric]) for r in group)
            for metric in METRICS
        ]

        lines.append(
            f"| {policy} | "
            + " | ".join(f"{v:,.2f}" for v in values)
            + " |"
        )

    lines += [
        "",
        "## Paired differences versus Requested",
        "",
        "Negative values favor the candidate. "
        "Intervals use paired bootstrap resampling of scenarios.",
        "",
        "| Policy | Metric | Mean difference | 95% CI |",
        "|---|---|---:|---|",
    ]

    for policy in policies:
        if policy == "REQUESTED":
            continue

        for metric in METRICS:
            differences = [
                float(indexed[(seed, policy)][metric])
                - float(indexed[(seed, "REQUESTED")][metric])
                for seed in seeds
            ]

            avg, low, high = paired_ci(differences)

            lines.append(
                f"| {policy} | {metric} | {avg:,.2f} | "
                f"[{low:,.2f}, {high:,.2f}] |"
            )

    lines += [
        "",
        "## Methodological limitations",
        "",
        "- Workloads are generated synthetically, not "
        "from independent production HPC traces.",
        "- The synthetic execution-time model is simplified.",
        "- XGBoost is evaluated under possible covariate shift "
        "relative to its original training data.",
        "- The resource cost units are simulated, not cloud prices.",
        "- Bootstrap intervals characterize variation under "
        "the specified workload generator.",
        "- Results should not be generalized to arbitrary "
        "HPC systems without external validation.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m experiments.step24_independent_benchmark",
        "python -m experiments.step24_research_report",
        "python -m pytest -q",
        "```",
        "",
    ]

    REPORT.write_text("\n".join(lines))

    print("=== STEP 24.3 | RESEARCH REPORT ===")
    print(f"Saved: {REPORT}")
    print("Paired comparisons generated.")
    print("Synthetic-validation limitations documented.")


if __name__ == "__main__":
    main()

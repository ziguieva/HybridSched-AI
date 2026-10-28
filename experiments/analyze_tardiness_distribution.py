"""Per-job tardiness distribution for established policies."""

import csv
from pathlib import Path

import numpy as np

from experiments.benchmark_ml_dynamic import load_test_data
from experiments.decision_shift_benchmark import transform_workload
from hybridsched.infrastructure import default_infrastructure
from hybridsched.ml_deadline import MLDeadlineSimulator
from hybridsched.decision_focused import DecisionFocusedSimulator
from hybridsched.slack_priority import SlackPrioritySimulator
from hybridsched.ml_predictor import MLRuntimePredictor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "tardiness_distribution.csv"


def run_policies(scenario, seed):
    jobs, actual = load_test_data()
    jobs, actual = transform_workload(
        jobs, actual, scenario, seed
    )
    predictor = MLRuntimePredictor()

    configs = {
        "REQUESTED": MLDeadlineSimulator(
            jobs, default_infrastructure(),
            actual_runtimes=actual, mode="requested",
        ),
        "P90": MLDeadlineSimulator(
            jobs, default_infrastructure(),
            actual_runtimes=actual,
            predictor=predictor, mode="p90",
        ),
        "DECISION": DecisionFocusedSimulator(
            jobs, default_infrastructure(), actual, predictor
        ),
        "SLACK": SlackPrioritySimulator(
            jobs, default_infrastructure(), actual, predictor
        ),
    }
    return {
        policy: sim.run()
        for policy, sim in configs.items()
    }


def distribution_metrics(results):
    delays = np.asarray(
        [r.tardiness_s for r in results],
        dtype=float,
    )
    late = delays[delays > 0]
    sorted_late = np.sort(late)[::-1]

    return {
        "late_jobs": int(len(late)),
        "tardiness_total": float(delays.sum()),
        "p50_all_s": float(np.percentile(delays, 50)),
        "p90_all_s": float(np.percentile(delays, 90)),
        "p95_all_s": float(np.percentile(delays, 95)),
        "max_tardiness_s": float(delays.max()),
        "mean_late_only_s": (
            float(late.mean()) if len(late) else 0.0
        ),
        "top10_share_pct": (
            float(
                100 * sorted_late[:10].sum() / delays.sum()
            )
            if delays.sum() else 0.0
        ),
    }


def main():
    rows = []
    for scenario in ("BURST", "GPU_HEAVY", "COMBINED"):
        outputs = run_policies(scenario, 20001)
        for policy, results in outputs.items():
            rows.append({
                "scenario": scenario,
                "policy": policy,
                **distribution_metrics(results),
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=== STEP 22.1 | TARDINESS DISTRIBUTION ===")
    print(
        f"{'Scenario':<13}{'Policy':<12}{'Late':>7}"
        f"{'P95':>12}{'Max':>12}{'Top10 %':>12}"
    )
    for r in rows:
        print(
            f"{r['scenario']:<13}{r['policy']:<12}"
            f"{r['late_jobs']:>7}"
            f"{r['p95_all_s']:>12.1f}"
            f"{r['max_tardiness_s']:>12.1f}"
            f"{r['top10_share_pct']:>12.1f}"
        )
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

"""Benchmark risk-aware scheduling on held-out test jobs."""

import csv
from pathlib import Path

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.metrics import compute_metrics
from hybridsched.predictive import (
    PredictedEASYBackfillSimulator,
)
from hybridsched.risk_aware import (
    RiskAwareEASYSimulator,
    RuntimeQuantiles,
)

from benchmark_ml_scheduling import (
    CAPACITY,
    RESULTS,
    prepare_predictions,
    prepare_test_workload,
)


def load_quantiles():
    path = RESULTS / "runtime_quantiles_test.csv"

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    quantiles = {}

    for row in rows:
        values = sorted([
            max(1, round(float(row["p10_s"]))),
            max(1, round(float(row["p50_s"]))),
            max(1, round(float(row["p90_s"]))),
        ])

        quantiles[row["job_id"]] = RuntimeQuantiles(
            *values
        )

    return quantiles


def main():
    rows, jobs, actual = prepare_test_workload()
    _, xgb = prepare_predictions(rows)
    quantiles = load_quantiles()

    policies = [
        (
            "EASY",
            EASYBackfillSimulator,
            {},
        ),
        (
            "EASY-XGB",
            PredictedEASYBackfillSimulator,
            {"runtime_predictions_s": xgb},
        ),
        (
            "RISK-EASY",
            RiskAwareEASYSimulator,
            {"runtime_quantiles": quantiles},
        ),
    ]

    print("=" * 80)
    print("HybridSched-AI | RISK-AWARE SCHEDULING BENCHMARK")
    print("=" * 80)
    print(f"Test jobs: {len(jobs)}")
    print("-" * 80)
    print(
        f"{'Algorithm':<14}"
        f"{'Makespan':>13}"
        f"{'Mean Wait':>14}"
        f"{'P95 Wait':>13}"
        f"{'GPU Util':>13}"
    )
    print("-" * 80)

    result_rows = []

    for name, cls, extra in policies:
        cluster = Cluster(name, CAPACITY)

        simulator = cls(
            cluster,
            jobs,
            actual_runtimes_s=actual,
            **extra,
        )

        results = simulator.run()

        assert len(results) == len(jobs)
        assert cluster.available == cluster.capacity

        for result in results:
            assert (
                result.finish_s - result.start_s
                == actual[result.job_id]
            )

        metrics = compute_metrics(
            jobs,
            results,
            CAPACITY,
        )

        result_rows.append({
            "algorithm": name,
            "makespan_s": metrics.makespan_s,
            "avg_wait_s": metrics.average_waiting_s,
            "p95_wait_s": metrics.p95_waiting_s,
            "gpu_util_pct": metrics.gpu_utilization_pct,
        })

        print(
            f"{name:<14}"
            f"{metrics.makespan_s:>13}"
            f"{metrics.average_waiting_s:>14.2f}"
            f"{metrics.p95_waiting_s:>13.2f}"
            f"{metrics.gpu_utilization_pct:>12.2f}%"
        )

    output = RESULTS / "benchmark_risk_aware.csv"

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(result_rows[0]),
        )
        writer.writeheader()
        writer.writerows(result_rows)

    print("-" * 80)
    print(f"Results saved: {output}")
    print("Risk-aware benchmark completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()

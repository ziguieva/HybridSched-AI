"""Compare per-job scheduling decisions on a paired workload."""

import csv
from pathlib import Path

from experiments.benchmark_ml_dynamic import load_test_data
from experiments.decision_shift_benchmark import transform_workload
from hybridsched.decision_focused import DecisionFocusedSimulator
from hybridsched.infrastructure import default_infrastructure
from hybridsched.ml_deadline import MLDeadlineSimulator
from hybridsched.ml_predictor import MLRuntimePredictor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "decision_trace_burst.csv"


def main():
    jobs, actual = load_test_data()
    shifted, runtimes = transform_workload(
        jobs, actual, "BURST", 20001
    )

    predictor = MLRuntimePredictor()
    policies = {
        "REQUESTED": MLDeadlineSimulator(
            shifted, default_infrastructure(),
            actual_runtimes=runtimes,
            mode="requested",
        ),
        "P90": MLDeadlineSimulator(
            shifted, default_infrastructure(),
            actual_runtimes=runtimes,
            predictor=predictor,
            mode="p90",
        ),
        "DECISION": DecisionFocusedSimulator(
            shifted, default_infrastructure(),
            runtimes, predictor,
        ),
    }

    job_by_id = {j.job_id: j for j in shifted}
    rows = []

    for policy, sim in policies.items():
        outputs = sim.run()

        if len(outputs) != len(shifted):
            raise RuntimeError("Incomplete simulation")

        for result in outputs:
            job = job_by_id[result.job_id]

            rows.append({
                "policy": policy,
                "job_id": result.job_id,
                "requested_s": job.requested_walltime_s,
                "actual_s": runtimes[job.job_id],
                "gpus": job.resources.gpus,
                "deadline_s": job.deadline_s,
                "cluster": result.cluster,
                "start_s": result.start_s,
                "finish_s": result.finish_s,
                "waiting_s": result.waiting_s,
                "tardiness_s": result.tardiness_s,
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    indexed = {
        (r["policy"], r["job_id"]): r for r in rows
    }

    differences = []

    for job in shifted:
        baseline = indexed[("P90", job.job_id)]
        decision = indexed[("DECISION", job.job_id)]

        differences.append({
            "job_id": job.job_id,
            "extra_tardiness": (
                decision["tardiness_s"]
                - baseline["tardiness_s"]
            ),
            "extra_wait": (
                decision["waiting_s"]
                - baseline["waiting_s"]
            ),
            "p90_cluster": baseline["cluster"],
            "decision_cluster": decision["cluster"],
            "p90_tardiness": baseline["tardiness_s"],
            "decision_tardiness": decision["tardiness_s"],
        })

    differences.sort(
        key=lambda x: x["extra_tardiness"],
        reverse=True,
    )

    print("=== STEP 21.1 | WORST DECISION REGRESSIONS ===")
    print(
        f"{'Job':<12}{'Extra tardiness':>18}"
        f"{'Extra wait':>15}"
        f"{'P90 cluster':>16}"
        f"{'Decision cluster':>20}"
    )

    for row in differences[:12]:
        print(
            f"{row['job_id']:<12}"
            f"{row['extra_tardiness']:>18.1f}"
            f"{row['extra_wait']:>15.1f}"
            f"{row['p90_cluster']:>16}"
            f"{row['decision_cluster']:>20}"
        )

    positive = [
        r for r in differences
        if r["extra_tardiness"] > 0
    ]

    print(f"\nJobs worse than P90: {len(positive)}")
    print(
        "Total positive tardiness regressions:",
        round(sum(r["extra_tardiness"] for r in positive), 1),
    )
    print(
        "Net tardiness difference:",
        round(sum(r["extra_tardiness"] for r in differences), 1),
    )
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

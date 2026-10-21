"""Extract nondominated solutions from HPC experiments."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "multiobjective.csv"
OUTPUT = ROOT / "results" / "pareto_frontier.csv"
FIGURE = ROOT / "results" / "pareto_frontier.png"

OBJECTIVES = ("cost", "makespan", "tardiness")


def dominates(a, b):
    """True when a Pareto-dominates b (minimization)."""
    return (
        all(a[key] <= b[key] for key in OBJECTIVES)
        and any(a[key] < b[key] for key in OBJECTIVES)
    )


def pareto_filter(rows):
    """Return unique nondominated objective vectors."""
    unique = {
        tuple(row[key] for key in OBJECTIVES): row
        for row in rows
    }

    candidates = list(unique.values())

    return sorted(
        [
            row for row in candidates
            if not any(
                dominates(other, row)
                for other in candidates
            )
        ],
        key=lambda row: (
            row["cost"],
            row["makespan"],
            row["tardiness"],
        ),
    )


def main():
    with SOURCE.open(newline="") as file:
        raw = list(csv.DictReader(file))

    rows = []

    for record in raw:
        if record["status"] not in ("OPTIMAL", "FEASIBLE"):
            continue

        row = dict(record)

        for key in ("budget", "cost", "makespan",
                    "late_jobs", "tardiness", "cloud_jobs"):
            row[key] = int(row[key])

        row["avg_wait"] = float(row["avg_wait"])
        rows.append(row)

    if not rows:
        raise ValueError("No feasible solutions available")

    frontier = pareto_filter(rows)

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(frontier[0]),
        )
        writer.writeheader()
        writer.writerows(frontier)

    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        [r["cost"] for r in rows],
        [r["makespan"] for r in rows],
        c=[r["tardiness"] for r in rows],
        cmap="viridis",
        s=75,
        alpha=0.65,
        label="Solutions evaluated",
    )

    ax.scatter(
        [r["cost"] for r in frontier],
        [r["makespan"] for r in frontier],
        facecolors="none",
        edgecolors="black",
        s=180,
        linewidths=1.5,
        label="Nondominated solutions",
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Total tardiness (s)",
    )

    ax.set_xlabel("Synthetic cost units")
    ax.set_ylabel("Makespan (s)")
    ax.set_title(
        "HybridSched-AI — Sampled Pareto Frontier"
    )
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    print("=" * 75)
    print("HybridSched-AI | PARETO FRONTIER")
    print("=" * 75)
    print(f"Feasible solutions: {len(rows)}")
    print(f"Nondominated solutions: {len(frontier)}")

    print("-" * 75)
    print(
        f"{'Cost':>12}"
        f"{'Makespan':>14}"
        f"{'Tardiness':>15}"
        f"{'Cloud Jobs':>14}"
    )

    for row in frontier:
        print(
            f"{row['cost']:>12}"
            f"{row['makespan']:>14}"
            f"{row['tardiness']:>15}"
            f"{row['cloud_jobs']:>14}"
        )

    print("-" * 75)
    print(f"CSV: {OUTPUT}")
    print(f"Figure: {FIGURE}")
    print("Pareto analysis completed.")


if __name__ == "__main__":
    main()

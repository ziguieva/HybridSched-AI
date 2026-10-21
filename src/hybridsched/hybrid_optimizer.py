"""Offline hybrid cloud/HPC placement and scheduling optimization."""

from dataclasses import dataclass
from math import isfinite

from ortools.sat.python import cp_model

from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job


@dataclass(frozen=True)
class HybridJobResult:
    job_id: str
    cluster: str
    arrival_s: int
    start_s: int
    finish_s: int
    waiting_s: int
    execution_s: int
    cost_units: int


@dataclass(frozen=True)
class HybridOptimizationResult:
    status: str
    objective: float
    makespan_s: int
    total_cost_units: int
    results: list[HybridJobResult]


def optimize_hybrid(
    jobs: list[Job],
    clusters: list[ClusterSpec],
    time_limit_s: float = 15.0,
    wait_weight: int = 2,
    makespan_weight: int = 1,
    cost_weight: int = 1,
) -> HybridOptimizationResult:
    """Joint offline job placement and resource-constrained scheduling."""

    if not clusters:
        raise ValueError("At least one cluster is required")

    names = [cluster.name for cluster in clusters]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate cluster names")

    ids = [job.job_id for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate job IDs")

    if not isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("Time limit must be positive")

    weights = (wait_weight, makespan_weight, cost_weight)
    if any(type(w) is not int or w < 0 for w in weights):
        raise ValueError("Objective weights must be nonnegative")

    if not any(weights):
        raise ValueError("At least one objective weight is required")

    if not jobs:
        return HybridOptimizationResult(
            "OPTIMAL", 0.0, 0, 0, []
        )

    eligible = {}

    for job in jobs:
        options = [
            cluster
            for cluster in clusters
            if job.resources.fits_in(cluster.capacity)
        ]

        if not options:
            raise ValueError(
                f"No eligible cluster for {job.job_id}"
            )

        eligible[job.job_id] = options

    horizon = (
        max(job.arrival_s for job in jobs)
        + sum(
            max(
                cluster.execution_time(job.requested_walltime_s)
                for cluster in eligible[job.job_id]
            )
            for job in jobs
        )
    )

    model = cp_model.CpModel()

    assignments = {}
    completions = {}
    starts = {}
    durations = {}
    costs = {}

    intervals_by_cluster = {
        cluster.name: [] for cluster in clusters
    }

    demands_by_cluster = {
        cluster.name: {
            "cpu": [],
            "gpu": [],
            "memory": [],
        }
        for cluster in clusters
    }

    for job in jobs:
        job_id = job.job_id

        start = model.new_int_var(
            job.arrival_s, horizon, f"start_{job_id}"
        )

        completion = model.new_int_var(
            job.arrival_s, horizon, f"completion_{job_id}"
        )

        starts[job_id] = start
        completions[job_id] = completion

        alternatives = []

        for cluster in eligible[job_id]:
            key = (job_id, cluster.name)

            selected = model.new_bool_var(
                f"place_{job_id}_{cluster.name}"
            )

            duration = cluster.execution_time(
                job.requested_walltime_s
            )

            end = model.new_int_var(
                job.arrival_s,
                horizon,
                f"end_{job_id}_{cluster.name}",
            )

            interval = model.new_optional_interval_var(
                start,
                duration,
                end,
                selected,
                f"interval_{job_id}_{cluster.name}",
            )

            model.add(
                completion == end
            ).only_enforce_if(selected)

            assignments[key] = selected
            durations[key] = duration

            costs[key] = cluster.execution_cost(
                job.resources,
                duration,
            )

            alternatives.append(selected)

            intervals_by_cluster[cluster.name].append(
                interval
            )

            demand = demands_by_cluster[cluster.name]

            demand["cpu"].append(
                job.resources.cpu_cores
            )
            demand["gpu"].append(
                job.resources.gpus
            )
            demand["memory"].append(
                job.resources.memory_gib
            )

        model.add_exactly_one(alternatives)

    for cluster in clusters:
        name = cluster.name
        intervals = intervals_by_cluster[name]
        demands = demands_by_cluster[name]

        for resource_name, capacity in (
            ("cpu", cluster.capacity.cpu_cores),
            ("gpu", cluster.capacity.gpus),
            ("memory", cluster.capacity.memory_gib),
        ):
            model.add_cumulative(
                intervals,
                demands[resource_name],
                capacity,
            )

    makespan = model.new_int_var(
        0, horizon, "makespan"
    )

    model.add_max_equality(
        makespan,
        list(completions.values()),
    )

    total_waiting = sum(
        starts[job.job_id] - job.arrival_s
        for job in jobs
    )

    total_cost = sum(
        costs[key] * selected
        for key, selected in assignments.items()
    )

    model.minimize(
        wait_weight * total_waiting
        + makespan_weight * makespan
        + cost_weight * total_cost
    )

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 4
    solver.parameters.random_seed = 42

    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return HybridOptimizationResult(
            status_name,
            float("nan"),
            0,
            0,
            [],
        )

    results = []

    for job in jobs:
        for cluster in eligible[job.job_id]:
            key = (job.job_id, cluster.name)

            if solver.value(assignments[key]):
                start_s = solver.value(
                    starts[job.job_id]
                )

                finish_s = solver.value(
                    completions[job.job_id]
                )

                results.append(
                    HybridJobResult(
                        job_id=job.job_id,
                        cluster=cluster.name,
                        arrival_s=job.arrival_s,
                        start_s=start_s,
                        finish_s=finish_s,
                        waiting_s=start_s - job.arrival_s,
                        execution_s=durations[key],
                        cost_units=costs[key],
                    )
                )

                break

    results.sort(key=lambda result: result.job_id)

    return HybridOptimizationResult(
        status=status_name,
        objective=solver.objective_value,
        makespan_s=max(r.finish_s for r in results),
        total_cost_units=sum(r.cost_units for r in results),
        results=results,
    )

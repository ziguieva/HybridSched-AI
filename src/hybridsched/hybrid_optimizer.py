"""Hybrid cloud/HPC scheduling with deadlines and budgets."""

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
    tardiness_s: int = 0


@dataclass(frozen=True)
class HybridOptimizationResult:
    status: str
    objective: float
    makespan_s: int
    total_cost_units: int
    results: list[HybridJobResult]
    total_tardiness_s: int = 0
    deadline_violations: int = 0
    cloud_cost_units: int = 0


def optimize_hybrid(
    jobs: list[Job],
    clusters: list[ClusterSpec],
    time_limit_s: float = 15.0,
    wait_weight: int = 2,
    makespan_weight: int = 1,
    cost_weight: int = 1,
    deadline_weight: int = 0,
    max_budget_units: int | None = None,
    max_cloud_budget_units: int | None = None,
    hard_deadlines: bool = False,
) -> HybridOptimizationResult:
    """Optimize offline placement and scheduling.

    All jobs and requested durations are known in advance.
    Costs are synthetic units.
    """

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

    weights = (
        wait_weight,
        makespan_weight,
        cost_weight,
        deadline_weight,
    )

    if any(type(w) is not int or w < 0 for w in weights):
        raise ValueError("Weights must be nonnegative integers")

    if not any(weights):
        raise ValueError("At least one weight must be positive")

    for budget in (
        max_budget_units,
        max_cloud_budget_units,
    ):
        if budget is not None and (
            type(budget) is not int or budget < 0
        ):
            raise ValueError(
                "Budgets must be nonnegative integers"
            )

    if not jobs:
        return HybridOptimizationResult(
            status="OPTIMAL",
            objective=0.0,
            makespan_s=0,
            total_cost_units=0,
            results=[],
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
                cluster.execution_time(
                    job.requested_walltime_s
                )
                for cluster in eligible[job.job_id]
            )
            for job in jobs
        )
    )

    model = cp_model.CpModel()

    assignments = {}
    starts = {}
    completions = {}
    durations = {}
    costs = {}
    tardiness_vars = {}

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
            job.arrival_s,
            horizon,
            f"start_{job_id}",
        )

        completion = model.new_int_var(
            job.arrival_s,
            horizon,
            f"completion_{job_id}",
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

        if job.deadline_s is not None:
            if hard_deadlines:
                model.add(
                    completion <= job.deadline_s
                )

            tardiness = model.new_int_var(
                0,
                horizon,
                f"tardiness_{job_id}",
            )

            model.add_max_equality(
                tardiness,
                [
                    completion - job.deadline_s,
                    0,
                ],
            )

            tardiness_vars[job_id] = tardiness

    for cluster in clusters:
        name = cluster.name

        for resource_name, capacity in (
            ("cpu", cluster.capacity.cpu_cores),
            ("gpu", cluster.capacity.gpus),
            ("memory", cluster.capacity.memory_gib),
        ):
            model.add_cumulative(
                intervals_by_cluster[name],
                demands_by_cluster[name][resource_name],
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

    cloud_cost = sum(
        costs[key] * selected
        for key, selected in assignments.items()
        if next(
            cluster.provider
            for cluster in clusters
            if cluster.name == key[1]
        ) == "GCP_SIMULATED"
    )

    total_tardiness = sum(
        tardiness_vars.values()
    )

    if max_budget_units is not None:
        model.add(total_cost <= max_budget_units)

    if max_cloud_budget_units is not None:
        model.add(
            cloud_cost <= max_cloud_budget_units
        )

    model.minimize(
        wait_weight * total_waiting
        + makespan_weight * makespan
        + cost_weight * total_cost
        + deadline_weight * total_tardiness
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 4
    solver.parameters.random_seed = 42

    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        return HybridOptimizationResult(
            status=status_name,
            objective=float("nan"),
            makespan_s=0,
            total_cost_units=0,
            results=[],
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

                tardiness_s = (
                    max(0, finish_s - job.deadline_s)
                    if job.deadline_s is not None
                    else 0
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
                        tardiness_s=tardiness_s,
                    )
                )

                break

    results.sort(key=lambda r: r.job_id)

    cloud_names = {
        cluster.name
        for cluster in clusters
        if cluster.provider == "GCP_SIMULATED"
    }

    return HybridOptimizationResult(
        status=status_name,
        objective=solver.objective_value,
        makespan_s=max(r.finish_s for r in results),
        total_cost_units=sum(r.cost_units for r in results),
        results=results,
        total_tardiness_s=sum(
            r.tardiness_s for r in results
        ),
        deadline_violations=sum(
            r.tardiness_s > 0 for r in results
        ),
        cloud_cost_units=sum(
            r.cost_units
            for r in results
            if r.cluster in cloud_names
        ),
    )

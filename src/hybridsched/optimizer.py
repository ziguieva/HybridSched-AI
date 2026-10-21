"""Offline HPC scheduling optimization with OR-Tools CP-SAT."""

from dataclasses import dataclass

from ortools.sat.python import cp_model

from hybridsched.job import Job
from hybridsched.resources import Resources
from hybridsched.simulation import JobResult


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    objective: float
    makespan_s: int
    results: list[JobResult]


def optimize_schedule(
    jobs: list[Job],
    capacity: Resources,
    time_limit_s: float = 10.0,
    makespan_weight: int = 1,
) -> OptimizationResult:
    """Minimize total waiting plus weighted makespan.

    All jobs and their requested durations are known
    before optimization. This is an offline planning model.
    """

    if not jobs:
        return OptimizationResult(
            status="OPTIMAL",
            objective=0.0,
            makespan_s=0,
            results=[],
        )

    if time_limit_s <= 0:
        raise ValueError("Time limit must be positive")

    if makespan_weight < 0:
        raise ValueError("Makespan weight must be nonnegative")

    ids = [job.job_id for job in jobs]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate job IDs")

    if any(
        not job.resources.fits_in(capacity)
        for job in jobs
    ):
        raise ValueError("Job exceeds cluster capacity")

    model = cp_model.CpModel()

    horizon = (
        max(job.arrival_s for job in jobs)
        + sum(job.requested_walltime_s for job in jobs)
    )

    starts = {}
    ends = {}
    intervals = {}

    for job in jobs:
        start = model.new_int_var(
            job.arrival_s,
            horizon,
            f"start_{job.job_id}",
        )

        end = model.new_int_var(
            job.arrival_s,
            horizon,
            f"end_{job.job_id}",
        )

        interval = model.new_interval_var(
            start,
            job.requested_walltime_s,
            end,
            f"interval_{job.job_id}",
        )

        starts[job.job_id] = start
        ends[job.job_id] = end
        intervals[job.job_id] = interval

    ordered_intervals = [
        intervals[job.job_id]
        for job in jobs
    ]

    resource_constraints = [
        (
            "CPU",
            [job.resources.cpu_cores for job in jobs],
            capacity.cpu_cores,
        ),
        (
            "GPU",
            [job.resources.gpus for job in jobs],
            capacity.gpus,
        ),
        (
            "RAM",
            [job.resources.memory_gib for job in jobs],
            capacity.memory_gib,
        ),
    ]

    for name, demands, limit in resource_constraints:
        model.add_cumulative(
            ordered_intervals,
            demands,
            limit,
        )

    makespan = model.new_int_var(
        0,
        horizon,
        "makespan",
    )

    model.add_max_equality(
        makespan,
        list(ends.values()),
    )

    total_waiting = sum(
        starts[job.job_id] - job.arrival_s
        for job in jobs
    )

    model.minimize(
        total_waiting + makespan_weight * makespan
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
        return OptimizationResult(
            status=status_name,
            objective=float("nan"),
            makespan_s=0,
            results=[],
        )

    results = []

    for job in jobs:
        start_s = solver.value(starts[job.job_id])
        finish_s = solver.value(ends[job.job_id])

        results.append(
            JobResult(
                job_id=job.job_id,
                arrival_s=job.arrival_s,
                start_s=start_s,
                finish_s=finish_s,
                waiting_s=start_s - job.arrival_s,
            )
        )

    results.sort(key=lambda r: r.job_id)

    return OptimizationResult(
        status=status_name,
        objective=solver.objective_value,
        makespan_s=solver.value(makespan),
        results=results,
    )

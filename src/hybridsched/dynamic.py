"""Online event-driven scheduling for hybrid HPC clusters."""

from dataclasses import dataclass
import heapq

from ortools.sat.python import cp_model

from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.resources import Resources


@dataclass(frozen=True)
class DynamicResult:
    job_id: str
    cluster: str
    arrival_s: int
    start_s: int
    finish_s: int
    waiting_s: int
    cost_units: int
    tardiness_s: int


class DynamicHybridSimulator:
    """Online scheduling with event-driven replanning."""

    def __init__(
        self,
        jobs: list[Job],
        clusters: list[ClusterSpec],
        policy: str = "rolling",
        time_limit_s: float = 2.0,
        cost_weight: int = 1,
        deadline_weight: int = 10,
    ):
        if policy not in ("fcfs", "sjf", "rolling"):
            raise ValueError("Unknown scheduling policy")

        if not clusters:
            raise ValueError("No clusters provided")

        if len({j.job_id for j in jobs}) != len(jobs):
            raise ValueError("Duplicate job IDs")

        if len({c.name for c in clusters}) != len(clusters):
            raise ValueError("Duplicate cluster names")

        if time_limit_s <= 0:
            raise ValueError("Time limit must be positive")

        if any(
            type(w) is not int or w < 0
            for w in (cost_weight, deadline_weight)
        ):
            raise ValueError("Weights must be nonnegative integers")

        for job in jobs:
            if not any(
                job.resources.fits_in(c.capacity)
                for c in clusters
            ):
                raise ValueError(
                    f"No eligible cluster for {job.job_id}"
                )

        self.jobs = sorted(
            jobs,
            key=lambda j: (j.arrival_s, j.job_id),
        )
        self.clusters = list(clusters)
        self.policy = policy
        self.time_limit_s = time_limit_s
        self.cost_weight = cost_weight
        self.deadline_weight = deadline_weight

        self.used = {
            c.name: Resources()
            for c in clusters
        }

        self.waiting = []
        self.running = []
        self.results = []
        self._run_completed = False

    def _available(self, cluster):
        return cluster.capacity - self.used[cluster.name]

    def _eligible_now(self, job):
        return [
            c for c in self.clusters
            if job.resources.fits_in(self._available(c))
        ]

    def _start(self, job, cluster, now):
        if not job.resources.fits_in(
            self._available(cluster)
        ):
            raise RuntimeError("Resource overcommit")

        self.used[cluster.name] = (
            self.used[cluster.name] + job.resources
        )

        duration = cluster.execution_time(
            job.requested_walltime_s
        )

        finish = now + duration
        tardiness = (
            max(0, finish - job.deadline_s)
            if job.deadline_s is not None
            else 0
        )

        self.results.append(
            DynamicResult(
                job_id=job.job_id,
                cluster=cluster.name,
                arrival_s=job.arrival_s,
                start_s=now,
                finish_s=finish,
                waiting_s=now - job.arrival_s,
                cost_units=cluster.execution_cost(
                    job.resources,
                    duration,
                ),
                tardiness_s=tardiness,
            )
        )

        heapq.heappush(
            self.running,
            (finish, job.job_id, cluster.name, job),
        )

    def _greedy_dispatch(self, now):
        while self.waiting:
            if self.policy == "sjf":
                ordered = sorted(
                    self.waiting,
                    key=lambda j: (
                        j.requested_walltime_s,
                        j.arrival_s,
                        j.job_id,
                    ),
                )
            else:
                ordered = sorted(
                    self.waiting,
                    key=lambda j: (
                        j.arrival_s,
                        j.job_id,
                    ),
                )

            selected = None

            for job in ordered:
                options = self._eligible_now(job)

                if options:
                    # Prefer the earliest predicted finish,
                    # then lower synthetic execution cost.
                    cluster = min(
                        options,
                        key=lambda c: (
                            c.execution_time(
                                job.requested_walltime_s
                            ),
                            c.execution_cost(
                                job.resources,
                                c.execution_time(
                                    job.requested_walltime_s
                                ),
                            ),
                            c.name,
                        ),
                    )

                    selected = (job, cluster)
                    break

                if self.policy == "fcfs":
                    break

            if selected is None:
                break

            job, cluster = selected
            self.waiting.remove(job)
            self._start(job, cluster, now)

    def _rolling_plan(self, now):
        """Optimize only currently waiting jobs.

        Already running jobs are not moved. Their
        resource use is conservatively held constant
        during the planning horizon.
        """

        if not self.waiting:
            return []

        model = cp_model.CpModel()
        jobs = self.waiting

        horizon = now + sum(
            max(
                c.execution_time(j.requested_walltime_s)
                for c in self.clusters
                if j.resources.fits_in(c.capacity)
            )
            for j in jobs
        )

        starts = {}
        ends = {}
        choices = {}
        intervals = {c.name: [] for c in self.clusters}
        cpu = {c.name: [] for c in self.clusters}
        gpu = {c.name: [] for c in self.clusters}
        memory = {c.name: [] for c in self.clusters}
        costs = {}
        tardiness = []

        for job in jobs:
            jid = job.job_id
            starts[jid] = model.new_int_var(
                now, horizon, f"start_{jid}"
            )
            ends[jid] = model.new_int_var(
                now, horizon, f"end_{jid}"
            )

            alternatives = []

            for cluster in self.clusters:
                if not job.resources.fits_in(
                    self._available(cluster)
                ):
                    continue

                key = (jid, cluster.name)
                selected = model.new_bool_var(
                    f"assign_{jid}_{cluster.name}"
                )

                duration = cluster.execution_time(
                    job.requested_walltime_s
                )

                interval = model.new_optional_interval_var(
                    starts[jid],
                    duration,
                    ends[jid],
                    selected,
                    f"interval_{jid}_{cluster.name}",
                )

                model.add(
                    ends[jid] <= horizon
                ).only_enforce_if(selected)

                choices[key] = selected
                alternatives.append(selected)
                intervals[cluster.name].append(interval)

                cpu[cluster.name].append(
                    job.resources.cpu_cores
                )
                gpu[cluster.name].append(
                    job.resources.gpus
                )
                memory[cluster.name].append(
                    job.resources.memory_gib
                )

                costs[key] = cluster.execution_cost(
                    job.resources, duration
                )

            if not alternatives:
                return []

            model.add_exactly_one(alternatives)

            if job.deadline_s is not None:
                late = model.new_int_var(
                    0, horizon,
                    f"late_{jid}",
                )
                model.add_max_equality(
                    late,
                    [ends[jid] - job.deadline_s, 0],
                )
                tardiness.append(late)

        for cluster in self.clusters:
            name = cluster.name
            available = self._available(cluster)

            for demands, capacity in (
                (cpu[name], available.cpu_cores),
                (gpu[name], available.gpus),
                (memory[name], available.memory_gib),
            ):
                model.add_cumulative(
                    intervals[name],
                    demands,
                    capacity,
                )

        makespan = model.new_int_var(
            now, horizon, "makespan"
        )
        model.add_max_equality(
            makespan, list(ends.values())
        )

        total_wait = sum(
            starts[j.job_id] - j.arrival_s
            for j in jobs
        )

        total_cost = sum(
            value * choices[key]
            for key, value in costs.items()
        )

        model.minimize(
            2 * total_wait
            + 5 * (makespan - now)
            + self.cost_weight * total_cost
            + self.deadline_weight * sum(tardiness)
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = (
            self.time_limit_s
        )
        solver.parameters.num_workers = 4
        solver.parameters.random_seed = 42

        status = solver.solve(model)

        if status not in (
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,
        ):
            return []

        plan = []

        for job in jobs:
            for cluster in self.clusters:
                key = (job.job_id, cluster.name)

                if key in choices and solver.value(
                    choices[key]
                ):
                    plan.append((
                        solver.value(starts[job.job_id]),
                        job,
                        cluster,
                    ))
                    break

        return sorted(
            plan,
            key=lambda item: (
                item[0], item[1].job_id
            ),
        )

    def _rolling_dispatch(self, now):
        """Commit only starts scheduled for the current event."""

        while self.waiting:
            plan = self._rolling_plan(now)

            immediate = [
                (job, cluster)
                for start, job, cluster in plan
                if start == now
            ]

            if not immediate:
                # Safe fallback if the optimization cannot
                # provide an immediate feasible decision.
                self._greedy_dispatch(now)
                break

            for job, cluster in immediate:
                if job in self.waiting and job.resources.fits_in(
                    self._available(cluster)
                ):
                    self.waiting.remove(job)
                    self._start(job, cluster, now)

    def run(self):
        if self._run_completed:
            raise RuntimeError(
                "Simulation can only run once"
            )

        self._run_completed = True
        index = 0
        total = len(self.jobs)

        while index < total or self.running or self.waiting:
            arrival_time = (
                self.jobs[index].arrival_s
                if index < total
                else float("inf")
            )

            finish_time = (
                self.running[0][0]
                if self.running
                else float("inf")
            )

            now = min(arrival_time, finish_time)

            if now == float("inf"):
                raise RuntimeError(
                    "Unscheduled jobs remain with no future event"
                )

            # Release completed tasks first.
            while self.running and self.running[0][0] == now:
                _, _, cluster_name, job = heapq.heappop(
                    self.running
                )

                self.used[cluster_name] = (
                    self.used[cluster_name] - job.resources
                )

            # Only jobs already arrived are visible.
            while (
                index < total
                and self.jobs[index].arrival_s == now
            ):
                self.waiting.append(self.jobs[index])
                index += 1

            if self.policy == "rolling":
                self._rolling_dispatch(now)
            else:
                self._greedy_dispatch(now)

            if self.waiting and not self.running:
                if index == total:
                    raise RuntimeError(
                        "Scheduling deadlock"
                    )

        if len(self.results) != total:
            raise RuntimeError(
                "Incomplete simulation"
            )

        if any(
            self.used[c.name] != Resources()
            for c in self.clusters
        ):
            raise RuntimeError(
                "Resources not fully released"
            )

        return sorted(
            self.results,
            key=lambda r: r.job_id,
        )

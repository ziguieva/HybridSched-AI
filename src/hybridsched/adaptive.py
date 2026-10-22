"""Adaptive online scheduling for hybrid HPC clusters."""

from math import isfinite

from hybridsched.dynamic import DynamicHybridSimulator


class AdaptiveHybridSimulator(DynamicHybridSimulator):
    """Use a fast deadline-aware rule most of the time.

    CP-SAT is activated selectively under contention.
    Running jobs remain non-preemptive.
    """

    def __init__(
        self,
        jobs,
        clusters,
        time_limit_s=0.2,
        cost_weight=1,
        deadline_weight=10,
        cp_threshold=8,
        max_cp_calls=4,
    ):
        if (
            type(cp_threshold) is not int
            or cp_threshold < 1
        ):
            raise ValueError(
                "cp_threshold must be a positive integer"
            )

        if (
            type(max_cp_calls) is not int
            or max_cp_calls < 0
        ):
            raise ValueError(
                "max_cp_calls must be nonnegative"
            )

        if not isfinite(time_limit_s) or time_limit_s <= 0:
            raise ValueError("Invalid solver time limit")

        super().__init__(
            jobs=jobs,
            clusters=clusters,
            policy="rolling",
            time_limit_s=time_limit_s,
            cost_weight=cost_weight,
            deadline_weight=deadline_weight,
        )

        self.cp_threshold = cp_threshold
        self.max_cp_calls = max_cp_calls

        self.heuristic_decisions = 0
        self.cp_decisions = 0
        self.adaptive_events = 0

    def _deadline_key(self, job, now):
        """Rank by estimated urgency, then short duration."""
        minimum_duration = min(
            cluster.execution_time(
                job.requested_walltime_s
            )
            for cluster in self.clusters
            if job.resources.fits_in(cluster.capacity)
        )

        if job.deadline_s is None:
            return (
                1,
                float("inf"),
                job.requested_walltime_s,
                job.arrival_s,
                job.job_id,
            )

        slack = (
            job.deadline_s
            - now
            - minimum_duration
        )

        return (
            0,
            slack,
            job.requested_walltime_s,
            job.arrival_s,
            job.job_id,
        )

    def _heuristic_dispatch(self, now):
        """Dispatch available jobs without moving running jobs."""
        started = 0

        while self.waiting:
            eligible_jobs = [
                job for job in self.waiting
                if self._eligible_now(job)
            ]

            if not eligible_jobs:
                break

            job = min(
                eligible_jobs,
                key=lambda j: self._deadline_key(j, now),
            )

            options = self._eligible_now(job)

            # Minimize projected lateness first, then
            # simulated execution cost and duration.
            def cluster_key(cluster):
                duration = cluster.execution_time(
                    job.requested_walltime_s
                )

                lateness = (
                    max(
                        0,
                        now + duration - job.deadline_s,
                    )
                    if job.deadline_s is not None
                    else 0
                )

                cost = cluster.execution_cost(
                    job.resources,
                    duration,
                )

                return (
                    lateness,
                    cost,
                    duration,
                    cluster.name,
                )

            chosen = min(options, key=cluster_key)

            self.waiting.remove(job)
            self._start(job, chosen, now)
            started += 1

        self.heuristic_decisions += started
        return started

    def _rolling_dispatch(self, now):
        """Choose between CP-SAT and the fast heuristic."""
        if not self.waiting:
            return None

        self.adaptive_events += 1

        eligible_now = [
            job for job in self.waiting
            if self._eligible_now(job)
        ]

        if not eligible_now:
            self.skipped_plans += 1
            return None

        use_cp = (
            len(self.waiting) >= self.cp_threshold
            and len(eligible_now) >= 2
            and self.solver_calls < self.max_cp_calls
        )

        if use_cp:
            self.cp_decisions += 1
            return super()._rolling_dispatch(now)

        self._heuristic_dispatch(now)
        return None

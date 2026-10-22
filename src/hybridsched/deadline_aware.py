"""Deadline-aware policies for online hybrid HPC scheduling."""

from hybridsched.adaptive import AdaptiveHybridSimulator


class DeadlineAwareSimulator(AdaptiveHybridSimulator):
    """EDF, LST and resource-aware online heuristics.

    All decisions use information available at the
    current event. Running jobs are never preempted.
    """

    POLICIES = ("edf", "lst", "resource")

    def __init__(
        self,
        jobs,
        clusters,
        priority_policy="lst",
        **kwargs,
    ):
        if priority_policy not in self.POLICIES:
            raise ValueError("Unknown deadline policy")

        # These three policies are pure heuristics.
        # They do not invoke CP-SAT.
        kwargs.pop("cp_threshold", None)
        kwargs.pop("max_cp_calls", None)

        super().__init__(
            jobs=jobs,
            clusters=clusters,
            cp_threshold=1,
            max_cp_calls=0,
            **kwargs,
        )

        self.priority_policy = priority_policy

    def _minimum_duration(self, job):
        return min(
            cluster.execution_time(
                job.requested_walltime_s
            )
            for cluster in self.clusters
            if job.resources.fits_in(cluster.capacity)
        )

    def _priority(self, job, now):
        duration = self._minimum_duration(job)

        if job.deadline_s is None:
            return (
                1,
                float("inf"),
                duration,
                job.arrival_s,
                job.job_id,
            )

        if self.priority_policy == "edf":
            urgency = job.deadline_s
        elif self.priority_policy == "lst":
            urgency = (
                job.deadline_s - now - duration
            )
        else:
            urgency = self._resource_slack(job, now)

        return (
            0,
            urgency,
            duration,
            job.arrival_s,
            job.job_id,
        )

    def _earliest_cluster_start(self, job, cluster, now):
        """Find earliest capacity availability for one job.

        Consider the completion of running jobs only.
        Future waiting-job placements are unknown.
        """

        if not job.resources.fits_in(cluster.capacity):
            return None

        used = self.used[cluster.name]

        events = sorted(
            (
                (
                    finish,
                    running_job.resources,
                )
                for finish, _, cluster_name, running_job
                in self.running
                if cluster_name == cluster.name
                and finish > now
            ),
            key=lambda event: event[0],
        )

        candidate_times = [now] + sorted({
            finish for finish, _ in events
        })

        for candidate in candidate_times:
            released = [
                resources
                for finish, resources in events
                if finish <= candidate
            ]

            released_cpu = sum(
                r.cpu_cores for r in released
            )
            released_gpu = sum(
                r.gpus for r in released
            )
            released_ram = sum(
                r.memory_gib for r in released
            )

            available_cpu = (
                cluster.capacity.cpu_cores
                - used.cpu_cores
                + released_cpu
            )
            available_gpu = (
                cluster.capacity.gpus
                - used.gpus
                + released_gpu
            )
            available_ram = (
                cluster.capacity.memory_gib
                - used.memory_gib
                + released_ram
            )

            if (
                available_cpu >= job.resources.cpu_cores
                and available_gpu >= job.resources.gpus
                and available_ram >= job.resources.memory_gib
            ):
                return candidate

        return None

    def _projected_finish(self, job, cluster, now):
        start = self._earliest_cluster_start(
            job, cluster, now
        )

        if start is None:
            return float("inf")

        return start + cluster.execution_time(
            job.requested_walltime_s
        )

    def _resource_slack(self, job, now):
        """Deadline margin using running-job releases."""

        projected_finish = min(
            self._projected_finish(job, cluster, now)
            for cluster in self.clusters
            if job.resources.fits_in(cluster.capacity)
        )

        return job.deadline_s - projected_finish

    def _heuristic_dispatch(self, now):
        started = 0

        while self.waiting:
            available_jobs = [
                job for job in self.waiting
                if self._eligible_now(job)
            ]

            if not available_jobs:
                break

            job = min(
                available_jobs,
                key=lambda j: self._priority(j, now),
            )

            options = self._eligible_now(job)

            def cluster_score(cluster):
                finish = now + cluster.execution_time(
                    job.requested_walltime_s
                )

                late = (
                    max(0, finish - job.deadline_s)
                    if job.deadline_s is not None
                    else 0
                )

                duration = cluster.execution_time(
                    job.requested_walltime_s
                )

                cost = cluster.execution_cost(
                    job.resources, duration
                )

                return (
                    late,
                    cost,
                    finish,
                    cluster.name,
                )

            cluster = min(
                options, key=cluster_score
            )

            self.waiting.remove(job)
            self._start(job, cluster, now)
            started += 1

        self.heuristic_decisions += started
        return started

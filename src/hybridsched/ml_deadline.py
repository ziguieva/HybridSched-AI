"""Deadline-aware ML scheduler with actual-runtime execution."""

from hybridsched.ml_dynamic import MLDynamicSimulator


class MLDeadlineSimulator(MLDynamicSimulator):
    """Online ML urgency scheduling.

    Predictions are used for priorities and placement.
    Execution always uses actual runtimes.
    """

    def __init__(
        self,
        jobs,
        clusters,
        actual_runtimes,
        predictor=None,
        mode="requested",
        duration_weight=1.0,
        lateness_weight=0.25,
    ):
        if duration_weight < 0 or lateness_weight < 0:
            raise ValueError("Invalid priority weights")

        super().__init__(
            jobs=jobs,
            clusters=clusters,
            actual_runtimes=actual_runtimes,
            predictor=predictor,
            mode=mode,
        )

        self.duration_weight = duration_weight
        self.lateness_weight = lateness_weight

    def _greedy_dispatch(self, now):
        while self.waiting:
            available = [
                j for j in self.waiting
                if self._eligible_now(j)
            ]

            if not available:
                break

            def priority(job):
                estimate = self._estimate(job)

                durations = [
                    self._cluster_duration(c, estimate)
                    for c in self._eligible_now(job)
                ]

                fastest = min(durations)

                if job.deadline_s is None:
                    return (
                        1,
                        fastest,
                        job.arrival_s,
                        job.job_id,
                    )

                slack = (
                    job.deadline_s - now - fastest
                )

                score = (
                    self.duration_weight * fastest
                    + self.lateness_weight
                    * max(0, -slack)
                )

                return (
                    0,
                    score,
                    job.arrival_s,
                    job.job_id,
                )

            chosen_job = min(
                available,
                key=priority,
            )

            estimate = self._estimate(chosen_job)

            def cluster_score(cluster):
                duration = self._cluster_duration(
                    cluster,
                    estimate,
                )

                predicted_finish = now + duration

                late = (
                    max(
                        0,
                        predicted_finish
                        - chosen_job.deadline_s,
                    )
                    if chosen_job.deadline_s is not None
                    else 0
                )

                cost = cluster.execution_cost(
                    chosen_job.resources,
                    duration,
                )

                return (
                    late,
                    cost,
                    duration,
                    cluster.name,
                )

            cluster = min(
                self._eligible_now(chosen_job),
                key=cluster_score,
            )

            self.waiting.remove(chosen_job)
            self._start(chosen_job, cluster, now)

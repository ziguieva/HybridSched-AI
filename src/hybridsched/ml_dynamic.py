"""Online scheduling with predictions and actual runtimes."""

import heapq
from math import ceil, isfinite

from hybridsched.dynamic import (
    DynamicHybridSimulator,
    DynamicResult,
)


class MLDynamicSimulator(DynamicHybridSimulator):
    """Non-preemptive online scheduler with actual durations.

    Runtime predictions influence decisions only.
    Actual runtimes influence execution and resource release.
    """

    def __init__(
        self,
        jobs,
        clusters,
        actual_runtimes,
        predictor=None,
        mode="requested",
    ):
        if mode not in ("requested", "point", "p50", "p90"):
            raise ValueError("Unknown runtime prediction mode")

        super().__init__(
            jobs,
            clusters,
            policy="sjf",
        )

        self.mode = mode
        self.predictor = predictor
        self.actual_runtimes = dict(actual_runtimes)
        self.predictions = {}

        if mode != "requested" and predictor is None:
            raise ValueError("Predictor required for ML mode")

        for job in self.jobs:
            duration = self.actual_runtimes.get(job.job_id)

            if (
                duration is None
                or not isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(
                    f"Missing/invalid actual runtime: {job.job_id}"
                )

    def _estimate(self, job):
        if self.mode == "requested":
            return float(job.requested_walltime_s)

        prediction = self.predictions.get(job.job_id)

        if prediction is None:
            prediction = self.predictor.predict_many(
                [job]
            )[job.job_id]
            self.predictions[job.job_id] = prediction

        if self.mode == "point":
            return prediction.point_s
        if self.mode == "p50":
            return prediction.p50_s
        return prediction.p90_s

    @staticmethod
    def _cluster_duration(cluster, estimated_s):
        """Apply the same cluster speed/startup transform."""
        return (
            ceil(
                estimated_s
                * cluster.speed_den
                / cluster.speed_num
            )
            + cluster.startup_s
        )

    def _start(self, job, cluster, now):
        """Start with actual runtime, never with prediction."""
        if not job.resources.fits_in(
            self._available(cluster)
        ):
            raise RuntimeError("Resource overcommit")

        self.used[cluster.name] = (
            self.used[cluster.name] + job.resources
        )

        actual_s = ceil(self.actual_runtimes[job.job_id])

        duration = cluster.execution_time(actual_s)
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
        """Online shortest-predicted-job-first dispatch."""
        while self.waiting:
            eligible = [
                job for job in self.waiting
                if self._eligible_now(job)
            ]

            if not eligible:
                break

            job = min(
                eligible,
                key=lambda j: (
                    self._estimate(j),
                    j.arrival_s,
                    j.job_id,
                ),
            )

            estimated_s = self._estimate(job)

            cluster = min(
                self._eligible_now(job),
                key=lambda c: (
                    self._cluster_duration(c, estimated_s),
                    c.execution_cost(
                        job.resources,
                        self._cluster_duration(c, estimated_s),
                    ),
                    c.name,
                ),
            )

            self.waiting.remove(job)
            self._start(job, cluster, now)

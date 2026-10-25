"""Uncertainty-aware hybrid scheduling with P50 and P90."""

from hybridsched.ml_ablation import MLAblationSimulator


class UncertaintyAwareSimulator(MLAblationSimulator):
    """Online scheduler with deadline-dependent risk levels."""

    def __init__(
        self,
        jobs,
        clusters,
        actual_runtimes,
        predictor,
    ):
        super().__init__(
            jobs,
            clusters,
            actual_runtimes,
            predictor,
            ranking_mode="p50",
            placement_mode="p90",
        )

    def _risk_duration(self, job, now):
        prediction = self.predictions.get(job.job_id)

        if prediction is None:
            prediction = self.predictor.predict_many(
                [job]
            )[job.job_id]
            self.predictions[job.job_id] = prediction

        p50 = prediction.p50_s
        p90 = max(p50, prediction.p90_s)

        if job.deadline_s is None:
            return p50

        compatible = [
            c for c in self.clusters
            if job.resources.fits_in(c.capacity)
        ]

        fastest_p50 = min(
            self._cluster_duration(c, p50)
            for c in compatible
        )

        slack = (
            job.deadline_s - now - fastest_p50
        )

        if slack <= 0:
            risk = 1.0
        elif slack <= 300:
            risk = 0.5
        else:
            risk = 0.0

        return (1 - risk) * p50 + risk * p90

    def _greedy_dispatch(self, now):
        while self.waiting:
            available = [
                j for j in self.waiting
                if self._eligible_now(j)
            ]

            if not available:
                break

            job = min(
                available,
                key=lambda j: (
                    self._risk_duration(j, now),
                    j.arrival_s,
                    j.job_id,
                ),
            )

            estimate = self._risk_duration(job, now)

            def cluster_score(cluster):
                duration = self._cluster_duration(
                    cluster, estimate
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
                    job.resources, duration
                )

                return (
                    lateness,
                    cost,
                    duration,
                    cluster.name,
                )

            cluster = min(
                self._eligible_now(job),
                key=cluster_score,
            )

            self.waiting.remove(job)
            self._start(job, cluster, now)

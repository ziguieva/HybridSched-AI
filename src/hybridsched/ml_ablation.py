"""Decouple runtime-based ranking and cluster placement."""

from hybridsched.ml_dynamic import MLDynamicSimulator


class MLAblationSimulator(MLDynamicSimulator):
    MODES = ("requested", "point", "p50", "p90")

    def __init__(
        self,
        jobs,
        clusters,
        actual_runtimes,
        predictor,
        ranking_mode="requested",
        placement_mode="requested",
    ):
        if ranking_mode not in self.MODES:
            raise ValueError("Invalid ranking mode")
        if placement_mode not in self.MODES:
            raise ValueError("Invalid placement mode")

        super().__init__(
            jobs,
            clusters,
            actual_runtimes=actual_runtimes,
            predictor=predictor,
            mode="requested",
        )

        self.ranking_mode = ranking_mode
        self.placement_mode = placement_mode

        if predictor is None and (
            ranking_mode != "requested"
            or placement_mode != "requested"
        ):
            raise ValueError("ML predictor required")

    def _estimate_mode(self, job, mode):
        if mode == "requested":
            return float(job.requested_walltime_s)

        prediction = self.predictions.get(job.job_id)

        if prediction is None:
            prediction = self.predictor.predict_many(
                [job]
            )[job.job_id]
            self.predictions[job.job_id] = prediction

        if mode == "point":
            return prediction.point_s
        if mode == "p50":
            return prediction.p50_s
        return prediction.p90_s

    def _greedy_dispatch(self, now):
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
                    self._estimate_mode(
                        j, self.ranking_mode
                    ),
                    j.arrival_s,
                    j.job_id,
                ),
            )

            estimate = self._estimate_mode(
                job, self.placement_mode
            )

            def cluster_score(cluster):
                duration = self._cluster_duration(
                    cluster, estimate
                )
                return (
                    duration,
                    cluster.execution_cost(
                        job.resources, duration
                    ),
                    cluster.name,
                )

            cluster = min(
                self._eligible_now(job),
                key=cluster_score,
            )

            self.waiting.remove(job)
            self._start(job, cluster, now)

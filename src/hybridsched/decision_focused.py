"""Decision-focused scheduling under runtime uncertainty."""

from hybridsched.ml_dynamic import MLDynamicSimulator


class DecisionFocusedSimulator(MLDynamicSimulator):
    """Online risk-aware scheduler using P50 and P90.

    Predictions influence decisions.
    Actual runtimes determine completion.
    """

    def __init__(
        self,
        jobs,
        clusters,
        actual_runtimes,
        predictor,
        p50_late_weight=0.25,
        p90_late_weight=0.50,
    ):
        if predictor is None:
            raise ValueError("ML predictor required")

        if min(
            p50_late_weight, p90_late_weight
        ) < 0:
            raise ValueError("Weights must be nonnegative")

        super().__init__(
            jobs,
            clusters,
            actual_runtimes=actual_runtimes,
            predictor=predictor,
            mode="p50",
        )

        self.p50_late_weight = p50_late_weight
        self.p90_late_weight = p90_late_weight

    def _prediction(self, job):
        prediction = self.predictions.get(job.job_id)

        if prediction is None:
            prediction = self.predictor.predict_many(
                [job]
            )[job.job_id]
            self.predictions[job.job_id] = prediction

        return prediction

    def _metrics_for_cluster(self, job, cluster, now):
        pred = self._prediction(job)

        p50 = self._cluster_duration(
            cluster, pred.p50_s
        )
        p90 = self._cluster_duration(
            cluster, pred.p90_s
        )

        if job.deadline_s is None:
            late50 = 0.0
            late90 = 0.0
        else:
            late50 = max(
                0.0, now + p50 - job.deadline_s
            )
            late90 = max(
                0.0, now + p90 - job.deadline_s
            )

        score = (
            p50
            + self.p50_late_weight * late50
            + self.p90_late_weight * late90
        )

        return score, p50, p90, late50, late90

    def _greedy_dispatch(self, now):
        while self.waiting:
            choices = []

            for job in self.waiting:
                for cluster in self._eligible_now(job):
                    score, p50, p90, late50, late90 = (
                        self._metrics_for_cluster(
                            job, cluster, now
                        )
                    )

                    cost = cluster.execution_cost(
                        job.resources, p90
                    )

                    choices.append((
                        score,
                        late90,
                        cost,
                        job.arrival_s,
                        job.job_id,
                        cluster.name,
                        job,
                        cluster,
                    ))

            if not choices:
                break

            choice = min(choices, key=lambda x: x[:6])
            chosen_job = choice[6]
            chosen_cluster = choice[7]

            self.waiting.remove(chosen_job)
            self._start(chosen_job, chosen_cluster, now)

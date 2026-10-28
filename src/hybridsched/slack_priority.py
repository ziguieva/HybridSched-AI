"""Signed-slack deadline priority with ML runtime uncertainty."""

from hybridsched.decision_focused import DecisionFocusedSimulator


class SlackPrioritySimulator(DecisionFocusedSimulator):
    """Joint task/cluster selection using signed P90 slack."""

    def __init__(
        self,
        jobs,
        clusters,
        actual_runtimes,
        predictor,
        slack_weight=0.25,
    ):
        if slack_weight < 0:
            raise ValueError(
                "slack_weight must be nonnegative"
            )

        super().__init__(
            jobs,
            clusters,
            actual_runtimes,
            predictor,
        )

        self.slack_weight = slack_weight

    def _greedy_dispatch(self, now):
        while self.waiting:
            choices = []

            for job in self.waiting:
                eligible_clusters = self._eligible_now(job)

                if not eligible_clusters:
                    continue

                prediction = self._prediction(job)

                for cluster in eligible_clusters:
                    p50 = self._cluster_duration(
                        cluster,
                        prediction.p50_s,
                    )

                    p90 = self._cluster_duration(
                        cluster,
                        prediction.p90_s,
                    )

                    if job.deadline_s is None:
                        score = p50
                    else:
                        slack = (
                            job.deadline_s
                            - now
                            - p90
                        )

                        score = (
                            p50
                            + self.slack_weight * slack
                        )

                    cost = cluster.execution_cost(
                        job.resources,
                        p90,
                    )

                    choices.append((
                        score,
                        cost,
                        job.arrival_s,
                        job.job_id,
                        cluster.name,
                        job,
                        cluster,
                    ))

            if not choices:
                break

            choice = min(
                choices,
                key=lambda x: x[:5],
            )

            chosen_job = choice[5]
            chosen_cluster = choice[6]

            self.waiting.remove(chosen_job)

            # Actual runtime, not predicted runtime.
            self._start(
                chosen_job,
                chosen_cluster,
                now,
            )

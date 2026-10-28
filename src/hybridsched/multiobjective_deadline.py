"""Multiobjective online scheduling with uncertain runtimes."""

from hybridsched.decision_focused import DecisionFocusedSimulator


class MultiobjectiveDeadlineSimulator(DecisionFocusedSimulator):
    """Fixed-weight heuristic for duration, slack and cost."""

    def __init__(
        self,
        jobs,
        clusters,
        actual_runtimes,
        predictor,
        slack_weight=0.20,
        overrun_weight=0.20,
        cost_weight=0.02,
    ):
        if min(
            slack_weight,
            overrun_weight,
            cost_weight,
        ) < 0:
            raise ValueError("Weights must be nonnegative")

        super().__init__(
            jobs, clusters, actual_runtimes, predictor
        )

        self.slack_weight = slack_weight
        self.overrun_weight = overrun_weight
        self.cost_weight = cost_weight

    def _candidate_score(self, job, cluster, now):
        prediction = self._prediction(job)

        p50 = self._cluster_duration(
            cluster, prediction.p50_s
        )
        p90 = self._cluster_duration(
            cluster, prediction.p90_s
        )

        cost = cluster.execution_cost(
            job.resources, p90
        )

        if job.deadline_s is None:
            slack_component = 0.0
            overrun = 0.0
        else:
            slack = job.deadline_s - now - p90
            slack_component = self.slack_weight * slack
            overrun = (
                self.overrun_weight
                * max(0.0, -slack)
            )

        return (
            p50
            + slack_component
            + overrun
            + self.cost_weight * cost
        )

    def _greedy_dispatch(self, now):
        while self.waiting:
            choices = []

            for job in self.waiting:
                for cluster in self._eligible_now(job):
                    score = self._candidate_score(
                        job, cluster, now
                    )

                    choices.append((
                        score,
                        job.arrival_s,
                        job.job_id,
                        cluster.name,
                        job,
                        cluster,
                    ))

            if not choices:
                break

            chosen = min(choices, key=lambda x: x[:4])
            job, cluster = chosen[4], chosen[5]

            self.waiting.remove(job)
            self._start(job, cluster, now)

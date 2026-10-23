"""Experimental positive/negative slack priority policies."""

from dataclasses import dataclass
from math import isfinite

from hybridsched.deadline_aware import DeadlineAwareSimulator


@dataclass(frozen=True)
class RiskWeights:
    duration: float = 1.0
    positive_slack: float = 0.5
    negative_slack: float = 0.0

    def __post_init__(self):
        values = (
            self.duration,
            self.positive_slack,
            self.negative_slack,
        )
        if not all(
            isfinite(v) and v >= 0 for v in values
        ):
            raise ValueError("Invalid risk weights")


class RiskPrioritySimulator(DeadlineAwareSimulator):
    def __init__(
        self,
        jobs,
        clusters,
        weights=None,
        **kwargs,
    ):
        self.weights = weights or RiskWeights()
        if not isinstance(self.weights, RiskWeights):
            raise TypeError("Expected RiskWeights")

        super().__init__(
            jobs,
            clusters,
            priority_policy="resource",
            **kwargs,
        )

    def _priority(self, job, now):
        duration = self._minimum_duration(job)

        if job.deadline_s is None:
            return (
                1,
                duration,
                job.arrival_s,
                job.job_id,
            )

        slack = self._resource_slack(job, now)
        w = self.weights

        score = (
            w.duration * duration
            + w.positive_slack * max(0, slack)
            + w.negative_slack * max(0, -slack)
        )

        return (
            0,
            score,
            job.arrival_s,
            job.job_id,
        )

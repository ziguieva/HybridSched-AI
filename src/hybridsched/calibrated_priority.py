"""Parameterized priority scheduling for HPC experiments."""

from dataclasses import dataclass
from math import isfinite

from hybridsched.deadline_aware import DeadlineAwareSimulator


@dataclass(frozen=True)
class PriorityWeights:
    duration: float = 1.0
    negative_slack: float = 2.0
    gpu: float = 30.0

    def __post_init__(self):
        for value in (
            self.duration,
            self.negative_slack,
            self.gpu,
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(
                    "Weights must be finite and nonnegative"
                )


class CalibratedPrioritySimulator(DeadlineAwareSimulator):
    """Online, non-preemptive, parameterized scheduler."""

    def __init__(
        self,
        jobs,
        clusters,
        weights=None,
        **kwargs,
    ):
        self.weights = (
            weights if weights is not None
            else PriorityWeights()
        )

        if not isinstance(self.weights, PriorityWeights):
            raise TypeError("Expected PriorityWeights")

        super().__init__(
            jobs,
            clusters,
            priority_policy="resource",
            **kwargs,
        )

    def _priority(self, job, now):
        w = self.weights
        duration = self._minimum_duration(job)
        gpu = job.resources.gpus

        if job.deadline_s is None:
            return (
                1,
                w.duration * duration + w.gpu * gpu,
                job.arrival_s,
                job.job_id,
            )

        slack = self._resource_slack(job, now)

        score = (
            w.duration * duration
            + w.negative_slack * max(0, -slack)
            + w.gpu * gpu
        )

        return (
            0,
            score,
            job.arrival_s,
            job.job_id,
        )

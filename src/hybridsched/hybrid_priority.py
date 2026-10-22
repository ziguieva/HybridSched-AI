"""Experimental hybrid priority for online HPC scheduling."""

from hybridsched.deadline_aware import DeadlineAwareSimulator


class HybridPrioritySimulator(DeadlineAwareSimulator):
    """Experimental duration/slack/GPU priority."""

    def __init__(self, jobs, clusters, **kwargs):
        super().__init__(
            jobs,
            clusters,
            priority_policy="resource",
            **kwargs,
        )

    def _priority(self, job, now):
        duration = self._minimum_duration(job)
        gpu_demand = job.resources.gpus

        if job.deadline_s is None:
            return (
                1,
                duration + 30 * gpu_demand,
                job.arrival_s,
                job.job_id,
            )

        slack = self._resource_slack(job, now)

        score = (
            duration
            + 2 * max(0, -slack)
            + 30 * gpu_demand
        )

        return (
            0,
            score,
            job.arrival_s,
            job.job_id,
        )

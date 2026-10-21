"""Risk-aware EASY backfilling using predictive quantiles."""

from collections.abc import Mapping
from dataclasses import dataclass

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.job import Job


@dataclass(frozen=True, slots=True)
class RuntimeQuantiles:
    p10_s: int
    p50_s: int
    p90_s: int

    def __post_init__(self):
        values = (self.p10_s, self.p50_s, self.p90_s)

        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError(
                "Runtime quantiles must be positive integers"
            )

        if not self.p10_s <= self.p50_s <= self.p90_s:
            raise ValueError(
                "Runtime quantiles must be ordered"
            )


class RiskAwareEASYSimulator(EASYBackfillSimulator):
    """
    EASY-style backfilling with risk-aware candidate ranking.

    Admission and reservation safety still use the
    requested walltime, not the predicted quantiles.

    P90 is used as a conservative ranking signal.
    """

    def __init__(
        self,
        cluster: Cluster,
        jobs: list[Job],
        runtime_quantiles: Mapping[str, RuntimeQuantiles],
        actual_runtimes_s: dict[str, int] | None = None,
    ):
        super().__init__(
            cluster,
            jobs,
            actual_runtimes_s=actual_runtimes_s,
        )

        ids = {job.job_id for job in jobs}

        if set(runtime_quantiles) != ids:
            raise ValueError(
                "Quantiles must match all job IDs"
            )

        if any(
            not isinstance(q, RuntimeQuantiles)
            for q in runtime_quantiles.values()
        ):
            raise TypeError(
                "Expected RuntimeQuantiles instances"
            )

        self.quantiles = dict(runtime_quantiles)

    def _dispatch(self):
        self._sync_active()

        while self.waiting:
            head = self.waiting[0]

            if self.cluster.can_allocate(head):
                self.waiting.popleft()
                self._start_job(head)
                continue

            reservation_s = self._reservation_time(head)

            eligible = [
                job
                for job in list(self.waiting)[1:]
                if self._can_backfill(
                    job,
                    head,
                    reservation_s,
                )
            ]

            if not eligible:
                break

            # Rank candidates by their predicted upper quantile.
            selected = min(
                eligible,
                key=lambda job: (
                    self.quantiles[job.job_id].p90_s,
                    self.quantiles[job.job_id].p50_s,
                    job.arrival_s,
                    job.job_id,
                ),
            )

            self.waiting.remove(selected)
            self._start_job(selected)

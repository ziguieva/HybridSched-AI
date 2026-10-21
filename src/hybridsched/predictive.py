"""Prediction-aware HPC scheduling algorithms."""

from collections.abc import Mapping

from hybridsched.backfill import EASYBackfillSimulator
from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.sjf import SJFSimulator


def validate_predictions(
    jobs: list[Job],
    predictions: Mapping[str, int],
) -> dict[str, int]:
    """Validate externally generated runtime estimates."""

    job_ids = {job.job_id for job in jobs}

    if set(predictions) != job_ids:
        raise ValueError(
            "Predictions must match all job IDs"
        )

    if any(
        type(value) is not int or value <= 0
        for value in predictions.values()
    ):
        raise ValueError(
            "Predictions must be positive integers"
        )

    return dict(predictions)


class PredictedSJFSimulator(SJFSimulator):
    """Resource-aware SJF using predicted runtimes."""

    def __init__(
        self,
        cluster: Cluster,
        jobs: list[Job],
        runtime_predictions_s: Mapping[str, int],
        actual_runtimes_s: dict[str, int] | None = None,
    ):
        super().__init__(
            cluster,
            jobs,
            actual_runtimes_s=actual_runtimes_s,
        )

        self.predictions = validate_predictions(
            jobs,
            runtime_predictions_s,
        )

    def _dispatch(self) -> None:
        while self.waiting:
            candidates = sorted(
                self.waiting,
                key=lambda job: (
                    self.predictions[job.job_id],
                    job.arrival_s,
                    job.job_id,
                ),
            )

            selected = next(
                (
                    job for job in candidates
                    if self.cluster.can_allocate(job)
                ),
                None,
            )

            if selected is None:
                break

            self.waiting.remove(selected)
            self.cluster.allocate(selected)

            self.env.process(
                self._execute(selected)
            )


class PredictedEASYBackfillSimulator(
    EASYBackfillSimulator
):
    """
    Prediction-aware EASY backfilling.

    Predictions rank eligible candidates.
    Requested walltime protects reservations.

    Reservation protection assumes actual runtime
    does not exceed requested walltime.
    """

    def __init__(
        self,
        cluster: Cluster,
        jobs: list[Job],
        runtime_predictions_s: Mapping[str, int],
        actual_runtimes_s: dict[str, int] | None = None,
    ):
        super().__init__(
            cluster,
            jobs,
            actual_runtimes_s=actual_runtimes_s,
        )

        self.predictions = validate_predictions(
            jobs,
            runtime_predictions_s,
        )

    def _dispatch(self) -> None:
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

            selected = min(
                eligible,
                key=lambda job: (
                    self.predictions[job.job_id],
                    job.arrival_s,
                    job.job_id,
                ),
            )

            self.waiting.remove(selected)
            self._start_job(selected)

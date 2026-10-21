"""EASY-style backfilling with protected head-job reservation."""

from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.metrics import compute_metrics
from hybridsched.resources import Resources
from hybridsched.simulation import FCFSSimulator


class EASYBackfillSimulator(FCFSSimulator):
    """
    Non-preemptive, EASY-style HPC backfilling.

    The first waiting job receives a protected
    earliest-start reservation.

    Other jobs may backfill only if they cannot
    delay that reservation.

    This version assumes requested walltime equals
    actual execution duration.
    """

    def __init__(
        self,
        cluster: Cluster,
        jobs: list[Job],
        actual_runtimes_s: dict[str, int] | None = None,
    ):
        super().__init__(
            cluster,
            jobs,
            actual_runtimes_s=actual_runtimes_s,
        )

        self._active: dict[str, tuple[Job, int]] = {}

    def _sync_active(self) -> None:
        running = set(self.cluster.running_job_ids)

        self._active = {
            job_id: entry
            for job_id, entry in self._active.items()
            if job_id in running
        }

    def _available_at(self, time_s: int) -> Resources:
        available = self.cluster.available

        for job, finish_s in self._active.values():
            if finish_s <= time_s:
                available = available + job.resources

        return available

    def _reservation_time(self, head: Job) -> int:
        now = int(self.env.now)

        if head.resources.fits_in(self.cluster.available):
            return now

        finish_times = sorted({
            finish_s
            for _, finish_s in self._active.values()
        })

        for finish_s in finish_times:
            if head.resources.fits_in(
                self._available_at(finish_s)
            ):
                return finish_s

        raise RuntimeError(
            f"No feasible reservation for {head.job_id}"
        )

    def _can_backfill(
        self,
        candidate: Job,
        head: Job,
        reservation_s: int,
    ) -> bool:

        if not self.cluster.can_allocate(candidate):
            return False

        candidate_finish = (
            int(self.env.now)
            + candidate.requested_walltime_s
        )

        if candidate_finish <= reservation_s:
            return True

        available_at_reservation = self._available_at(
            reservation_s
        )

        combined_resources = (
            head.resources + candidate.resources
        )

        return combined_resources.fits_in(
            available_at_reservation
        )

    def _start_job(self, job: Job) -> None:
        self.cluster.allocate(job)

        expected_finish = (
            int(self.env.now)
            + job.requested_walltime_s
        )

        self._active[job.job_id] = (
            job,
            expected_finish,
        )

        self.env.process(self._execute(job))

    def _dispatch(self) -> None:
        self._sync_active()

        while self.waiting:
            head = self.waiting[0]

            if self.cluster.can_allocate(head):
                self.waiting.popleft()
                self._start_job(head)
                continue

            reservation_s = self._reservation_time(head)

            selected = None

            for candidate in list(self.waiting)[1:]:
                if self._can_backfill(
                    candidate,
                    head,
                    reservation_s,
                ):
                    selected = candidate
                    break

            if selected is None:
                break

            self.waiting.remove(selected)
            self._start_job(selected)


def main() -> None:

    capacity = Resources(
        cpu_cores=16,
        gpus=4,
        memory_gib=64,
    )

    jobs = [
        Job("J001", 0, 900, Resources(4, 2, 16)),
        Job("J002", 60, 1200, Resources(8, 3, 32)),
        Job("J003", 120, 300, Resources(2, 1, 8)),
    ]

    algorithms = {
        "FCFS": FCFSSimulator,
        "EASY Backfilling": EASYBackfillSimulator,
    }

    print("=" * 65)
    print("HybridSched-AI | FCFS vs EASY BACKFILLING")
    print("=" * 65)

    for name, scheduler in algorithms.items():

        cluster = Cluster(
            name="local-hpc",
            capacity=capacity,
        )

        results = scheduler(cluster, jobs).run()

        metrics = compute_metrics(
            jobs,
            results,
            capacity,
        )

        print(f"\nALGORITHM: {name}")
        print("-" * 65)

        print(
            f"{'Job':<8}"
            f"{'Arrival':>10}"
            f"{'Start':>10}"
            f"{'Finish':>10}"
            f"{'Waiting':>10}"
        )

        for result in results:
            print(
                f"{result.job_id:<8}"
                f"{result.arrival_s:>10}"
                f"{result.start_s:>10}"
                f"{result.finish_s:>10}"
                f"{result.waiting_s:>10}"
            )

        print("-" * 65)
        print(
            f"Average waiting: "
            f"{metrics.average_waiting_s:.2f} s"
        )
        print(f"Makespan: {metrics.makespan_s} s")
        print(
            f"GPU utilization: "
            f"{metrics.gpu_utilization_pct:.2f}%"
        )
        print(
            f"Throughput: "
            f"{metrics.throughput_jobs_hour:.2f} jobs/h"
        )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()

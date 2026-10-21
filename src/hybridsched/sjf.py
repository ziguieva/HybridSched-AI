"""Resource-aware Shortest Job First HPC scheduling."""

from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.metrics import compute_metrics
from hybridsched.resources import Resources
from hybridsched.simulation import FCFSSimulator


class SJFSimulator(FCFSSimulator):
    """
    Non-preemptive Shortest Job First scheduler.

    Select the shortest runnable job among all
    currently waiting jobs.

    Requested walltime is used as the duration
    estimate in this initial implementation.
    """

    def _dispatch(self) -> None:
        while self.waiting:

            candidates = sorted(
                self.waiting,
                key=lambda job: (
                    job.requested_walltime_s,
                    job.arrival_s,
                    job.job_id,
                ),
            )

            selected = None

            for job in candidates:
                if self.cluster.can_allocate(job):
                    selected = job
                    break

            if selected is None:
                break

            self.waiting.remove(selected)

            self.cluster.allocate(selected)

            self.env.process(
                self._execute(selected)
            )


def main() -> None:

    capacity = Resources(
        cpu_cores=16,
        gpus=4,
        memory_gib=64,
    )

    jobs = [
        Job(
            "J001",
            arrival_s=0,
            requested_walltime_s=900,
            resources=Resources(4, 2, 16),
        ),
        Job(
            "J002",
            arrival_s=60,
            requested_walltime_s=1200,
            resources=Resources(8, 3, 32),
        ),
        Job(
            "J003",
            arrival_s=120,
            requested_walltime_s=300,
            resources=Resources(2, 1, 8),
        ),
    ]

    algorithms = {
        "FCFS": FCFSSimulator,
        "SJF": SJFSimulator,
    }

    print("=" * 68)
    print("HybridSched-AI | FCFS vs SJF COMPARISON")
    print("=" * 68)

    for name, simulator_class in algorithms.items():

        cluster = Cluster(
            name=f"{name.lower()}-hpc",
            capacity=capacity,
        )

        simulator = simulator_class(
            cluster,
            jobs,
        )

        results = simulator.run()

        metrics = compute_metrics(
            jobs,
            results,
            capacity,
        )

        print(f"\nALGORITHM: {name}")
        print("-" * 68)

        print(
            f"{'Job':<10}"
            f"{'Arrival':>10}"
            f"{'Start':>10}"
            f"{'Finish':>10}"
            f"{'Waiting':>10}"
        )

        for result in results:
            print(
                f"{result.job_id:<10}"
                f"{result.arrival_s:>10}"
                f"{result.start_s:>10}"
                f"{result.finish_s:>10}"
                f"{result.waiting_s:>10}"
            )

        print("-" * 68)

        print(
            f"Average waiting: "
            f"{metrics.average_waiting_s:.2f} s"
        )

        print(
            f"Makespan: "
            f"{metrics.makespan_s} s"
        )

        print(
            f"GPU utilization: "
            f"{metrics.gpu_utilization_pct:.2f}%"
        )

        print(
            f"Throughput: "
            f"{metrics.throughput_jobs_hour:.2f} jobs/h"
        )

    print("\n" + "=" * 68)


if __name__ == "__main__":
    main()

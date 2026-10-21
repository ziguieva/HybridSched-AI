"""HybridSched-AI initial HPC demonstration."""

from hybridsched.cluster import Cluster
from hybridsched.job import Job
from hybridsched.resources import Resources


def main() -> None:
    cluster = Cluster(
        "local-hpc",
        Resources(
            cpu_cores=16,
            gpus=4,
            memory_gib=64,
        ),
    )

    job_a = Job(
        "J001",
        arrival_s=0,
        requested_walltime_s=900,
        resources=Resources(
            cpu_cores=4,
            gpus=2,
            memory_gib=16,
        ),
    )

    job_b = Job(
        "J002",
        arrival_s=60,
        requested_walltime_s=1200,
        resources=Resources(
            cpu_cores=8,
            gpus=3,
            memory_gib=32,
        ),
        priority=1,
    )

    print("Cluster:", cluster.name)
    print("Capacity:", cluster.capacity)

    cluster.allocate(job_a)
    print("After allocation J001:", cluster.available)

    print(
        "Can allocate J002 now?",
        cluster.can_allocate(job_b),
    )

    cluster.release(job_a.job_id)
    print("After release J001:", cluster.available)

    cluster.allocate(job_b)
    print("After allocation J002:", cluster.available)

    print("Running jobs:", cluster.running_job_ids)


if __name__ == "__main__":
    main()

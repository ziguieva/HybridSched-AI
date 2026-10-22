"""Individual deadline feasibility lower bounds."""


def analyze_deadline_feasibility(jobs, clusters):
    """Analyze independently achievable deadlines.

    Assumes deterministic cluster execution times
    and no queueing. Does not prove joint feasibility.
    """
    if not clusters:
        raise ValueError("At least one cluster required")

    records = []

    for job in jobs:
        candidates = [
            cluster
            for cluster in clusters
            if job.resources.fits_in(cluster.capacity)
        ]

        if not candidates:
            raise ValueError(
                f"No eligible cluster: {job.job_id}"
            )

        best_duration = min(
            c.execution_time(job.requested_walltime_s)
            for c in candidates
        )

        earliest_finish = (
            job.arrival_s + best_duration
        )

        if job.deadline_s is None:
            impossible = False
            unavoidable_tardiness = 0
        else:
            unavoidable_tardiness = max(
                0,
                earliest_finish - job.deadline_s,
            )
            impossible = unavoidable_tardiness > 0

        records.append({
            "job_id": job.job_id,
            "earliest_finish_s": earliest_finish,
            "deadline_s": job.deadline_s,
            "intrinsically_late": impossible,
            "minimum_tardiness_s": (
                unavoidable_tardiness
            ),
        })

    return {
        "total_jobs": len(jobs),
        "intrinsically_late": sum(
            r["intrinsically_late"]
            for r in records
        ),
        "minimum_tardiness_s": sum(
            r["minimum_tardiness_s"]
            for r in records
        ),
        "records": records,
    }

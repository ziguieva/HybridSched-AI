import pytest

from hybridsched.adaptive import AdaptiveHybridSimulator
from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.resources import Resources


def clusters():
    return [
        ClusterSpec(
            "local", "ON_PREMISE",
            Resources(4, 2, 16),
        ),
        ClusterSpec(
            "cloud", "GCP_SIMULATED",
            Resources(4, 2, 16),
            gpu_cost_per_s=5,
        ),
    ]


def jobs(count=12):
    return [
        Job(
            f"J{i:03d}",
            i * 5,
            100 + (i % 3) * 50,
            Resources(2, 1, 8),
            deadline_s=200 + i * 10,
        )
        for i in range(count)
    ]


def test_all_jobs_complete():
    results = AdaptiveHybridSimulator(
        jobs(), clusters()
    ).run()

    assert len(results) == 12
    assert len({r.job_id for r in results}) == 12


def test_no_start_before_arrival():
    results = AdaptiveHybridSimulator(
        jobs(), clusters()
    ).run()

    assert all(r.start_s >= r.arrival_s for r in results)


def test_cp_call_limit():
    simulator = AdaptiveHybridSimulator(
        jobs(20),
        clusters(),
        cp_threshold=2,
        max_cp_calls=2,
        time_limit_s=0.1,
    )

    simulator.run()
    assert simulator.solver_calls <= 2


def test_deadline_priority():
    workload = [
        Job("LATE", 0, 200, Resources(2, 2, 8),
            deadline_s=500),
        Job("URGENT", 0, 100, Resources(2, 2, 8),
            deadline_s=120),
    ]

    results = AdaptiveHybridSimulator(
        workload,
        clusters()[:1],
        cp_threshold=10,
    ).run()

    by_id = {r.job_id: r for r in results}

    assert by_id["URGENT"].start_s == 0
    assert by_id["LATE"].start_s == 100


@pytest.mark.parametrize(
    "threshold,max_calls",
    [(0, 2), (2, -1)],
)
def test_invalid_parameters(threshold, max_calls):
    with pytest.raises(ValueError):
        AdaptiveHybridSimulator(
            jobs(),
            clusters(),
            cp_threshold=threshold,
            max_cp_calls=max_calls,
        )


def test_no_resource_overcommit():
    workload = jobs(20)
    infrastructure = clusters()

    results = AdaptiveHybridSimulator(
        workload, infrastructure
    ).run()

    requirements = {
        job.job_id: job.resources
        for job in workload
    }

    for cluster in infrastructure:
        selected = [
            r for r in results
            if r.cluster == cluster.name
        ]

        event_times = sorted({
            t
            for r in selected
            for t in (r.start_s, r.finish_s)
        })

        for t in event_times:
            active = [
                requirements[r.job_id]
                for r in selected
                if r.start_s <= t < r.finish_s
            ]

            assert sum(r.cpu_cores for r in active) <= cluster.capacity.cpu_cores
            assert sum(r.gpus for r in active) <= cluster.capacity.gpus
            assert sum(r.memory_gib for r in active) <= cluster.capacity.memory_gib

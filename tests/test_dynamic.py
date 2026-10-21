"""Tests for online hybrid scheduling."""

import pytest

from hybridsched.dynamic import DynamicHybridSimulator
from hybridsched.infrastructure import ClusterSpec
from hybridsched.job import Job
from hybridsched.resources import Resources


def clusters():
    return [
        ClusterSpec(
            "local",
            "ON_PREMISE",
            Resources(4, 2, 16),
        ),
        ClusterSpec(
            "cloud",
            "GCP_SIMULATED",
            Resources(4, 2, 16),
            gpu_cost_per_s=5,
        ),
    ]


def jobs():
    return [
        Job("A", 0, 100, Resources(2, 2, 8)),
        Job("B", 10, 50, Resources(2, 2, 8)),
        Job("C", 20, 30, Resources(2, 1, 4)),
        Job("D", 40, 20, Resources(2, 1, 4)),
    ]


@pytest.mark.parametrize(
    "policy",
    ["fcfs", "sjf", "rolling"],
)
def test_dynamic_capacity_and_arrival(policy):
    infrastructure = clusters()

    simulation = DynamicHybridSimulator(
        jobs(),
        infrastructure,
        policy=policy,
        time_limit_s=2,
    )

    results = simulation.run()

    assert len(results) == 4

    for result in results:
        assert result.start_s >= result.arrival_s
        assert result.waiting_s >= 0
        assert result.finish_s > result.start_s

    # Rebuild instantaneous allocations from results.
    for cluster in infrastructure:
        cluster_results = [
            r for r in results
            if r.cluster == cluster.name
        ]

        job_map = {j.job_id: j for j in jobs()}

        event_times = sorted({
            t
            for result in cluster_results
            for t in (
                result.start_s,
                result.finish_s,
            )
        })

        for t in event_times:
            active = [
                job_map[r.job_id].resources
                for r in cluster_results
                if r.start_s <= t < r.finish_s
            ]

            assert sum(
                x.cpu_cores for x in active
            ) <= cluster.capacity.cpu_cores

            assert sum(
                x.gpus for x in active
            ) <= cluster.capacity.gpus

            assert sum(
                x.memory_gib for x in active
            ) <= cluster.capacity.memory_gib

    assert all(
        used == Resources()
        for used in simulation.used.values()
    )


def test_online_future_job_cannot_affect_first_start():
    initial = [
        Job("A", 0, 100, Resources(2, 2, 8)),
    ]

    future = Job(
        "FUTURE",
        50,
        10,
        Resources(2, 2, 8),
    )

    first = DynamicHybridSimulator(
        initial,
        clusters(),
        policy="rolling",
    ).run()

    second = DynamicHybridSimulator(
        initial + [future],
        clusters(),
        policy="rolling",
    ).run()

    first_a = next(
        r for r in first if r.job_id == "A"
    )

    second_a = next(
        r for r in second if r.job_id == "A"
    )

    assert first_a.start_s == 0
    assert second_a.start_s == 0


def test_reproducibility():
    def execute():
        return DynamicHybridSimulator(
            jobs(),
            clusters(),
            policy="fcfs",
        ).run()

    assert execute() == execute()


def test_empty_workload():
    simulator = DynamicHybridSimulator(
        [],
        clusters(),
        policy="rolling",
    )

    assert simulator.run() == []


def test_impossible_job_rejected():
    impossible = [
        Job("A", 0, 100, Resources(2, 8, 8)),
    ]

    with pytest.raises(ValueError, match="eligible"):
        DynamicHybridSimulator(
            impossible,
            clusters(),
        )


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        DynamicHybridSimulator(
            jobs(),
            clusters(),
            policy="invalid",
        )


def test_duplicate_ids_rejected():
    job = jobs()[0]

    with pytest.raises(ValueError, match="Duplicate"):
        DynamicHybridSimulator(
            [job, job],
            clusters(),
        )

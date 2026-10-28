"""STEP 24 independent workload and reproducibility tests."""

import json
from pathlib import Path

import pytest

from experiments.step24_independent_benchmark import generate_workload
from experiments.step24_research_report import paired_ci


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "experiments/config/step24_protocol.json").read_text()
)


def test_reproducible_generation():
    assert generate_workload(24001, CONFIG) == generate_workload(24001, CONFIG)


def test_different_seeds_differ():
    assert generate_workload(24001, CONFIG) != generate_workload(24002, CONFIG)


def test_correct_job_count():
    jobs, actual = generate_workload(24001, CONFIG)
    assert len(jobs) == 150
    assert len(actual) == 150


def test_all_runtimes_positive():
    _, actual = generate_workload(24001, CONFIG)
    assert all(value > 0 for value in actual.values())


def test_deadlines_after_arrival():
    jobs, _ = generate_workload(24001, CONFIG)
    assert all(j.deadline_s > j.arrival_s for j in jobs)


def test_constant_paired_bootstrap():
    assert paired_ci([-10] * 30) == (-10, -10, -10)


def test_bootstrap_requires_multiple_scenarios():
    with pytest.raises(ValueError):
        paired_ci([1])


def test_generated_ids_are_distinct_from_original_dataset():
    jobs, _ = generate_workload(24001, CONFIG)
    assert all(j.job_id.startswith("NEW_") for j in jobs)
    assert len({j.job_id for j in jobs}) == len(jobs)

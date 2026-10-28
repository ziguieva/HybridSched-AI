"""STEP 19 calibration and Pareto regression tests."""

from dataclasses import dataclass

from hybridsched.conditional_calibration import (
    CalibratedRuntimePredictor,
    duration_group,
    upper_quantile,
)
from experiments.pareto_ml_policies import dominates


@dataclass(frozen=True)
class Prediction:
    point_s: float = 40
    p10_s: float = 20
    p50_s: float = 40
    p90_s: float = 80


class Predictor:
    def predict_many(self, jobs):
        return {job.job_id: Prediction() for job in jobs}


@dataclass
class FakeJob:
    job_id: str
    requested_walltime_s: float


def test_duration_boundaries():
    assert duration_group(300) == "SHORT"
    assert duration_group(900) == "MEDIUM"
    assert duration_group(901) == "LONG"


def test_upper_quantile():
    assert upper_quantile(list(range(1, 11)), 0.9) == 10


def test_calibration_adds_positive_margin():
    model = CalibratedRuntimePredictor(
        Predictor(), {"LONG": 35}
    )
    result = model.predict_many([FakeJob("A", 1200)])
    assert result["A"].p90_s == 115


def test_calibration_never_reduces_p90():
    model = CalibratedRuntimePredictor(
        Predictor(), {"SHORT": -100}
    )
    result = model.predict_many([FakeJob("A", 100)])
    assert result["A"].p90_s == 80


def test_pareto_dominance():
    better = {
        "cost_units": 100,
        "tardiness_s": 200,
        "makespan_s": 300,
    }
    worse = {
        "cost_units": 120,
        "tardiness_s": 200,
        "makespan_s": 310,
    }
    assert dominates(better, worse)
    assert not dominates(worse, better)

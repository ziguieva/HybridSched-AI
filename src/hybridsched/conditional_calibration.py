"""Groupwise upper-quantile recalibration."""

import math
from dataclasses import replace

from hybridsched.ml_predictor import MLRuntimePredictor


def duration_group(seconds):
    if seconds <= 300:
        return "SHORT"
    if seconds <= 900:
        return "MEDIUM"
    return "LONG"


def upper_quantile(values, level=0.90):
    if not values:
        raise ValueError("Empty calibration group")
    ordered = sorted(float(v) for v in values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil((len(ordered) + 1) * level) - 1),
    )
    return ordered[index]


class CalibratedRuntimePredictor:
    def __init__(self, base_predictor, corrections):
        self.base_predictor = base_predictor
        self.corrections = dict(corrections)

    def predict_many(self, jobs):
        raw = self.base_predictor.predict_many(jobs)
        output = {}

        for job in jobs:
            prediction = raw[job.job_id]
            group = duration_group(job.requested_walltime_s)
            correction = max(
                0.0, self.corrections.get(group, 0.0)
            )
            output[job.job_id] = replace(
                prediction,
                p90_s=max(
                    prediction.p50_s,
                    prediction.p90_s + correction,
                ),
            )
        return output


def fit_corrections(jobs, actual, predictor, level=0.90):
    predictions = predictor.predict_many(jobs)
    groups = {}

    for job in jobs:
        group = duration_group(job.requested_walltime_s)
        groups.setdefault(group, []).append(
            actual[job.job_id]
            - predictions[job.job_id].p90_s
        )

    return {
        group: max(0.0, upper_quantile(errors, level))
        for group, errors in groups.items()
    }

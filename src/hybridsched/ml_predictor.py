"""XGBoost runtime and quantile predictions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb


FEATURES = (
    "requested_walltime_s",
    "cpu_cores",
    "gpus",
    "memory_gib",
    "priority",
    "arrival_s",
)


@dataclass(frozen=True)
class RuntimePrediction:
    point_s: float
    p10_s: float
    p50_s: float
    p90_s: float


def job_features(job):
    return [
        float(job.requested_walltime_s),
        float(job.resources.cpu_cores),
        float(job.resources.gpus),
        float(job.resources.memory_gib),
        float(job.priority),
        float(job.arrival_s),
    ]


class MLRuntimePredictor:
    def __init__(self, root=None):
        root = (
            Path(root)
            if root is not None
            else Path(__file__).resolve().parents[2]
        )

        self.point_model = xgb.Booster()
        self.point_model.load_model(
            str(root / "results/runtime_xgboost.json")
        )

        self.quantile_model = xgb.Booster()
        self.quantile_model.load_model(
            str(root / "results/runtime_quantiles.json")
        )

        if (
            self.point_model.num_features() != len(FEATURES)
            or self.quantile_model.num_features() != len(FEATURES)
        ):
            raise ValueError("Model feature count mismatch")

    def predict_many(self, jobs):
        if not jobs:
            return {}

        matrix = np.asarray(
            [job_features(j) for j in jobs],
            dtype=np.float32,
        )
        dmatrix = xgb.DMatrix(matrix)

        point = np.asarray(
            self.point_model.predict(dmatrix)
        ).reshape(-1)

        quantiles = np.asarray(
            self.quantile_model.predict(dmatrix)
        )

        if (
            quantiles.ndim != 2
            or quantiles.shape != (len(jobs), 3)
        ):
            raise ValueError(
                f"Expected N x 3 quantiles, got "
                f"{quantiles.shape}. Check training configuration."
            )

        if len(point) != len(jobs):
            raise ValueError("Point prediction count mismatch")

        result = {}

        for i, job in enumerate(jobs):
            q10, q50, q90 = [
                max(1.0, float(v))
                for v in quantiles[i]
            ]

            if not np.all(np.isfinite(
                [point[i], q10, q50, q90]
            )):
                raise ValueError("Non-finite ML prediction")

            # Enforce non-crossing predicted quantiles.
            q10, q50, q90 = sorted((q10, q50, q90))

            result[job.job_id] = RuntimePrediction(
                point_s=max(1.0, float(point[i])),
                p10_s=q10,
                p50_s=q50,
                p90_s=q90,
            )

        return result

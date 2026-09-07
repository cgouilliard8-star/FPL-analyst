"""Baselines, in ascending order of how embarrassing it is to lose to them.

A model is only as impressive as the thing it beats, so these are scored on exactly
the same folds as everything else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl.evaluate.walkforward import Predictor


class ConstantPredictor(Predictor):
    """Predict the training-set mean for everyone. The floor."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(np.full(len(frame), self.value), index=frame.index)


class ColumnPredictor(Predictor):
    """Read an existing column straight out of the feature table.

    Used for rolling-average form and for FPL's own published expected points.
    """

    def __init__(self, column: str, fill: float = 0.0):
        self.column = column
        self.fill = fill

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return frame[self.column].fillna(self.fill)


class LightGBMPredictor(Predictor):
    """One gradient-boosted regressor fitted straight onto total points."""

    def __init__(self, model, features: list[str]):
        self.model = model
        self.features = features

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        values = self.model.predict(frame[self.features])
        return pd.Series(values, index=frame.index)


def fit_constant(train: pd.DataFrame, features: list[str]) -> ConstantPredictor:  # noqa: ARG001
    return ConstantPredictor(float(train["total_points"].mean()))


def fit_rolling3(train: pd.DataFrame, features: list[str]) -> ColumnPredictor:  # noqa: ARG001
    return ColumnPredictor("total_points_mean3")


def fit_fpl_xp(train: pd.DataFrame, features: list[str]) -> ColumnPredictor:  # noqa: ARG001
    """FPL's own expected-points figure, shipped free in the data."""
    return ColumnPredictor("xP")


def fit_lightgbm(
    train: pd.DataFrame,
    features: list[str],
    *,
    target: str = "total_points",
    **params,
) -> LightGBMPredictor:
    """Fit the monolithic baseline: one regressor, one target."""
    import lightgbm as lgb

    settings = {
        "objective": "regression",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
    }
    settings.update(params)

    model = lgb.LGBMRegressor(**settings)
    model.fit(train[features], train[target])
    return LightGBMPredictor(model, features)

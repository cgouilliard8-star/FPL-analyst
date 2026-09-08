"""The sub-models. One per way of scoring FPL points.

A defender's score is produced by three nearly unrelated processes: whether he plays,
whether his team keeps the ball out, and whether he happens to attack. Forcing one
regressor to learn all of them at once spends its capacity mediating between them.
Fitting them separately is both more accurate and, because each term survives into the
output, explainable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Shared tree settings. Deliberately conservative: the signal here is weak and the
# class balance brutal (60% of player-gameweeks score nothing).
# Chosen by scripts/tune.py on walk-forward folds of 2024-25: shallower trees, half
# the features per tree and a slower learning rate beat the earlier defaults on
# precision@10 (0.476 -> 0.515 on the coarse folds) with equal rank correlation --
# the sub-models were overfitting the noisy count targets.
_COMMON = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 60,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


@dataclass
class ComponentSpec:
    """One sub-model: what it predicts and how."""

    name: str
    target: str
    kind: str  # "binary" or "count"
    description: str


# Every event that FPL scores per fixture is modelled as a count, so that a double
# gameweek is worth double. Only the rotation-risk signal shown to the reader is a
# probability, and it is derived from the count rather than fitted separately.
SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec("e_appearances", "appearances", "count", "expected appearances"),
    ComponentSpec("e_full", "full_appearances", "count", "expected 60-minute appearances"),
    ComponentSpec("e_goals", "goals_scored", "count", "expected goals scored"),
    ComponentSpec("e_assists", "assists", "count", "expected assists"),
    ComponentSpec("e_cs", "clean_sheets", "count", "expected clean sheets"),
    ComponentSpec("e_saves", "saves", "count", "expected saves"),
    ComponentSpec("e_conceded", "goals_conceded", "count", "expected goals conceded"),
    ComponentSpec("e_bonus", "bonus", "count", "expected bonus points"),
    ComponentSpec("e_dc", "dc_hits", "count", "expected defensive-contribution awards"),
)

REQUIRED_TARGETS = tuple(spec.target for spec in SPECS)


def prepare_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Check the count targets exist. They are produced by the aggregation step."""
    missing = [t for t in REQUIRED_TARGETS if t not in frame.columns]
    if missing:
        raise ValueError(f"training frame is missing component targets {missing}")
    return frame


def _fit_one(spec: ComponentSpec, train: pd.DataFrame, features: list[str]):
    import lightgbm as lgb

    y = train[spec.target]
    if spec.kind == "binary":
        model = lgb.LGBMClassifier(objective="binary", **_COMMON)
    else:
        # Poisson: these targets are non-negative counts with a heavy zero mass.
        model = lgb.LGBMRegressor(objective="poisson", **_COMMON)
        y = y.clip(lower=0)

    if float(y.sum()) == 0.0:
        # An event that has never happened in the training window (defensive
        # contribution before the 2025-26 rules, in an early-season backtest) has
        # nothing to learn: predict zero rather than fail the whole ensemble.
        return _Zero()

    model.fit(train[features], y)
    return model


class _Zero:
    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class ComponentEnsemble:
    """The fitted sub-models, one per scoring event."""

    def __init__(self, models: dict[str, object], features: list[str]):
        self.models = models
        self.features = features

    @classmethod
    def fit(cls, train: pd.DataFrame, features: list[str]) -> ComponentEnsemble:
        train = prepare_targets(train)
        models = {}
        for spec in SPECS:
            models[spec.name] = _fit_one(spec, train, features)
            log.debug("fitted %s (%s)", spec.name, spec.description)
        return cls(models, features)

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Raw sub-model outputs, one column per component."""
        X = frame[self.features]
        out = pd.DataFrame(index=frame.index)
        for spec in SPECS:
            model = self.models[spec.name]
            if spec.kind == "binary":
                out[spec.name] = model.predict_proba(X)[:, 1]
            else:
                out[spec.name] = np.clip(model.predict(X), 0.0, None)

        # A player cannot appear more often than his club plays, nor play 60 minutes
        # more often than he appears; the Poisson models do not know that, so clip.
        fixtures = frame["fixtures_this_gw"].clip(lower=1).to_numpy()
        out["e_appearances"] = np.minimum(out["e_appearances"].to_numpy(), fixtures)
        out["e_full"] = np.minimum(out["e_full"].to_numpy(), out["e_appearances"].to_numpy())
        out["e_cs"] = np.minimum(out["e_cs"].to_numpy(), out["e_full"].to_numpy())

        # Rotation risk for the reader: expected 60-minute appearances per fixture,
        # which for a single-fixture week is simply the probability of starting.
        out["p_60"] = np.clip(out["e_full"].to_numpy() / fixtures, 0.0, 1.0)
        return out

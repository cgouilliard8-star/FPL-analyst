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
#
# The two minutes components are fitted as per-fixture *rates* (label = events per
# fixture, in [0, 1], cross-entropy objective) and multiplied back by the number of
# fixtures: a probability of playing is what they really are, and a classifier
# calibrates that better than a Poisson count does. Set ``MINUTES_MODEL`` to
# "poisson" to get the older behaviour.
MINUTES_MODEL = "rate"

# Recency weighting of the training rows, as a half-life in days (0 = off). Football
# changes: a rule change, a new manager, a squad turned over. Down-weighting old
# seasons lets the trees follow the game as it is played now.
RECENCY_HALFLIFE_DAYS = 0.0
# Bagging: average this many fits with different seeds. Trades fit time for a
# little variance -- the count targets are noisy and one seed's split choices show.
SEEDS = 1

SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec("e_appearances", "appearances", "rate", "expected appearances"),
    ComponentSpec("e_full", "full_appearances", "rate", "expected 60-minute appearances"),
    ComponentSpec("e_goals", "goals_scored", "count", "expected goals scored"),
    ComponentSpec("e_assists", "assists", "count", "expected assists"),
    ComponentSpec("e_cs", "clean_sheets", "count", "expected clean sheets"),
    ComponentSpec("e_saves", "saves", "count", "expected saves"),
    ComponentSpec("e_conceded", "goals_conceded", "count", "expected goals conceded"),
    ComponentSpec("e_bonus", "bonus", "count", "expected bonus points"),
    ComponentSpec("e_dc", "dc_hits", "count", "expected defensive-contribution awards"),
    # Not a scoring event: the chance of a haul, for the armband. Binary, so it is
    # a calibrated probability, and it passes through the decomposition untouched.
    ComponentSpec("p_haul", "haul", "binary", "chance of a haul (8+ points)"),
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
    kind = spec.kind if not (spec.kind == "rate" and MINUTES_MODEL == "poisson") else "count"
    if kind == "binary":
        model = lgb.LGBMClassifier(objective="binary", **_COMMON)
    elif kind == "rate":
        # Events per fixture, a label in [0, 1]; LightGBM's cross-entropy objective
        # accepts fractional labels, so a double gameweek with one appearance is 0.5.
        model = lgb.LGBMRegressor(objective="cross_entropy", **_COMMON)
        y = (y.clip(lower=0) / train["fixtures_this_gw"].clip(lower=1)).clip(0.0, 1.0)
    else:
        # Poisson: these targets are non-negative counts with a heavy zero mass.
        model = lgb.LGBMRegressor(objective="poisson", **_COMMON)
        y = y.clip(lower=0)

    if float(y.sum()) == 0.0:
        # An event that has never happened in the training window (defensive
        # contribution before the 2025-26 rules, in an early-season backtest) has
        # nothing to learn: predict zero rather than fail the whole ensemble.
        return _Zero()

    weight = None
    if RECENCY_HALFLIFE_DAYS > 0 and "kickoff_time" in train.columns:
        age = (train["kickoff_time"].max() - train["kickoff_time"]).dt.total_seconds() / 86400.0
        weight = np.power(0.5, age.to_numpy() / RECENCY_HALFLIFE_DAYS)
    if SEEDS > 1 and kind != "binary":
        members = []
        for seed in range(SEEDS):
            member = lgb.LGBMRegressor(**{**model.get_params(), "random_state": 42 + seed})
            member.fit(train[features], y, sample_weight=weight)
            members.append(member)
        return _Bag(members)
    model.fit(train[features], y, sample_weight=weight)
    return model


class _Bag:
    def __init__(self, members):
        self.members = members

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)


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
        fixtures = frame["fixtures_this_gw"].clip(lower=1).to_numpy()
        for spec in SPECS:
            model = self.models[spec.name]
            if spec.kind == "binary":
                out[spec.name] = model.predict_proba(X)[:, 1]
            elif spec.kind == "rate" and MINUTES_MODEL != "poisson":
                out[spec.name] = np.clip(model.predict(X), 0.0, 1.0) * fixtures
            else:
                out[spec.name] = np.clip(model.predict(X), 0.0, None)

        # A player cannot appear more often than his club plays, nor play 60 minutes
        # more often than he appears; the Poisson models do not know that, so clip.
        out["e_appearances"] = np.minimum(out["e_appearances"].to_numpy(), fixtures)
        out["e_full"] = np.minimum(out["e_full"].to_numpy(), out["e_appearances"].to_numpy())
        out["e_cs"] = np.minimum(out["e_cs"].to_numpy(), out["e_full"].to_numpy())

        # Rotation risk for the reader: expected 60-minute appearances per fixture,
        # which for a single-fixture week is simply the probability of starting.
        out["p_60"] = np.clip(out["e_full"].to_numpy() / fixtures, 0.0, 1.0)
        return out

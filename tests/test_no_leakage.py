"""The spine of the project.

A feature describing gameweek *t* must be computable from data timestamped strictly
before gameweek *t*'s deadline. The test does not inspect the feature code; it
rewrites the future and asserts the past does not move. Any leak — a stray
``groupby().mean()`` over a whole season, a forgotten ``shift(1)`` — changes a past
row when a future row changes, and this fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl.data.archive import load_players
from fpl.data.silver import load_silver
from fpl.features.build import build_features, feature_columns

CUTOFF_GW = 20
SEASONS = ("2023-24",)

# Columns that describe the current gameweek's outcome. They are the labels, not
# features, and are expected to change when the future is rewritten.
OUTCOME_COLUMNS = {
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "starts",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "played",
    "started",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
}

CORRUPTIBLE = sorted(OUTCOME_COLUMNS - {"played", "started"})


@pytest.fixture(scope="module")
def silver() -> pd.DataFrame:
    frame = load_silver()
    subset = frame[frame["season"].isin(SEASONS)].copy()
    if subset.empty:
        pytest.skip("silver table not built; run `fpl silver` first")
    return subset


@pytest.fixture(scope="module")
def players() -> pd.DataFrame:
    return load_players(SEASONS)


def _build(frame: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    return build_features(SEASONS, write=False, silver=frame, players=players)


def test_rewriting_the_future_does_not_change_the_past(silver, players):
    """The load-bearing assertion. Corrupt every gameweek after the cutoff; the
    features for gameweeks up to the cutoff must be bit-identical."""
    honest = _build(silver, players)

    corrupted = silver.copy()
    future = corrupted["GW"] > CUTOFF_GW
    assert future.any(), "fixture must contain gameweeks after the cutoff"
    for column in CORRUPTIBLE:
        if column in corrupted.columns:
            corrupted.loc[future, column] = corrupted.loc[future, column] * 1000 + 999

    tampered = _build(corrupted, players)

    key = ["code", "season", "GW"]
    features = [c for c in feature_columns(honest) if c not in OUTCOME_COLUMNS]

    a = honest[honest["GW"] <= CUTOFF_GW].sort_values(key).reset_index(drop=True)
    b = tampered[tampered["GW"] <= CUTOFF_GW].sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) and len(a) > 0

    leaking = []
    for column in features:
        left, right = a[column].to_numpy(dtype="float64"), b[column].to_numpy(dtype="float64")
        if not np.allclose(left, right, equal_nan=True, rtol=1e-9, atol=1e-9):
            leaking.append(column)

    assert not leaking, (
        f"{len(leaking)} feature(s) changed when only the future was rewritten, so they "
        f"depend on data after the deadline they predict: {leaking[:12]}"
    )


def test_first_gameweek_has_no_history(silver, players):
    """A player's debut cannot carry form. If it does, the shift is missing."""
    features = _build(silver, players)
    debut = features.sort_values(["code", "kickoff_time"]).groupby("code").head(1)
    assert (debut["games_played"] == 0).all()
    assert debut["total_points_mean1"].isna().all(), "debut rows must have no prior form"


def test_lagged_form_equals_previous_gameweek_points(silver, players):
    """``total_points_mean1`` must be exactly the previous gameweek's points."""
    features = _build(silver, players).sort_values(["code", "kickoff_time"])
    expected = features.groupby("code")["total_points"].shift(1)
    actual = features["total_points_mean1"]
    both = expected.notna() & actual.notna()
    assert both.sum() > 1000
    np.testing.assert_allclose(actual[both], expected[both], rtol=1e-9)


def test_feature_columns_exclude_the_target(silver, players):
    features = _build(silver, players)
    assert "total_points" not in feature_columns(features)
    assert "played" not in feature_columns(features)

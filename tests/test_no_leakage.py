"""The spine of the project.

A feature describing gameweek *t* must be computable from data timestamped strictly
before gameweek *t*'s deadline. The test does not inspect the feature code; it
rewrites outcomes and asserts the features do not move.

Two corruptions are applied, because they catch different leaks:

* Rewriting every gameweek *after* the cutoff catches a feature that looks forward --
  a whole-season mean, a missing ``shift(1)``.
* Rewriting the cutoff gameweek *itself* catches a feature that looks sideways -- a
  prior that accumulates row by row and so absorbs team-mates' results from the very
  match being predicted. The first version of this suite only did the former, and
  the position prior leaked sideways for weeks without it noticing.
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

# Columns that describe an outcome. They are labels or label ingredients, and are
# expected to change when outcomes are rewritten.
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
    "appearances",
    "full_appearances",
}
# ``xP`` is deliberately not an outcome: FPL publishes it before the deadline, so the
# current gameweek's value is legitimate pre-match information (``fpl_xp_now``).

CORRUPTIBLE = sorted(OUTCOME_COLUMNS - {"played", "started", "appearances", "full_appearances"})


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


@pytest.fixture(scope="module")
def honest(silver, players) -> pd.DataFrame:
    return _build(silver, players)


def _build(frame: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    return build_features(SEASONS, write=False, silver=frame, players=players)


def _corrupt(silver: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    corrupted = silver.copy()
    assert mask.any(), "corruption mask selects nothing"
    for column in CORRUPTIBLE:
        if column in corrupted.columns:
            corrupted.loc[mask, column] = corrupted.loc[mask, column] * 1000 + 999
    return corrupted


def _leaking_features(honest: pd.DataFrame, tampered: pd.DataFrame, through_gw: int) -> list[str]:
    key = ["code", "season", "GW"]
    features = [c for c in feature_columns(honest) if c not in OUTCOME_COLUMNS]

    a = honest[honest["GW"] <= through_gw].sort_values(key).reset_index(drop=True)
    b = tampered[tampered["GW"] <= through_gw].sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) > 0

    leaking = []
    for column in features:
        left = a[column].to_numpy(dtype="float64")
        right = b[column].to_numpy(dtype="float64")
        if not np.allclose(left, right, equal_nan=True, rtol=1e-9, atol=1e-9):
            leaking.append(column)
    return leaking


def test_rewriting_the_future_does_not_change_the_past(silver, players, honest):
    """Corrupt every gameweek after the cutoff; everything up to it must be identical."""
    tampered = _build(_corrupt(silver, silver["GW"] > CUTOFF_GW), players)
    leaking = _leaking_features(honest, tampered, CUTOFF_GW)
    assert not leaking, (
        f"{len(leaking)} feature(s) depend on gameweeks after the one they describe: {leaking[:12]}"
    )


def test_rewriting_the_present_does_not_change_its_own_features(silver, players, honest):
    """Corrupt the cutoff gameweek too. Its features describe it, so they must be
    built only from what came before it -- including other players' results."""
    tampered = _build(_corrupt(silver, silver["GW"] >= CUTOFF_GW), players)
    leaking = _leaking_features(honest, tampered, CUTOFF_GW)
    assert not leaking, (
        f"{len(leaking)} feature(s) absorb results from the gameweek they predict: {leaking[:12]}"
    )


def test_first_gameweek_has_no_history(honest):
    """A player's debut cannot carry form. If it does, the shift is missing."""
    debut = honest.sort_values(["code", "kickoff_time"]).groupby("code").head(1)
    assert (debut["games_played"] == 0).all()
    assert debut["total_points_mean1"].isna().all(), "debut rows must have no prior form"


def test_lagged_form_equals_previous_gameweek_points(honest):
    """``total_points_mean1`` must be exactly the previous gameweek's points."""
    ordered = honest.sort_values(["code", "kickoff_time"])
    expected = ordered.groupby("code")["total_points"].shift(1)
    actual = ordered["total_points_mean1"]
    both = expected.notna() & actual.notna()
    assert both.sum() > 1000
    np.testing.assert_allclose(actual[both], expected[both], rtol=1e-9)


def test_double_gameweek_counts_are_per_fixture(honest):
    """Two fixtures with 90 minutes each must count as two full appearances."""
    doubles = honest[honest["fixtures_this_gw"] == 2]
    assert len(doubles) > 0, "fixture season should contain double gameweeks"
    assert doubles["appearances"].max() == 2
    assert doubles["full_appearances"].max() == 2
    assert (honest.loc[honest["fixtures_this_gw"] == 1, "appearances"] <= 1).all()


def test_feature_columns_exclude_outcomes(honest):
    exposed = set(feature_columns(honest)) & OUTCOME_COLUMNS
    assert not exposed, f"outcome columns exposed as features: {sorted(exposed)}"

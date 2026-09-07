"""Assemble the gold feature table.

Contract: every column describing gameweek *t* is computable from data timestamped
before gameweek *t*'s deadline. ``tests/test_no_leakage.py`` enforces it by rewriting
the future and asserting the past does not move.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fpl.config import GOLD, TRAIN_SEASONS
from fpl.data.archive import load_players
from fpl.data.silver import load_silver
from fpl.features.aggregate import aggregate_to_gameweek, attach_opponent, build_team_id_map
from fpl.features.windows import (
    WINDOWS,
    build_team_strength,
    days_since_last_match,
    expanding_league_mean,
    expanding_position_prior,
    lagged_expanding_sum,
    lagged_rolling_mean,
    shrunk_per90,
)

log = logging.getLogger(__name__)

GOLD_PATH = GOLD / "features.parquet"

TARGET = "total_points"

# Form: recent per-gameweek averages.
FORM_COLUMNS = [
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "clean_sheets",
    "saves",
    "goals_conceded",
]

# Rates: shrunk per-90 numbers built from career-to-date totals.
RATE_COLUMNS = [
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "bps",
    "saves",
    "clean_sheets",
    "expected_goals_conceded",
]


def _add_rate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Shrunk per-90 rates, and the raw totals they came from."""
    totals = lagged_expanding_sum(frame, [*RATE_COLUMNS, "minutes"])
    frame = pd.concat([frame, totals], axis=1)

    minutes_todate = frame["minutes_todate"].fillna(0.0)
    for column in RATE_COLUMNS:
        prior = expanding_position_prior(frame, column)
        frame[f"{column}_p90"] = shrunk_per90(
            frame[f"{column}_todate"].fillna(0.0), minutes_todate, prior
        )
    return frame


def _add_opponent_features(frame: pd.DataFrame, strength: pd.DataFrame) -> pd.DataFrame:
    """Join each row to its own club's form and its opponent's."""
    own = strength.rename(columns={c: f"own_{c}" for c in strength.columns if "_r" in c})
    frame = frame.merge(own, on=["season", "team", "GW"], how="left")

    opponent = strength.rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in strength.columns if "_r" in c}}
    )
    frame = frame.merge(opponent, on=["season", "opponent", "GW"], how="left")

    # Fixture difficulty: how leaky the opponent has been relative to the league so
    # far. The normaliser must itself be a to-date average -- dividing by the whole
    # season's mean would put the future into every row.
    league_mean = expanding_league_mean(frame, "opp_team_xgc_r10")
    frame["fixture_ease"] = frame["opp_team_xgc_r10"] / league_mean.replace(0.0, np.nan)
    return frame


def build_features(
    seasons: tuple[str, ...] = TRAIN_SEASONS,
    *,
    write: bool = True,
    silver: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Produce the gold table: one row per player-gameweek, features plus target.

    ``silver`` and ``players`` are injectable so the leakage test can feed in a
    deliberately corrupted future and check that the past is unaffected.
    """
    silver = load_silver() if silver is None else silver
    silver = silver[silver["season"].isin(seasons)].copy()
    if silver.empty:
        raise ValueError(f"no silver rows for seasons {seasons}")

    players = load_players(seasons) if players is None else players
    team_map = build_team_id_map(players, silver)

    frame = aggregate_to_gameweek(silver)
    frame = attach_opponent(frame, team_map)
    frame = frame.sort_values(["code", "kickoff_time"]).reset_index(drop=True)

    # --- player history ----------------------------------------------------
    form = lagged_rolling_mean(frame, FORM_COLUMNS, WINDOWS)
    frame = pd.concat([frame, form], axis=1)
    frame = _add_rate_features(frame)

    frame["games_played"] = frame.groupby("code").cumcount()
    frame["days_rest"] = days_since_last_match(frame)
    frame["minutes_share_r5"] = (frame["minutes_mean5"] / 90.0).clip(0, 1)
    frame["started_share_r5"] = frame["starts_mean5"].clip(0, 1)

    # --- team and opponent -------------------------------------------------
    strength = build_team_strength(frame)
    frame = _add_opponent_features(frame, strength)

    # --- context -----------------------------------------------------------
    frame["is_home"] = frame["was_home"].astype(int)
    frame["price"] = frame["value"] / 10.0
    frame["ownership"] = frame["selected"]
    for position in ("GK", "DEF", "MID", "FWD"):
        frame[f"is_{position.lower()}"] = (frame["position"] == position).astype(int)

    # --- targets -----------------------------------------------------------
    frame["played"] = (frame["minutes"] > 0).astype(int)
    frame["started"] = (frame["minutes"] >= 60).astype(int)

    frame = frame.replace([np.inf, -np.inf], np.nan)

    log.info("gold table: %d rows x %d columns", len(frame), frame.shape[1])
    if write:
        GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(GOLD_PATH, index=False)
        log.info("wrote %s", GOLD_PATH)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The columns a model may train on.

    Anything describing the current gameweek's outcome is excluded by construction:
    if it is not derived from a lagged window, a rate-to-date or fixed context, it
    does not belong here.
    """
    allowed_suffixes = ("_mean1", "_mean3", "_mean5", "_mean10", "_mean38", "_p90", "_todate")
    allowed_exact = {
        "games_played",
        "days_rest",
        "minutes_share_r5",
        "started_share_r5",
        "is_home",
        "price",
        "ownership",
        "fixtures_this_gw",
        "fixture_ease",
        "is_gk",
        "is_def",
        "is_mid",
        "is_fwd",
        "GW",
    }
    columns = [
        c
        for c in frame.columns
        if c.endswith(allowed_suffixes)
        or c in allowed_exact
        or c.startswith(("own_team_", "opp_team_"))
    ]
    return sorted(columns)


def load_features() -> pd.DataFrame:
    """Read the gold table, building it if absent."""
    if not GOLD_PATH.exists():
        log.info("gold table missing, building it")
        return build_features()
    return pd.read_parquet(GOLD_PATH)

"""Collapse per-fixture rows to one row per player per gameweek.

Double gameweeks are real: a few thousand player-gameweeks per season contain two
fixtures. Taking one row would throw away half a player's return; the target for
gameweek *t* must be the sum across every fixture played in it, and so must the
per-fixture events the component model predicts (appearances, clean sheets).
"""

from __future__ import annotations

import logging

import pandas as pd

from fpl.config import DC_THRESHOLD, HAUL_POINTS

log = logging.getLogger(__name__)

# Summed across fixtures within a gameweek.
SUM_COLUMNS = [
    # Bookmaker-implied expectations per fixture (fpl.data.odds); summed so a double
    # gameweek carries two matches' worth, like every other count here.
    "odds_win",
    "odds_draw",
    "odds_lose",
    "odds_xg",
    "odds_xgc",
    "odds_cs",
    "odds_over25",
    # Opponent-adjusted club ratings per fixture (fpl.models.team_strength): the
    # expected-goal and probability terms add up across a double gameweek.
    "ts_xg_for",
    "ts_xg_against",
    "ts_cs",
    "ts_win",
    # FPL publishes its own expected-points figure before each deadline. It is the
    # baseline the backtest scores against, and also a legitimate pre-match input
    # (see ``fpl_xp_now`` in features.build).
    "xP",
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "defensive_contribution",
]

# Taken from the first fixture of the gameweek (they describe the player, not the match).
FIRST_COLUMNS = [
    "full_name", "position", "team", "value", "selected", "element",
    # present only on live upcoming rows
    "web_name", "status", "chance_of_playing", "news", "availability", "fpl_ep_next",
    "penalties_order", "corners_order", "freekicks_order", "transfers_in_event",
    "transfers_out_event", "cost_change_start", "value_season",
    "is_upcoming", "selected_by_percent",
    # the ratings themselves describe the club, not the match
    "ts_att", "ts_def", "ts_opp_att", "ts_opp_def",
]  # fmt: skip


def aggregate_to_gameweek(silver: pd.DataFrame) -> pd.DataFrame:
    """One row per (code, season, GW), summing returns across fixtures."""
    present_sum = [c for c in SUM_COLUMNS if c in silver.columns]
    present_first = [c for c in FIRST_COLUMNS if c in silver.columns]

    frame = silver.sort_values(["code", "season", "GW", "kickoff_time"]).copy()
    # Per-fixture events, so that a double gameweek counts twice where the rules
    # score it twice. Binary flags at gameweek level would cap a two-clean-sheet
    # week at one clean sheet's worth of points.
    frame["appearances"] = (frame["minutes"] > 0).astype(int)
    frame["full_appearances"] = (frame["minutes"] >= 60).astype(int)
    # Defensive-contribution points are awarded per fixture on reaching a positional
    # threshold. Seasons before the rule have no column; they count as never reached.
    if "defensive_contribution" not in frame.columns:
        frame["defensive_contribution"] = 0.0
    threshold = frame["position"].map(DC_THRESHOLD).fillna(10**6)
    frame["dc_hits"] = (frame["defensive_contribution"].fillna(0) >= threshold).astype(int)
    grouped = frame.groupby(["code", "season", "GW"], sort=False)

    aggregated = grouped.agg(
        **{col: (col, "sum") for col in present_sum},
        appearances=("appearances", "sum"),
        full_appearances=("full_appearances", "sum"),
        dc_hits=("dc_hits", "sum"),
        **{col: (col, "first") for col in present_first},
        fixtures_this_gw=("fixture", "nunique"),
        kickoff_time=("kickoff_time", "min"),
        was_home=("was_home", "first"),
        opponent_team=("opponent_team", "first"),
    ).reset_index()

    # The captaincy target: a haul, eight points or more in the gameweek. Expected
    # points say who scores most on average; this says who has the ceiling.
    aggregated["haul"] = (aggregated["total_points"].fillna(0) >= HAUL_POINTS).astype(int)

    doubles = int((aggregated["fixtures_this_gw"] > 1).sum())
    log.info(
        "aggregated to %d player-gameweeks (%d were double gameweeks)",
        len(aggregated),
        doubles,
    )
    return aggregated.sort_values(["code", "kickoff_time"]).reset_index(drop=True)


def build_team_id_map(players: pd.DataFrame, silver: pd.DataFrame) -> pd.DataFrame:
    """Map the season-local numeric team id to a canonical club name.

    ``opponent_team`` in the gameweek data is an integer 1-20 that is only meaningful
    within its season. The player registry carries the same integer, so joining on
    (season, element) recovers the club name.
    """
    registry = players.rename(columns={"id": "element", "team": "team_id"})[
        ["season", "element", "team_id"]
    ]
    joined = (
        silver[["season", "element", "team"]]
        .drop_duplicates()
        .merge(registry, on=["season", "element"], how="inner")
    )
    mapping = (
        joined.groupby(["season", "team_id"])["team"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={"team": "team_name"})
    )

    per_season = mapping.groupby("season").size()
    if (per_season != 20).any():
        raise ValueError(f"expected 20 clubs per season, got {per_season.to_dict()}")
    return mapping


def attach_opponent(frame: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    """Resolve ``opponent_team`` to a club name."""
    merged = frame.merge(
        team_map.rename(columns={"team_id": "opponent_team", "team_name": "opponent"}),
        on=["season", "opponent_team"],
        how="left",
    )
    unresolved = merged["opponent"].isna().sum()
    if unresolved:
        raise ValueError(f"{unresolved} rows have an unresolvable opponent_team id")
    return merged

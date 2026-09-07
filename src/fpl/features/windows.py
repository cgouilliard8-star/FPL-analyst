"""Leak-free rolling aggregates.

Every function here shifts by one gameweek before aggregating. That single ``shift(1)``
is what separates a model from a time machine: without it, the row describing gameweek
*t* would include gameweek *t*'s own result, and the model would appear to predict the
future perfectly while being useless in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP = "code"

# Rolling horizons, in gameweeks. Short windows capture form, long ones capture ability.
WINDOWS: tuple[int, ...] = (1, 3, 5, 10, 38)

# Strength of the shrinkage prior, expressed in 90-minute appearances. A player with
# 5 full matches behind him is weighted equally against the position-average rate.
PRIOR_STRENGTH_90S = 5.0


def _shifted(frame: pd.DataFrame, columns: list[str], group: str = GROUP) -> pd.DataFrame:
    """The player's own history, excluding the current gameweek."""
    shifted = frame.groupby(group, sort=False)[columns].shift(1)
    shifted[group] = frame[group].to_numpy()
    return shifted


def lagged_rolling_mean(
    frame: pd.DataFrame,
    columns: list[str],
    windows: tuple[int, ...] = WINDOWS,
    *,
    group: str = GROUP,
    prefix: str = "",
) -> pd.DataFrame:
    """Rolling means over the previous ``w`` gameweeks, never including the current one."""
    shifted = _shifted(frame, columns, group)
    out = {}
    for window in windows:
        rolled = (
            shifted.groupby(group, sort=False)[columns]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )
        for col in columns:
            out[f"{prefix}{col}_mean{window}"] = rolled[col].to_numpy()
    return pd.DataFrame(out, index=frame.index)


def lagged_expanding_sum(
    frame: pd.DataFrame, columns: list[str], *, group: str = GROUP, prefix: str = ""
) -> pd.DataFrame:
    """Career-to-date totals, excluding the current gameweek."""
    shifted = _shifted(frame, columns, group)
    cumulative = (
        shifted.groupby(group, sort=False)[columns]
        .cumsum()
        .rename(columns={c: f"{prefix}{c}_todate" for c in columns})
    )
    return cumulative


def expanding_position_prior(frame: pd.DataFrame, stat: str, minutes: str = "minutes") -> pd.Series:
    """League-wide per-90 rate for the player's position, using only earlier gameweeks.

    Used as the shrinkage target: a player in gameweek *t* is compared against what his
    position was averaging *before* t.

    The ordering here is load-bearing. Every step — cumulative sum and the forward fill
    that covers the opening gameweeks — happens in kickoff order and is only mapped back
    to the caller's row order at the very end. Forward-filling in the caller's order
    instead would drag values between chronologically unrelated rows, which is a leak
    (and was caught by tests/test_no_leakage.py rather than by reading the code).
    """
    ordered = frame.sort_values("kickoff_time")
    positions = ordered["position"]
    by_position = ordered.groupby(positions, sort=False)

    stat_cum = by_position[stat].transform(lambda s: s.shift(1).cumsum())
    mins_cum = by_position[minutes].transform(lambda s: s.shift(1).cumsum())

    prior = (stat_cum / (mins_cum / 90.0)).replace([np.inf, -np.inf], np.nan)
    prior = prior.groupby(positions, sort=False).ffill().fillna(0.0)
    return prior.reindex(frame.index)


def expanding_league_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    """League average of ``column`` using only gameweeks that have already happened.

    A plain ``frame[column].mean()`` averages the entire season, future included. Used
    as a normaliser it silently leaks into every row.
    """
    ordered = frame.sort_values("kickoff_time")
    values = ordered[column]
    mean_to_date = values.shift(1).expanding(min_periods=1).mean()
    mean_to_date = mean_to_date.ffill().bfill()
    return mean_to_date.reindex(frame.index)


def shrunk_per90(
    stat_todate: pd.Series,
    minutes_todate: pd.Series,
    prior_rate: pd.Series,
    strength: float = PRIOR_STRENGTH_90S,
) -> pd.Series:
    """Per-90 rate pulled toward the position average in proportion to sample size.

    Without this, a substitute who scores in his only 12 minutes shows a goals-per-90
    of 7.5 and tops every ranking forever. The empirical-Bayes form below gives him
    roughly the position average until he has actually played.
    """
    nineties = (minutes_todate / 90.0).clip(lower=0)
    return (stat_todate + strength * prior_rate) / (nineties + strength)


def days_since_last_match(frame: pd.DataFrame, group: str = GROUP) -> pd.Series:
    """Rest between fixtures. Congestion drives rotation, which drives minutes."""
    previous = frame.groupby(group, sort=False)["kickoff_time"].shift(1)
    delta = (frame["kickoff_time"] - previous).dt.total_seconds() / 86400.0
    return delta.clip(upper=60.0)


def build_team_strength(
    gameweeks: pd.DataFrame, windows: tuple[int, ...] = (5, 10)
) -> pd.DataFrame:
    """Rolling attacking and defensive strength per club, lagged.

    Derived from the squad's own returns rather than an external ratings feed, so the
    pipeline stays self-contained. ``fpl.data.elo`` can layer ClubElo on top where the
    network allows it.
    """
    team_gw = (
        gameweeks.groupby(["season", "team", "GW"], as_index=False)
        .agg(
            team_goals=("goals_scored", "sum"),
            team_xg=("expected_goals", "sum"),
            team_conceded=("goals_conceded", "max"),
            team_xgc=("expected_goals_conceded", "max"),
        )
        .sort_values(["season", "team", "GW"])
    )

    metrics = ["team_goals", "team_xg", "team_conceded", "team_xgc"]
    shifted = team_gw.groupby(["season", "team"], sort=False)[metrics].shift(1)
    shifted[["season", "team"]] = team_gw[["season", "team"]].to_numpy()

    for window in windows:
        rolled = (
            shifted.groupby(["season", "team"], sort=False)[metrics]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .sort_index()
        )
        for metric in metrics:
            team_gw[f"{metric}_r{window}"] = rolled[metric].to_numpy()

    return team_gw.drop(columns=metrics)

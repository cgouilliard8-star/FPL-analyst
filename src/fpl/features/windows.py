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
            # Align by index label, not position: groupby.rolling returns rows in
            # group order, which is not the caller's row order.
            .reindex(frame.index)
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


def _gameweek_level(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Totals per (season, GW), with the earliest kickoff, for chronological ordering."""
    return (
        frame.groupby(["season", "GW"], as_index=False)
        .agg(**{c: (c, "sum") for c in columns}, kickoff_time=("kickoff_time", "min"))
        .sort_values("kickoff_time")
    )


def expanding_position_prior(frame: pd.DataFrame, stat: str, minutes: str = "minutes") -> pd.Series:
    """League-wide per-90 rate for the player's position, using only earlier gameweeks.

    Used as the shrinkage target: a player in gameweek *t* is compared against what his
    position was averaging *before* t.

    The accumulation happens at gameweek granularity, not row granularity. A row-level
    ``shift(1)`` in kickoff order looks equivalent but is not: rows from the same match
    share a timestamp, so a player's prior would include team-mates' results from the
    very gameweek being predicted. That is a leak the future-corruption test cannot
    see, because it lives inside the cutoff gameweek rather than after it.
    """
    per_position = []
    for position, rows in frame.groupby("position", sort=False):
        totals = _gameweek_level(rows, [stat, minutes])
        totals[stat] = totals[stat].shift(1).cumsum()
        totals[minutes] = totals[minutes].shift(1).cumsum()
        totals["prior"] = (totals[stat] / (totals[minutes] / 90.0)).replace(
            [np.inf, -np.inf], np.nan
        )
        totals["prior"] = totals["prior"].ffill().fillna(0.0)
        totals["position"] = position
        per_position.append(totals[["season", "GW", "position", "prior"]])

    lookup = pd.concat(per_position, ignore_index=True)
    merged = frame[["season", "GW", "position"]].merge(
        lookup, on=["season", "GW", "position"], how="left"
    )
    return pd.Series(merged["prior"].to_numpy(), index=frame.index).fillna(0.0)


def expanding_league_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    """League average of ``column`` over gameweeks that have already been played.

    A plain ``frame[column].mean()`` averages the entire season, future included. Used
    as a normaliser it silently leaks into every row. Like the position prior, this is
    accumulated per gameweek so the current gameweek never contributes to itself.
    """
    totals = (
        frame.groupby(["season", "GW"], as_index=False)
        .agg(total=(column, "sum"), count=(column, "count"), kickoff_time=("kickoff_time", "min"))
        .sort_values("kickoff_time")
    )
    totals["mean_to_date"] = totals["total"].shift(1).cumsum() / totals["count"].shift(1).cumsum()
    totals["mean_to_date"] = totals["mean_to_date"].ffill().bfill()

    merged = frame[["season", "GW"]].merge(
        totals[["season", "GW", "mean_to_date"]], on=["season", "GW"], how="left"
    )
    return pd.Series(merged["mean_to_date"].to_numpy(), index=frame.index)


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
    gameweeks: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10, 38),
    form_halflife: float = 2.0,
) -> pd.DataFrame:
    """Rolling attacking and defensive strength per club, lagged.

    Derived from the squad's own returns rather than an external ratings feed, so the
    pipeline stays self-contained. The windows run *across* seasons: at gameweek 1 a
    club's form is last season's closing run, not an empty window, and the 38-match
    window is the "big club" prior -- a season of underlying quality that a bad
    fortnight does not erase. A promoted club has no history and is left NaN for the
    model (and filled with the league mean where a rank is needed).

    ``form_halflife`` produces an exponentially weighted version (``_ew``) in which
    the last two or three matches carry most of the weight: current form, for the
    fixture-difficulty tables.
    """
    team_gw = (
        gameweeks.groupby(["season", "team", "GW"], as_index=False)
        .agg(
            kickoff=("kickoff_time", "min"),
            team_goals=("goals_scored", "sum"),
            team_xg=("expected_goals", "sum"),
            team_conceded=("goals_conceded", "max"),
            team_xgc=("expected_goals_conceded", "max"),
        )
        .sort_values(["team", "kickoff", "season", "GW"])
        .reset_index(drop=True)
    )

    metrics = ["team_goals", "team_xg", "team_conceded", "team_xgc"]
    shifted = team_gw.groupby("team", sort=False)[metrics].shift(1)
    shifted["team"] = team_gw["team"].to_numpy()
    grouped = shifted.groupby("team", sort=False)[metrics]

    for window in windows:
        rolled = (
            grouped.rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .reindex(team_gw.index)
        )
        for metric in metrics:
            team_gw[f"{metric}_r{window}"] = rolled[metric].to_numpy()

    weighted = grouped.transform(lambda s: s.ewm(halflife=form_halflife, min_periods=1).mean())
    for metric in metrics:
        team_gw[f"{metric}_ew"] = weighted[metric].to_numpy()

    return (
        team_gw.drop(columns=[*metrics, "kickoff"])
        .sort_values(["season", "team", "GW"])
        .reset_index(drop=True)
    )

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
from fpl.data.odds import ODDS_FEATURES, attach_odds, load_odds
from fpl.data.schedule import congestion_features, load_schedule
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
from fpl.models.team_strength import FEATURES as TEAM_STRENGTH_FEATURES
from fpl.models.team_strength import attach_team_strength

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
    keys = {"season", "team", "GW"}
    metrics = [c for c in strength.columns if c not in keys]
    own = strength.rename(columns={c: f"own_{c}" for c in metrics})
    frame = frame.merge(own, on=["season", "team", "GW"], how="left")

    opponent = strength.rename(columns={"team": "opponent", **{c: f"opp_{c}" for c in metrics}})
    frame = frame.merge(opponent, on=["season", "opponent", "GW"], how="left")

    # Fixture difficulty: how leaky the opponent has been relative to the league so
    # far. The normaliser must itself be a to-date average -- dividing by the whole
    # season's mean would put the future into every row.
    league_mean = expanding_league_mean(frame, "opp_team_xgc_r10")
    frame["fixture_ease"] = frame["opp_team_xgc_r10"] / league_mean.replace(0.0, np.nan)
    return frame


CONGESTION = ("other_games_7d", "euro_midweek", "other_game_next_4d", "days_since_any_match")

# Set-piece duties as the player registry records them: archive column name ->
# the name used here. The live snapshot registry is normalised to the same names.
DUTY_COLUMNS = {
    "penalties_order": "pen_order",
    "corners_and_indirect_freekicks_order": "corner_order",
    "direct_freekicks_order": "freekick_order",
}
DUTY_FEATURES = ("pen_taker", "pen_order_inv", "corner_taker", "freekick_taker", "set_piece_share")
MINUTES_PATTERN_FEATURES = (
    "start_streak", "games_since_start", "cameo_share_r5", "minutes_std5",
    "pos_minutes_rank", "pos_minutes_share", "pos_regulars",
)  # fmt: skip


def _registry_duties(players: pd.DataFrame) -> pd.DataFrame:
    """Per (season, code): who takes the penalties, corners and free kicks.

    The archive's registry is the end-of-season state, so for past seasons this is
    known slightly "too early" -- a taker who inherited the duty in March is flagged
    from August. Duties change rarely enough that the signal is worth that blemish;
    for the live season the registry is today's, so it is exactly point-in-time.
    """
    frame = players.rename(columns=DUTY_COLUMNS).copy()
    for column in DUTY_COLUMNS.values():
        if column not in frame.columns:
            frame[column] = np.nan
    duties = frame[["season", "code", *DUTY_COLUMNS.values()]].drop_duplicates(["season", "code"])
    pen = pd.to_numeric(duties["pen_order"], errors="coerce")
    corner = pd.to_numeric(duties["corner_order"], errors="coerce")
    fk = pd.to_numeric(duties["freekick_order"], errors="coerce")
    out = duties[["season", "code"]].copy()
    out["pen_taker"] = (pen == 1).astype(int)
    out["pen_order_inv"] = (1.0 / pen).fillna(0.0)
    out["corner_taker"] = (corner <= 2).astype(int)
    out["freekick_taker"] = (fk == 1).astype(int)
    out["set_piece_share"] = (
        (1.0 / pen).fillna(0.0) + (1.0 / corner).fillna(0.0) + (1.0 / fk).fillna(0.0)
    )
    return out


def _add_duties(frame: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    duties = _registry_duties(players)
    frame = frame.merge(duties, on=["season", "code"], how="left")
    for column in DUTY_FEATURES:
        frame[column] = frame[column].fillna(0.0)
    return frame


def _add_minutes_pattern(frame: pd.DataFrame) -> pd.DataFrame:
    """How a player has been used, beyond his average minutes.

    A 60-minute average can be a regular starter withdrawn late or a rotation
    option who starts half the time; the two carry different risk. All windows are
    lagged (the current gameweek is excluded) like every other form feature.
    """
    code = frame["code"]
    started = pd.Series((frame["starts"].fillna(0) > 0).astype(int).to_numpy(), index=frame.index)
    played = pd.Series((frame["minutes"].fillna(0) > 0).astype(int).to_numpy(), index=frame.index)
    cameo = ((played == 1) & (started == 0)).astype(int)

    # Consecutive starts coming into this gameweek, and gameweeks since the last one.
    prev = started.groupby(code).shift(1).fillna(0).astype(int)
    blocks = (prev == 0).groupby(code).cumsum()
    frame["start_streak"] = prev.groupby([code, blocks]).cumsum().clip(upper=38)
    since = (prev == 1).groupby(code).cumsum()
    frame["games_since_start"] = (
        (prev == 0).astype(int).groupby([code, since]).cumsum().clip(upper=20)
    )

    def _lagged_roll(series: pd.Series, window: int, fn: str, min_periods: int = 1) -> pd.Series:
        lagged = series.groupby(code).shift(1)
        rolled = getattr(lagged.groupby(code).rolling(window, min_periods=min_periods), fn)()
        return rolled.reset_index(level=0, drop=True).reindex(frame.index)

    frame["cameo_share_r5"] = _lagged_roll(cameo, 5, "mean")
    frame["minutes_std5"] = _lagged_roll(frame["minutes"], 5, "std", min_periods=2)

    # Competition for the shirt: where the player's recent minutes rank among his
    # club's players in the same position, and how many of them are regular starters.
    keys = [frame[k] for k in ("season", "GW", "team", "position")]
    share = frame["minutes_mean5"].fillna(0.0)
    frame["pos_minutes_rank"] = share.groupby(keys).rank(ascending=False, method="min")
    total = share.groupby(keys).transform("sum")
    frame["pos_minutes_share"] = (share / total.replace(0.0, np.nan)).fillna(0.0)
    regular = (frame["started_share_r5"].fillna(0.0) >= 0.5).astype(int)
    frame["pos_regulars"] = regular.groupby(keys).transform("sum")
    return frame


def attach_congestion(frame: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Cup and European matches around each fixture, for the club and its opponent.

    Computed once per (club, kick-off) and joined, not per player row.
    """
    keys = frame[["team", "kickoff_time"]].drop_duplicates().reset_index(drop=True)
    feats = pd.concat([keys, congestion_features(keys, schedule)], axis=1)
    frame = frame.merge(feats, on=["team", "kickoff_time"], how="left")
    opp = feats.rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in ("other_games_7d", "euro_midweek")}}
    )[["opponent", "kickoff_time", "opp_other_games_7d", "opp_euro_midweek"]]
    frame = frame.merge(opp, on=["opponent", "kickoff_time"], how="left")
    for c in (
        "other_games_7d",
        "euro_midweek",
        "other_game_next_4d",
        "opp_other_games_7d",
        "opp_euro_midweek",
    ):
        frame[c] = frame[c].fillna(0).astype(int)
    # Rest counting every competition: the league gap, or the cup gap if shorter.
    frame["days_rest_all"] = frame[["days_rest", "days_since_any_match"]].min(axis=1)
    return frame


def build_features(
    seasons: tuple[str, ...] = TRAIN_SEASONS,
    *,
    write: bool = True,
    silver: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
    schedule: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Produce the gold table: one row per player-gameweek, features plus target.

    ``silver`` and ``players`` are injectable so the leakage test can feed in a
    deliberately corrupted future and check that the past is unaffected.
    ``schedule`` is the clubs' non-league fixture list (cups, Europe); when absent
    the congestion features are zero.
    """
    silver = load_silver() if silver is None else silver
    schedule = load_schedule(seasons) if schedule is None else schedule
    silver = silver[silver["season"].isin(seasons)].copy()
    if silver.empty:
        raise ValueError(f"no silver rows for seasons {seasons}")

    players = load_players(seasons) if players is None else players
    team_map = build_team_id_map(players, silver)

    # Bookmaker odds join per fixture (a double gameweek has two markets), so they
    # go on before aggregation; the opponent name is needed for the key.
    odds = load_odds(seasons) if odds is None else odds
    silver = attach_odds(attach_opponent(silver, team_map), odds)
    # Opponent-adjusted club ratings, refitted before each gameweek on the matches
    # played so far (fpl.models.team_strength); per fixture, like the odds.
    silver = attach_team_strength(silver).drop(columns=["opponent"])

    frame = aggregate_to_gameweek(silver)
    frame = attach_opponent(frame, team_map)
    frame = frame.sort_values(["code", "kickoff_time"]).reset_index(drop=True)

    # --- player history ----------------------------------------------------
    form = lagged_rolling_mean(frame, FORM_COLUMNS, WINDOWS)
    frame = pd.concat([frame, form], axis=1)
    frame = _add_rate_features(frame)

    frame["games_played"] = frame.groupby("code").cumcount()
    frame["days_rest"] = days_since_last_match(frame)
    frame = attach_congestion(frame, schedule)
    frame["minutes_share_r5"] = (frame["minutes_mean5"] / 90.0).clip(0, 1)
    frame["started_share_r5"] = frame["starts_mean5"].clip(0, 1)
    frame = _add_minutes_pattern(frame)
    frame = _add_duties(frame, players)

    # --- team and opponent -------------------------------------------------
    strength = build_team_strength(frame)
    frame = _add_opponent_features(frame, strength)

    # --- context -----------------------------------------------------------
    # FPL's own expected points (``xP`` in the archive) is NOT a feature. It looks
    # like pre-match information, but the archive's copy was scraped after each
    # gameweek and FPL's figure folds the gameweek's real points into "form" by
    # then: a haul that "predicted" 49.6 points is a giveaway. Stacking it lifted
    # precision@10 to 0.90, which is how the leak was caught. It stays as the
    # baseline the backtest scores against, with that caveat.
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
        "days_rest_all",
        "other_games_7d",
        "euro_midweek",
        "other_game_next_4d",
        "opp_other_games_7d",
        "opp_euro_midweek",
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
        *DUTY_FEATURES,
        *MINUTES_PATTERN_FEATURES,
    }
    columns = [
        c
        for c in frame.columns
        if c.endswith(allowed_suffixes)
        or c in allowed_exact
        or c.startswith(("own_team_", "opp_team_"))
        or c in ODDS_FEATURES
        or c in TEAM_STRENGTH_FEATURES
    ]
    return sorted(columns)


def load_features() -> pd.DataFrame:
    """Read the gold table, building it if absent."""
    if not GOLD_PATH.exists():
        log.info("gold table missing, building it")
        return build_features()
    return pd.read_parquet(GOLD_PATH)

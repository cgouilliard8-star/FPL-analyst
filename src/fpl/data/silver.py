"""Build the canonical (silver) table: one row per player per gameweek, one identity.

Bronze is whatever the sources gave us. Silver is the version we are willing to
build features on: every row carries a stable ``code``, a canonical club name, and a
position, and anything that could not be resolved has already caused a loud failure.
"""

from __future__ import annotations

import logging

import pandas as pd

from fpl.config import SILVER, TRAIN_SEASONS
from fpl.data.archive import load_players, load_seasons
from fpl.entity.resolve import (
    attach_player_code,
    build_player_index,
    canonical_team,
    drop_managers,
    load_team_aliases,
)

log = logging.getLogger(__name__)

SILVER_PATH = SILVER / "gameweeks.parquet"


def _canonicalise_teams(frame: pd.DataFrame) -> pd.DataFrame:
    """Map club spellings to FPL short names, failing on anything unrecognised."""
    aliases = load_team_aliases()
    frame = frame.copy()
    frame["team"] = frame["team"].map(lambda name: canonical_team(name, aliases))

    unknown = frame["team"].isna()
    if unknown.any():
        raise ValueError(
            f"{unknown.sum()} rows have an unrecognised club. "
            "Add the spelling to src/fpl/entity/mappings/teams.csv."
        )
    return frame


def build_silver(seasons: tuple[str, ...] = TRAIN_SEASONS, *, write: bool = True) -> pd.DataFrame:
    """Resolve identity across the archive and produce the canonical table."""
    gameweeks = load_seasons(seasons)
    index = build_player_index(load_players(seasons))

    resolved = attach_player_code(gameweeks, index)
    resolved = drop_managers(resolved)
    resolved = _canonicalise_teams(resolved)

    # Sorting by identity then time is what every rolling feature will assume.
    resolved = resolved.sort_values(["code", "kickoff_time"]).reset_index(drop=True)

    # The archive occasionally repeats a player-fixture row. Keeping both would count
    # a return twice, so the first occurrence wins and the rest are dropped, loudly.
    key = ["code", "season", "GW", "fixture"]
    duplicates = int(resolved.duplicated(subset=key).sum())
    if duplicates:
        log.warning("dropping %d duplicate (code, season, GW, fixture) rows", duplicates)
        resolved = resolved.drop_duplicates(subset=key, keep="first").reset_index(drop=True)

    if write:
        SILVER_PATH.parent.mkdir(parents=True, exist_ok=True)
        resolved.to_parquet(SILVER_PATH, index=False)
        log.info("wrote %s (%d rows)", SILVER_PATH, len(resolved))

    return resolved


def load_silver() -> pd.DataFrame:
    """Read the canonical table, building it first if it does not exist."""
    if not SILVER_PATH.exists():
        log.info("silver table missing, building it")
        return build_silver()
    return pd.read_parquet(SILVER_PATH)

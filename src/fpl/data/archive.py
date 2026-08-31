"""Historical FPL data from the vaastav/Fantasy-Premier-League archive.

The archive stopped weekly updates after 2024-25 (it now refreshes ~3x/year), so it
is used strictly to bootstrap the training set. In-season data comes from our own
deadline-stamped collector in ``fpl.data.fpl_api``.
"""
from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import requests

from fpl.config import ARCHIVE_BASE, BRONZE, TRAIN_SEASONS

log = logging.getLogger(__name__)

# Columns we require from every season. If the archive changes shape, fail loudly
# here rather than silently producing a half-empty feature table three steps later.
REQUIRED_COLUMNS = frozenset(
    {
        "name", "position", "team", "GW", "minutes", "total_points",
        "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves",
        "bonus", "bps", "yellow_cards", "red_cards", "own_goals",
        "penalties_missed", "penalties_saved", "influence", "creativity",
        "threat", "ict_index", "was_home", "opponent_team", "value",
        "selected", "kickoff_time",
    }
)

TIMEOUT = 60


def _season_url(season: str) -> str:
    return f"{ARCHIVE_BASE}/{season}/gws/merged_gw.csv"


def fetch_season(season: str, *, session: requests.Session | None = None) -> pd.DataFrame:
    """Download one season of merged gameweek data.

    Raises:
        requests.HTTPError: the season is not published.
        ValueError: the file is missing columns we depend on.
    """
    url = _season_url(season)
    log.info("fetching %s", url)
    getter = session.get if session else requests.get
    response = getter(url, timeout=TIMEOUT)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"{season}: archive is missing expected columns {sorted(missing)}. "
            "The upstream schema has changed and the loader needs updating."
        )

    frame["season"] = season
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    return frame


def load_seasons(
    seasons: tuple[str, ...] = TRAIN_SEASONS, *, refresh: bool = False
) -> pd.DataFrame:
    """Load several seasons, caching each to bronze as parquet.

    Cached locally because the archive is static history — re-downloading it on every
    run is slow and rude. Pass ``refresh=True`` to force a re-fetch.
    """
    frames = []
    for season in seasons:
        cache = BRONZE / f"archive_{season}.parquet"
        if cache.exists() and not refresh:
            log.info("using cached %s", cache.name)
            frames.append(pd.read_parquet(cache))
            continue
        frame = fetch_season(season)
        frame.to_parquet(cache, index=False)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    log.info("loaded %d rows across %d seasons", len(combined), len(seasons))
    return combined

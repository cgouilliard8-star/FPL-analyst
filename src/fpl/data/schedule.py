"""Matches Premier League clubs play outside the league: Champions League, Europa
League, Conference League, FA Cup and League Cup.

FPL only knows about league fixtures, but a Wednesday in Munich is why a full-back is
benched on Saturday. The schedule is public and known weeks ahead, so it is a
legitimate feature at any deadline: nothing here is learned after the fact.

Sources (all free, no key):

* fixturedownload.com -- JSON feeds for the three UEFA competitions, current and
  past seasons.
* TheSportsDB -- the two domestic cups (free key ``123``; a 15-requests-a-month cap,
  so the files are refreshed weekly, not daily).

Fetched files are written to ``data/external/schedule_<season>.csv`` and committed,
so a refresh that cannot reach a source keeps the last good copy.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fpl.config import CURRENT_SEASON, PROJECT_ROOT, TRAIN_SEASONS
from fpl.entity.resolve import canonical_team, load_team_aliases

log = logging.getLogger(__name__)

EXTERNAL = PROJECT_ROOT / "data" / "external"
TIMEOUT = 30

UEFA = {
    "champions-league": "UCL",
    "europa-league": "UEL",
    "conference-league": "UECL",
}
FIXTUREDOWNLOAD = "https://fixturedownload.com/feed/json/{slug}-{year}"

SPORTSDB = "https://www.thesportsdb.com/api/v1/json/123/eventsseason.php?id={league}&s={season}"
DOMESTIC = {4482: "FAC", 4570: "EFL"}  # FA Cup, League Cup
DOMESTIC_MAX_AGE_DAYS = 7

EUROPEAN = {"UCL", "UEL", "UECL"}
COLUMNS = ["season", "competition", "kickoff_time", "home", "away"]


def _start_year(season: str) -> int:
    return int(season.split("-")[0])


def schedule_path(season: str) -> Path:
    return EXTERNAL / f"schedule_{season}.csv"


def _team(name: str, aliases: dict[str, str]) -> str | None:
    return canonical_team(name, aliases)


def parse_fixturedownload(records: list[dict], season: str, competition: str) -> pd.DataFrame:
    """Rows for the matches involving at least one Premier League club."""
    aliases = load_team_aliases()
    rows = []
    for r in records:
        home, away = _team(r.get("HomeTeam", ""), aliases), _team(r.get("AwayTeam", ""), aliases)
        if home is None and away is None:
            continue
        when = pd.to_datetime(r.get("DateUtc"), utc=True, errors="coerce")
        if pd.isna(when):
            continue
        rows.append(
            {
                "season": season,
                "competition": competition,
                "kickoff_time": when,
                "home": home or r.get("HomeTeam"),
                "away": away or r.get("AwayTeam"),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_sportsdb(payload: dict, season: str, competition: str) -> pd.DataFrame:
    aliases = load_team_aliases()
    rows = []
    for e in payload.get("events") or []:
        home, away = (
            _team(e.get("strHomeTeam", ""), aliases),
            _team(e.get("strAwayTeam", ""), aliases),
        )
        if home is None and away is None:
            continue
        stamp = e.get("strTimestamp") or f"{e.get('dateEvent')} {e.get('strTime') or '15:00:00'}"
        when = pd.to_datetime(stamp, utc=True, errors="coerce")
        if pd.isna(when):
            continue
        rows.append(
            {
                "season": season,
                "competition": competition,
                "kickoff_time": when,
                "home": home or e.get("strHomeTeam"),
                "away": away or e.get("strAwayTeam"),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def _get_json(session: requests.Session, url: str, retries: int = 3):
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 429 and attempt < retries:
                raise requests.HTTPError("429", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries:
                raise
            log.warning("%s failed (%s); retry in %.0fs", url, e, delay)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def fetch_schedule(
    season: str = CURRENT_SEASON,
    *,
    session: requests.Session | None = None,
    domestic: bool = True,
) -> pd.DataFrame:
    """Fetch every competition's fixtures for ``season`` and merge with what is on
    disk, so a source that is down today does not erase yesterday's rows."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "fpl-analyst/0.1 (+https://github.com)")
    year = _start_year(season)
    existing = load_schedule((season,))
    parts: list[pd.DataFrame] = []
    fetched: set[str] = set()

    for slug, code in UEFA.items():
        try:
            records = _get_json(session, FIXTUREDOWNLOAD.format(slug=slug, year=year))
            frame = parse_fixturedownload(records, season, code)
            parts.append(frame)
            fetched.add(code)
            log.info("%s %s: %d matches involving PL clubs", code, season, len(frame))
        except Exception as e:  # noqa: BLE001 - one source down must not stop the rest
            log.warning("%s %s: fetch failed (%s); keeping the stored rows", code, season, e)

    if domestic and _domestic_stale(existing):
        for league, code in DOMESTIC.items():
            try:
                payload = _get_json(
                    session, SPORTSDB.format(league=league, season=f"{year}-{year + 1}")
                )
                frame = parse_sportsdb(payload, season, code)
                parts.append(frame)
                fetched.add(code)
                log.info("%s %s: %d matches involving PL clubs", code, season, len(frame))
            except Exception as e:  # noqa: BLE001
                log.warning("%s %s: fetch failed (%s); keeping the stored rows", code, season, e)

    kept = existing[~existing["competition"].isin(fetched)] if not existing.empty else existing
    merged = pd.concat([kept, *parts], ignore_index=True) if parts else existing
    if merged.empty:
        return merged
    merged = merged.drop_duplicates(["competition", "kickoff_time", "home", "away"])
    merged = merged.sort_values("kickoff_time").reset_index(drop=True)
    save_schedule(merged, season)
    return merged


def _domestic_stale(existing: pd.DataFrame) -> bool:
    path = None
    if not existing.empty:
        season = existing["season"].iloc[0]
        path = schedule_path(season)
    if path is None or not path.exists():
        return True
    if not (existing["competition"].isin(DOMESTIC.values())).any():
        return True
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return age > timedelta(days=DOMESTIC_MAX_AGE_DAYS)


def save_schedule(frame: pd.DataFrame, season: str) -> Path:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    path = schedule_path(season)
    out = frame.copy()
    out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    out[COLUMNS].to_csv(path, index=False)
    return path


def load_schedule(seasons: tuple[str, ...] = (*TRAIN_SEASONS, CURRENT_SEASON)) -> pd.DataFrame:
    """Stored non-league matches for the seasons given; empty (but well-formed)
    when none have been fetched, so features degrade to zero rather than crash."""
    parts = []
    for season in seasons:
        path = schedule_path(season)
        if path.exists():
            frame = pd.read_csv(path)
            frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True)
            parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=COLUMNS).astype({"kickoff_time": "datetime64[ns, UTC]"})
    return pd.concat(parts, ignore_index=True)[COLUMNS]


def club_matches(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per club per match: ``team, kickoff_time, competition, is_european``."""
    if schedule.empty:
        return pd.DataFrame(columns=["team", "kickoff_time", "competition", "is_european"])
    home = schedule.rename(columns={"home": "team"})[["team", "kickoff_time", "competition"]]
    away = schedule.rename(columns={"away": "team"})[["team", "kickoff_time", "competition"]]
    both = pd.concat([home, away], ignore_index=True)
    both["is_european"] = both["competition"].isin(EUROPEAN)
    return both.sort_values(["team", "kickoff_time"]).reset_index(drop=True)


BEFORE_DAYS = 7
AFTER_DAYS = 4
MIDWEEK_DAYS = 4


def congestion_features(fixtures: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """For each ``(team, kickoff_time)`` in ``fixtures``, what the club played around it.

    Returns a frame aligned to ``fixtures``'s index with:

    * ``other_games_7d`` -- non-league matches in the seven days before kick-off.
    * ``euro_midweek`` -- 1 if a European tie fell within four days before kick-off.
    * ``other_game_next_4d`` -- 1 if a non-league match follows within four days
      (managers rest players ahead of a big tie as well as after one).
    * ``days_since_any_match`` -- days since the club's last match in any
      competition, capped at 60; ``NaN`` when no other match is known.
    """
    n = len(fixtures)
    games7 = np.zeros(n, dtype=int)
    euro_mid = np.zeros(n, dtype=int)
    next4 = np.zeros(n, dtype=int)
    since = np.full(n, np.nan)
    matches = club_matches(schedule)
    positions = {label: i for i, label in enumerate(fixtures.index)}
    if not matches.empty:
        by_team = {t: g for t, g in matches.groupby("team")}
        kick = pd.to_datetime(fixtures["kickoff_time"], utc=True)
        day = pd.Timedelta(days=1).to_timedelta64()
        for team, idx in fixtures.groupby("team").groups.items():
            games = by_team.get(team)
            if games is None:
                continue
            times = pd.to_datetime(games["kickoff_time"], utc=True).dt.tz_convert(None).to_numpy()
            euro = games["is_european"].to_numpy()
            for label in idx:
                k = kick.loc[label]
                if pd.isna(k):
                    continue
                k64 = k.tz_convert(None).to_datetime64()
                i = positions[label]
                before = (times < k64) & (times >= k64 - BEFORE_DAYS * day)
                mid = (times < k64) & (times >= k64 - MIDWEEK_DAYS * day)
                after = (times > k64) & (times <= k64 + AFTER_DAYS * day)
                games7[i] = int(before.sum())
                euro_mid[i] = int((mid & euro).any())
                next4[i] = int(after.any())
                earlier = times[times < k64]
                if len(earlier):
                    since[i] = min(60.0, float((k64 - earlier.max()) / day))
    return pd.DataFrame(
        {
            "other_games_7d": games7,
            "euro_midweek": euro_mid,
            "other_game_next_4d": next4,
            "days_since_any_match": since,
        },
        index=fixtures.index,
    )

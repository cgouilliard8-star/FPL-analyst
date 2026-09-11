"""Per-match player statistics from the FPL-Core-Insights dataset.

FPL's own feed says what a player *scored*; it does not say how many shots he had,
how often he touched the ball in the box, how many chances he created or how many
tackles, interceptions, blocks and clearances he made. Those are the actions that
turn into goals, assists and defensive-contribution points, and they are far less
noisy than the points themselves. olbauday/FPL-Core-Insights publishes them for every
Premier League match from 2024-25 on, keyed by the official FPL ids and refreshed
twice a day (https://github.com/olbauday/FPL-Core-Insights -- data used with thanks,
as its README invites).

The repository is fetched with a shallow, sparse git clone and boiled down to one CSV
per season in ``data/external/core_insights_<season>.csv``: one row per player per league match,
with the handful of columns this project uses. ``load_core_stats`` sums those to the
player-gameweek grain the feature table works at. Seasons the dataset does not cover
come back as NaN, which LightGBM treats as "unknown" rather than zero.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fpl.config import PROJECT_ROOT, TRAIN_SEASONS

log = logging.getLogger(__name__)

REPO = "https://github.com/olbauday/FPL-Core-Insights.git"
# Committed like the odds files (data/external is the one data folder in git), so a
# refresh that cannot reach GitHub keeps the last copy.
CORE_DIR = PROJECT_ROOT / "data" / "external"

# Per-match columns kept, and the feature name each becomes. All are per-match
# counts, so a double gameweek adds up like every other counting stat.
KEEP = {
    "total_shots": "ci_shots",
    "shots_on_target": "ci_sot",
    "touches_opposition_box": "ci_box_touches",
    "chances_created": "ci_chances",
    "big_chances_missed": "ci_bcm",
    "recoveries": "ci_recoveries",
    "goals_prevented": "ci_goals_prevented",
    "xgot_faced": "ci_xgot_faced",
}
DEFENSIVE = ("tackles_won", "interceptions", "blocks", "clearances")
CORE_COLUMNS = (
    *KEEP.values(),
    "ci_def_actions",  # tackles + interceptions + blocks + clearances (FPL's "CBIT")
    "ci_started",  # on the pitch from the first minute
    "ci_matches",
)
ROW_COLUMNS = ["season", "code", "GW", "match_id", "minutes", *CORE_COLUMNS[:-1]]


def _season_dir(season: str) -> str:
    """'2024-25' -> '2024-2025', the dataset's folder name."""
    start = season[:4]
    return f"{start}-{int(start) + 1}"


def _read_gameweek_files(root: Path, name: str) -> pd.DataFrame:
    """Every ``GW*/<name>.csv`` under a season folder, whichever layout the season uses."""
    files = [
        p
        for p in root.rglob(f"{name}.csv")
        if p.parent.name.startswith("GW") and "By Tournament" not in p.parts
    ]
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in sorted(files)]
    return pd.concat(frames, ignore_index=True)


def _season_rows(root: Path, season: str) -> pd.DataFrame:
    """One row per player per league match for a season folder of the dataset."""
    matches = _read_gameweek_files(root, "matches")
    stats = _read_gameweek_files(root, "playermatchstats")
    players_file = next(iter(root.rglob("players.csv")), None)
    if matches.empty or stats.empty or players_file is None:
        log.warning("core insights: %s has no usable files", root)
        return pd.DataFrame(columns=ROW_COLUMNS)
    if "tournament" in matches.columns:
        matches = matches[matches["tournament"].fillna("prem").eq("prem")]
    matches = matches.drop_duplicates("match_id")
    matches = matches[matches["gameweek"].notna() & (matches["gameweek"] > 0)]
    players = pd.read_csv(players_file).drop_duplicates("player_id")
    codes = dict(zip(players["player_id"], players["player_code"], strict=True))

    stats = stats.drop_duplicates(["player_id", "match_id"])
    stats = stats[stats["match_id"].isin(matches["match_id"])]
    gw = dict(zip(matches["match_id"], matches["gameweek"], strict=True))
    out = pd.DataFrame(
        {
            "season": season,
            "code": stats["player_id"].map(codes),
            "GW": stats["match_id"].map(gw).astype(int),
            "match_id": stats["match_id"],
            "minutes": pd.to_numeric(stats["minutes_played"], errors="coerce").fillna(0.0),
        }
    )
    col = lambda name: (  # noqa: E731 - a column the season lacks counts as zero
        pd.to_numeric(stats[name], errors="coerce").fillna(0.0)
        if name in stats.columns
        else pd.Series(0.0, index=stats.index)
    )
    for source, name in KEEP.items():
        out[name] = col(source)
    out["ci_def_actions"] = sum(col(c) for c in DEFENSIVE)
    start = (
        pd.to_numeric(stats["start_min"], errors="coerce")
        if "start_min" in stats.columns
        else pd.Series(np.nan, index=stats.index)
    )
    # A starter is on from minute 0. Where the timeline is missing, sixty minutes
    # on the pitch is taken as having started.
    out["ci_started"] = np.where(
        start.notna(), (start.fillna(1) == 0) & (out["minutes"] > 0), out["minutes"] >= 60
    ).astype(int)
    out = out[out["code"].notna()].copy()
    out["code"] = out["code"].astype(int)
    return out[ROW_COLUMNS]


def fetch_core_insights(
    seasons: tuple[str, ...] = TRAIN_SEASONS, *, source: str | Path | None = None
) -> dict[str, int]:
    """Refresh the bronze copies. ``source`` is a local checkout; otherwise a
    shallow sparse clone is made (only the season folders needed are checked out).

    Returns rows written per season. A season the dataset does not have is skipped
    with a warning rather than failing the refresh.
    """
    CORE_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    tmp = None
    try:
        if source is None:
            tmp = tempfile.mkdtemp(prefix="core-insights-")
            root = Path(tmp) / "repo"
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--quiet",
                 REPO, str(root)],
                check=True, timeout=300,
            )  # fmt: skip
            subprocess.run(
                ["git", "-C", str(root), "sparse-checkout", "set", "--no-cone",
                 *[f"data/{_season_dir(s)}" for s in seasons], "!*/By Tournament/*"],
                check=True, timeout=600,
            )  # fmt: skip
        else:
            root = Path(source)
        for season in seasons:
            folder = root / "data" / _season_dir(season)
            if not folder.exists():
                log.warning("core insights: no folder for %s", season)
                continue
            rows = _season_rows(folder, season)
            path = CORE_DIR / f"core_insights_{season}.csv"
            rows.to_csv(path, index=False)
            written[season] = len(rows)
            log.info("core insights: %s -> %d player-match rows", season, len(rows))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return written


def load_core_rows(seasons: tuple[str, ...] = TRAIN_SEASONS) -> pd.DataFrame:
    """The stored per-match rows for the seasons on disk."""
    parts = []
    for season in seasons:
        path = CORE_DIR / f"core_insights_{season}.csv"
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        return pd.DataFrame(columns=ROW_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def load_core_stats(seasons: tuple[str, ...] = TRAIN_SEASONS) -> pd.DataFrame:
    """Per player-gameweek sums: ``season, code, GW`` plus the ``ci_*`` columns.

    Only seasons with data are returned; the feature builder fills a covered
    season's missing player-gameweeks with zero and leaves the rest NaN.
    """
    rows = load_core_rows(seasons)
    if rows.empty:
        return pd.DataFrame(columns=["season", "code", "GW", *CORE_COLUMNS])
    rows = rows.drop(columns=["match_id", "minutes"])
    agg = rows.groupby(["season", "code", "GW"], as_index=False).sum(numeric_only=True)
    agg["ci_matches"] = rows.groupby(["season", "code", "GW"]).size().to_numpy()
    return agg[["season", "code", "GW", *CORE_COLUMNS]]


def covered_seasons() -> set[str]:
    return {p.stem.replace("core_insights_", "") for p in CORE_DIR.glob("core_insights_*.csv")}

# FPL Analyst — full source

Phases 00–01. 3 commits, 33 tests passing, 1 strict-xfail.
`fpl bootstrap` → 109,282 rows. `fpl silver` → 109,282 resolved, 1,559 players.

## Contents

- `pyproject.toml`
- `src/fpl/config.py`
- `src/fpl/data/archive.py`
- `src/fpl/entity/resolve.py`
- `src/fpl/entity/mappings/teams.csv`
- `src/fpl/data/silver.py`
- `src/fpl/cli.py`
- `tests/test_config.py`
- `tests/test_archive.py`
- `tests/test_entity.py`
- `tests/test_no_leakage.py`
- `.github/workflows/tests.yml`
- `.github/workflows/weekly-refresh.yml`
- `.gitignore`


---

## `pyproject.toml`
*46 lines*

```toml
[project]
name = "fpl-analyst"
version = "0.1.0"
description = "Weekly Fantasy Premier League points projection, squad optimisation and grounded explanation."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.1",
    "numpy>=1.26",
    "pyarrow>=15.0",
    "requests>=2.31",
    "lightgbm>=4.3",
    "scikit-learn>=1.4",
    "scipy>=1.11",
    "pulp>=2.8",
    "rapidfuzz>=3.6",
    "platformdirs>=4.2",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.6", "responses>=0.25"]
llm = ["anthropic>=0.40"]

[project.scripts]
fpl = "fpl.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/fpl"]

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "PD"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = ["network: hits a live external API (skipped by default)"]
```

---

## `src/fpl/config.py`
*53 lines*

```python
"""Central configuration: paths, seasons, and FPL scoring constants."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FPL_DATA_DIR", PROJECT_ROOT / "data"))

BRONZE = DATA_DIR / "bronze"  # immutable, deadline-stamped snapshots
SILVER = DATA_DIR / "silver"  # canonical, entity-resolved
GOLD = DATA_DIR / "gold"  # leak-free feature store

for _d in (BRONZE, SILVER, GOLD):
    _d.mkdir(parents=True, exist_ok=True)

# Seasons used for training. The archive covers 2016-17 onward, but expected-goals
# columns only become reliable from 2021-22, so we start there.
TRAIN_SEASONS: tuple[str, ...] = ("2021-22", "2022-23", "2023-24", "2024-25")
CURRENT_SEASON = "2026-27"

FPL_API_BASE = "https://fantasy.premierleague.com/api"
ARCHIVE_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

POSITIONS = ("GK", "DEF", "MID", "FWD")

# --- FPL scoring rules (2025-26 onward) -------------------------------------
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
APPEARANCE_POINTS = 1  # for playing at all
SIXTY_MINUTE_POINTS = 1  # additional, for 60+ minutes
DEFENSIVE_CONTRIBUTION_POINTS = 2
# defensive-contribution thresholds (CBIT for defenders, CBIRT for others)
DC_THRESHOLD = {"GK": None, "DEF": 10, "MID": 12, "FWD": 12}
SAVES_PER_POINT = 3
CONCEDED_PER_MINUS_ONE = 2  # GK/DEF only
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
OWN_GOAL_POINTS = -2
PENALTY_MISS_POINTS = -2
PENALTY_SAVE_POINTS = 5

# --- squad constraints ------------------------------------------------------
SQUAD_SIZE = 15
SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
XI_SIZE = 11
MAX_PER_CLUB = 3
BUDGET_TENTHS = 1000  # £100.0m, stored in tenths as the API does
TRANSFER_HIT = 4  # points cost of an extra transfer
```

---

## `src/fpl/data/archive.py`
*154 lines*

```python
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
        "name",
        "position",
        "team",
        "GW",
        "minutes",
        "total_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "bps",
        "yellow_cards",
        "red_cards",
        "own_goals",
        "penalties_missed",
        "penalties_saved",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "was_home",
        "opponent_team",
        "value",
        "selected",
        "kickoff_time",
    }
)

TIMEOUT = 60


def _season_url(season: str) -> str:
    return f"{ARCHIVE_BASE}/{season}/gws/merged_gw.csv"


def _players_url(season: str) -> str:
    return f"{ARCHIVE_BASE}/{season}/players_raw.csv"


PLAYER_COLUMNS = frozenset({"id", "code", "first_name", "second_name", "element_type", "team"})


def fetch_players(season: str, *, session: requests.Session | None = None) -> pd.DataFrame:
    """Download one season's player registry.

    This is what maps a season-local ``element`` id to the cross-season ``code``.
    See ``fpl.entity.resolve`` for why that distinction is load-bearing.
    """
    url = _players_url(season)
    log.info("fetching %s", url)
    getter = session.get if session else requests.get
    response = getter(url, timeout=TIMEOUT)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))
    missing = PLAYER_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"{season}: player registry is missing {sorted(missing)}. "
            "The upstream schema has changed and the loader needs updating."
        )
    frame["season"] = season
    return frame


def load_players(
    seasons: tuple[str, ...] = TRAIN_SEASONS, *, refresh: bool = False
) -> pd.DataFrame:
    """Load player registries for several seasons, cached to bronze."""
    frames = []
    for season in seasons:
        cache = BRONZE / f"players_{season}.parquet"
        if cache.exists() and not refresh:
            frames.append(pd.read_parquet(cache))
            continue
        frame = fetch_players(season)
        frame.to_parquet(cache, index=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


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
```

---

## `src/fpl/entity/resolve.py`
*231 lines*

```python
"""Canonical player and team identity across seasons and across sources.

Why this module exists
----------------------
FPL's ``element`` id (the ``element`` column in ``merged_gw.csv``) is **season-local
and reused**. Measured across the 2023-24 and 2024-25 registries: 803 ids appear in
both seasons, and in all 803 cases they refer to a *different footballer*.

Joining gameweek rows across seasons on ``element`` therefore produces 100% wrong
player identity, silently. Every rolling form feature would be computed over a
mixture of unrelated players and nothing would visibly break.

The cross-season identifier is ``code``, which lives in ``players_raw.csv``. This
module builds the ``(season, element) -> code`` bridge and exposes name matching for
the sources that have no FPL id at all.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MAPPINGS_DIR = Path(__file__).parent / "mappings"
TEAM_ALIASES_PATH = MAPPINGS_DIR / "teams.csv"
PLAYER_OVERRIDES_PATH = MAPPINGS_DIR / "players.csv"

# element_type 5 is a *manager*, introduced in 2024-25 alongside the Assistant
# Manager chip. Managers score by completely different rules (match result, team
# clean sheet, goal difference) and are not footballers. They are labelled here
# rather than dropped, so that filtering them out is a visible decision.
ELEMENT_TYPE_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", 5: "MGR"}
MANAGER_POSITION = "MGR"

# Below this rapidfuzz score a match is not trusted and must be resolved by hand.
MATCH_THRESHOLD = 88


def normalise_name(name: str) -> str:
    """Fold a name to a comparable form.

    Strips diacritics, punctuation and case, and collapses whitespace, so that
    "Gabriel dos Santos Magalhães" and "Gabriel Dos Santos Magalhaes" agree.
    """
    if not isinstance(name, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", ascii_only)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def build_player_index(players: pd.DataFrame) -> pd.DataFrame:
    """Build the ``(season, element) -> code`` bridge from player registries.

    Args:
        players: concatenated ``players_raw`` frames, as returned by
            ``fpl.data.archive.load_players``.

    Returns:
        One row per (season, element) with the stable ``code``, full name and position.
    """
    index = players.rename(columns={"id": "element"})[
        ["season", "element", "code", "first_name", "second_name", "element_type"]
    ].copy()

    index["position"] = index["element_type"].map(ELEMENT_TYPE_TO_POSITION)
    if index["position"].isna().any():
        unknown = sorted(index.loc[index["position"].isna(), "element_type"].unique())
        raise ValueError(f"unknown element_type values {unknown}; FPL added a position")

    index["full_name"] = (
        index["first_name"].fillna("") + " " + index["second_name"].fillna("")
    ).str.strip()
    index["norm_name"] = index["full_name"].map(normalise_name)

    duplicated = index.duplicated(subset=["season", "element"]).sum()
    if duplicated:
        raise ValueError(f"{duplicated} duplicate (season, element) pairs in the registry")

    return index.drop(columns=["element_type"])


def drop_managers(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove manager rows, logging how many went.

    The player model predicts footballer points. Managers need their own model and
    are out of scope for v1 — but they are removed loudly, because a silent filter
    is how a row count quietly stops matching the thing you think you are modelling.
    """
    if "position" not in frame.columns:
        raise ValueError("call attach_player_code before drop_managers")
    is_manager = frame["position"] == MANAGER_POSITION
    if is_manager.any():
        log.info("dropping %d manager rows (out of scope for the player model)", is_manager.sum())
    return frame.loc[~is_manager].copy()


def attach_player_code(gameweeks: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Attach the cross-season ``code`` to gameweek rows.

    Raises:
        ValueError: any gameweek row fails to resolve. A partial join here would
            corrupt every downstream rolling feature, so it is fatal rather than
            something to fix up later.
    """
    merged = gameweeks.merge(
        index[["season", "element", "code", "position", "full_name"]],
        on=["season", "element"],
        how="left",
        suffixes=("", "_registry"),
    )

    unresolved = merged["code"].isna()
    if unresolved.any():
        sample = (
            merged.loc[unresolved, ["season", "element", "name"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{unresolved.sum()} gameweek rows have no registry entry. Sample: {sample}"
        )

    merged["code"] = merged["code"].astype("int64")
    return merged


def _load_overrides(path: Path) -> dict[str, int]:
    """Read the hand-curated name -> code overrides, if any exist yet."""
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return dict(zip(frame["source_name"].map(normalise_name), frame["code"], strict=False))


def match_external_names(
    names: list[str],
    index: pd.DataFrame,
    *,
    threshold: int = MATCH_THRESHOLD,
    overrides_path: Path = PLAYER_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Match names from an external source (Understat, odds feeds) to FPL codes.

    Fuzzy matching proposes; the overrides file decides. Anything scoring below
    ``threshold`` is returned unmatched rather than guessed at, so that ambiguous
    cases surface as a short review list instead of quietly wrong joins.

    Returns:
        Frame with columns ``source_name, code, matched_name, score, method``.
        ``code`` is ``<NA>`` where no confident match was found.
    """
    from rapidfuzz import process
    from rapidfuzz.fuzz import WRatio

    overrides = _load_overrides(overrides_path)
    candidates = index.drop_duplicates(subset="code")[["code", "norm_name", "full_name"]]
    lookup = candidates["norm_name"].tolist()

    rows = []
    for name in names:
        norm = normalise_name(name)

        if norm in overrides:
            rows.append(
                {
                    "source_name": name,
                    "code": overrides[norm],
                    "matched_name": None,
                    "score": 100.0,
                    "method": "override",
                }
            )
            continue

        best = process.extractOne(norm, lookup, scorer=WRatio)
        if best is None or best[1] < threshold:
            rows.append(
                {
                    "source_name": name,
                    "code": pd.NA,
                    "matched_name": None,
                    "score": float(best[1]) if best else 0.0,
                    "method": "unmatched",
                }
            )
            continue

        hit = candidates.iloc[best[2]]
        rows.append(
            {
                "source_name": name,
                "code": int(hit["code"]),
                "matched_name": hit["full_name"],
                "score": float(best[1]),
                "method": "fuzzy",
            }
        )

    result = pd.DataFrame(rows)
    unmatched = int((result["method"] == "unmatched").sum())
    if unmatched:
        log.warning(
            "%d of %d names unmatched at threshold %d; add them to %s",
            unmatched,
            len(names),
            threshold,
            overrides_path.name,
        )
    return result


def load_team_aliases(path: Path = TEAM_ALIASES_PATH) -> dict[str, str]:
    """Map every known spelling of a club to its FPL short name."""
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return dict(zip(frame["alias"].map(normalise_name), frame["fpl_name"], strict=False))


def canonical_team(name: str, aliases: dict[str, str] | None = None) -> str | None:
    """Resolve one club spelling. Returns ``None`` when unknown, never a guess."""
    aliases = load_team_aliases() if aliases is None else aliases
    return aliases.get(normalise_name(name))
```

---

## `src/fpl/entity/mappings/teams.csv`
*46 lines*

```csv
alias,fpl_name,source
Arsenal,Arsenal,fpl
Arsenal FC,Arsenal,understat
Aston Villa,Aston Villa,fpl
Bournemouth,Bournemouth,fpl
AFC Bournemouth,Bournemouth,understat
Brentford,Brentford,fpl
Brighton,Brighton,fpl
Brighton & Hove Albion,Brighton,understat
Burnley,Burnley,fpl
Chelsea,Chelsea,fpl
Crystal Palace,Crystal Palace,fpl
Everton,Everton,fpl
Fulham,Fulham,fpl
Ipswich,Ipswich,fpl
Ipswich Town,Ipswich,understat
Leeds,Leeds,fpl
Leeds United,Leeds,understat
Leicester,Leicester,fpl
Leicester City,Leicester,understat
Liverpool,Liverpool,fpl
Luton,Luton,fpl
Luton Town,Luton,understat
Man City,Man City,fpl
Manchester City,Man City,understat
Man Utd,Man Utd,fpl
Man United,Man Utd,football-data
Manchester United,Man Utd,understat
Newcastle,Newcastle,fpl
Newcastle United,Newcastle,understat
Norwich,Norwich,fpl
Norwich City,Norwich,understat
Nott'm Forest,Nott'm Forest,fpl
Nottingham Forest,Nott'm Forest,understat
Sheffield Utd,Sheffield Utd,fpl
Sheffield United,Sheffield Utd,football-data
Southampton,Southampton,fpl
Spurs,Spurs,fpl
Tottenham,Spurs,football-data
Tottenham Hotspur,Spurs,understat
Sunderland,Sunderland,fpl
Watford,Watford,fpl
West Ham,West Ham,fpl
West Ham United,West Ham,understat
Wolves,Wolves,fpl
Wolverhampton Wanderers,Wolves,understat
```

---

## `src/fpl/data/silver.py`
*73 lines*

```python
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

    duplicates = resolved.duplicated(subset=["code", "season", "GW", "fixture"]).sum()
    if duplicates:
        log.warning("%d duplicate (code, season, GW, fixture) rows retained", duplicates)

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
```

---

## `src/fpl/cli.py`
*72 lines*

```python
"""Command line entry point: ``fpl <command>``."""

from __future__ import annotations

import argparse
import logging
import sys

from fpl.config import TRAIN_SEASONS


def _bootstrap(args: argparse.Namespace) -> int:
    from fpl.data.archive import load_seasons

    seasons = tuple(args.seasons) if args.seasons else TRAIN_SEASONS
    frame = load_seasons(seasons, refresh=args.refresh)
    print(f"{len(frame):,} player-gameweek rows across {len(seasons)} seasons")
    print(
        f"gameweeks {frame['GW'].min()}-{frame['GW'].max()}, "
        f"{frame['name'].nunique():,} distinct players"
    )
    return 0


def _silver(args: argparse.Namespace) -> int:
    from fpl.data.silver import build_silver

    seasons = tuple(args.seasons) if args.seasons else TRAIN_SEASONS
    frame = build_silver(seasons)
    print(
        f"{len(frame):,} rows | {frame['code'].nunique():,} players | "
        f"{frame['team'].nunique()} clubs | seasons {frame['season'].min()}-{frame['season'].max()}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    # Shared flags live on a parent parser so they work either side of the
    # subcommand: both "fpl -v bootstrap" and "fpl bootstrap -v" are accepted.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="log progress")

    parser = argparse.ArgumentParser(prog="fpl", description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser(
        "bootstrap",
        parents=[common],
        help="download historical seasons into bronze",
    )
    boot.add_argument("--seasons", nargs="*", help=f"default: {' '.join(TRAIN_SEASONS)}")
    boot.add_argument("--refresh", action="store_true", help="ignore the local cache")
    boot.set_defaults(func=_bootstrap)

    silver = sub.add_parser(
        "silver",
        parents=[common],
        help="resolve identity and build the canonical table",
    )
    silver.add_argument("--seasons", nargs="*", help=f"default: {' '.join(TRAIN_SEASONS)}")
    silver.set_defaults(func=_silver)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

---

## `tests/test_config.py`
*27 lines*

```python
"""The scoring constants are the model's contract with the game. Guard them."""

from fpl import config


def test_squad_quota_sums_to_squad_size():
    assert sum(config.SQUAD_QUOTA.values()) == config.SQUAD_SIZE


def test_xi_bounds_are_satisfiable():
    assert sum(config.XI_MIN.values()) <= config.XI_SIZE <= sum(config.XI_MAX.values())


def test_every_position_has_scoring_rules():
    for position in config.POSITIONS:
        assert position in config.GOAL_POINTS
        assert position in config.CLEAN_SHEET_POINTS
        assert position in config.DC_THRESHOLD


def test_clean_sheet_points_match_fpl_rules():
    assert config.CLEAN_SHEET_POINTS == {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}


def test_squad_cannot_be_bought_from_one_club():
    # 15 players, max 3 per club, so at least 5 clubs must be represented.
    assert config.SQUAD_SIZE / config.MAX_PER_CLUB >= 5
```

---

## `tests/test_archive.py`
*51 lines*

```python
"""The archive loader must fail loudly when upstream changes shape."""

import pandas as pd
import pytest

from fpl.data import archive


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, text: str):
        self._text = text

    def get(self, url, timeout=None):  # noqa: ARG002
        return _FakeResponse(self._text)


def _valid_csv() -> str:
    row = dict.fromkeys(archive.REQUIRED_COLUMNS, 0)
    row["name"] = "Test Player"
    row["position"] = "MID"
    row["team"] = "Arsenal"
    row["kickoff_time"] = "2024-08-16T19:00:00Z"
    row["was_home"] = True
    return pd.DataFrame([row]).to_csv(index=False)


def test_fetch_season_parses_valid_data():
    frame = archive.fetch_season("2024-25", session=_FakeSession(_valid_csv()))
    assert len(frame) == 1
    assert frame.loc[0, "season"] == "2024-25"
    assert pd.api.types.is_datetime64_any_dtype(frame["kickoff_time"])


def test_fetch_season_rejects_missing_columns():
    text = pd.DataFrame([{"name": "x", "GW": 1}]).to_csv(index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        archive.fetch_season("2024-25", session=_FakeSession(text))


def test_required_columns_include_the_target():
    # If total_points ever drops out of the contract, there is nothing to learn.
    assert "total_points" in archive.REQUIRED_COLUMNS
    assert "minutes" in archive.REQUIRED_COLUMNS
```

---

## `tests/test_entity.py`
*200 lines*

```python
"""Identity is the foundation. If these break, every downstream feature is wrong."""

import pandas as pd
import pytest

from fpl.entity import resolve


def _registry() -> pd.DataFrame:
    """Two seasons in which element id 7 is reused for a different footballer."""
    return pd.DataFrame(
        [
            {
                "season": "2023-24",
                "id": 7,
                "code": 1001,
                "first_name": "Gabriel",
                "second_name": "dos Santos Magalhães",
                "element_type": 2,
                "team": 1,
            },
            {
                "season": "2023-24",
                "id": 9,
                "code": 1002,
                "first_name": "Son",
                "second_name": "Heung-min",
                "element_type": 3,
                "team": 2,
            },
            {
                "season": "2024-25",
                "id": 7,
                "code": 1002,
                "first_name": "Heung-Min",
                "second_name": "Son",
                "element_type": 3,
                "team": 2,
            },
            {
                "season": "2024-25",
                "id": 12,
                "code": 1003,
                "first_name": "Mikel",
                "second_name": "Arteta",
                "element_type": 5,
                "team": 1,
            },
        ]
    )


# --- name normalisation -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Gabriel dos Santos Magalhães", "gabriel dos santos magalhaes"),
        ("Gabriel Dos Santos Magalhaes", "gabriel dos santos magalhaes"),
        ("  Kevin   De  Bruyne  ", "kevin de bruyne"),
        ("N'Golo Kanté", "n golo kante"),
        ("Nott'm Forest", "nott m forest"),
    ],
)
def test_normalise_name_folds_accents_and_punctuation(raw, expected):
    assert resolve.normalise_name(raw) == expected


def test_normalise_name_handles_non_strings():
    assert resolve.normalise_name(None) == ""
    assert resolve.normalise_name(float("nan")) == ""


# --- the element trap -------------------------------------------------------


def test_element_ids_are_reused_across_seasons():
    """The premise of the whole module: element 7 is two different people."""
    index = resolve.build_player_index(_registry())
    element_7 = index[index["element"] == 7]
    assert len(element_7) == 2
    assert element_7["code"].nunique() == 2, "element must not be treated as an identity"


def test_code_is_stable_across_seasons_despite_name_reordering():
    """Son appears as 'Son Heung-min' then 'Heung-Min Son'. Same code, both times."""
    index = resolve.build_player_index(_registry())
    son = index[index["code"] == 1002]
    assert son["season"].nunique() == 2
    assert son["full_name"].nunique() == 2, "fixture should exercise the rename"


def test_build_player_index_maps_positions():
    index = resolve.build_player_index(_registry())
    assert set(index["position"]) == {"DEF", "MID", "MGR"}


def test_build_player_index_rejects_unknown_position():
    registry = _registry()
    registry.loc[0, "element_type"] = 99
    with pytest.raises(ValueError, match="unknown element_type"):
        resolve.build_player_index(registry)


def test_build_player_index_rejects_duplicate_keys():
    registry = pd.concat([_registry(), _registry().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        resolve.build_player_index(registry)


# --- attaching codes to gameweeks -------------------------------------------


def _gameweeks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": "2023-24", "element": 7, "name": "Gabriel", "total_points": 6},
            {"season": "2024-25", "element": 7, "name": "Son", "total_points": 2},
        ]
    )


def test_attach_player_code_resolves_reused_ids_correctly():
    resolved = resolve.attach_player_code(_gameweeks(), resolve.build_player_index(_registry()))
    by_season = dict(zip(resolved["season"], resolved["code"], strict=True))
    assert by_season["2023-24"] == 1001
    assert by_season["2024-25"] == 1002


def test_attach_player_code_is_fatal_on_unresolved_rows():
    """A partial join would corrupt every rolling feature. It must not be survivable."""
    orphan = pd.DataFrame([{"season": "2023-24", "element": 999, "name": "Nobody"}])
    with pytest.raises(ValueError, match="no registry entry"):
        resolve.attach_player_code(orphan, resolve.build_player_index(_registry()))


def test_drop_managers_removes_only_managers():
    index = resolve.build_player_index(_registry())
    frame = pd.DataFrame(
        [
            {"season": "2024-25", "element": 12, "name": "Arteta"},
            {"season": "2024-25", "element": 7, "name": "Son"},
        ]
    )
    kept = resolve.drop_managers(resolve.attach_player_code(frame, index))
    assert list(kept["position"]) == ["MID"]


def test_drop_managers_requires_resolution_first():
    with pytest.raises(ValueError, match="attach_player_code"):
        resolve.drop_managers(pd.DataFrame([{"name": "x"}]))


# --- external name matching -------------------------------------------------


def test_match_external_names_finds_close_matches():
    index = resolve.build_player_index(_registry())
    result = resolve.match_external_names(["Gabriel Dos Santos Magalhaes"], index)
    assert result.loc[0, "code"] == 1001
    assert result.loc[0, "method"] == "fuzzy"


def test_match_external_names_refuses_to_guess():
    """Unknown names come back unmatched, never attached to the nearest stranger."""
    index = resolve.build_player_index(_registry())
    result = resolve.match_external_names(["Cristiano Ronaldo"], index)
    assert pd.isna(result.loc[0, "code"])
    assert result.loc[0, "method"] == "unmatched"


# --- teams ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("Manchester United", "Man Utd"),
        ("Man United", "Man Utd"),
        ("MAN UTD", "Man Utd"),
        ("Nottingham Forest", "Nott'm Forest"),
        ("Tottenham", "Spurs"),
        ("Wolverhampton Wanderers", "Wolves"),
    ],
)
def test_canonical_team_resolves_known_spellings(spelling, expected):
    assert resolve.canonical_team(spelling) == expected


def test_canonical_team_returns_none_for_unknown():
    assert resolve.canonical_team("Real Madrid") is None


def test_every_fpl_spelling_maps_to_itself():
    """Each club's own FPL name must round-trip, or joins to FPL data will fail."""
    aliases = resolve.load_team_aliases()
    fpl_names = set(aliases.values())
    for name in fpl_names:
        assert resolve.canonical_team(name, aliases) == name, f"{name} does not round-trip"
```

---

## `tests/test_no_leakage.py`
*17 lines*

```python
"""The spine of the project.

Every feature used to predict gameweek t must be computable from data timestamped
strictly before gameweek t's deadline. This file is deliberately created empty of
real assertions in phase 00 and filled in phase 02, when the feature builder exists.

It is here from day one so that it is never "added later".
"""

import pytest


@pytest.mark.xfail(reason="feature builder arrives in phase 02", strict=True)
def test_no_feature_uses_future_information():
    from fpl.features.build import build_features  # noqa: F401

    raise AssertionError("not implemented yet")
```

---

## `.github/workflows/tests.yml`
*34 lines*

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip

      - name: Install
        run: pip install -e ".[dev]"

      - name: Lint
        run: ruff check .

      - name: Format check
        run: ruff format --check .

      - name: Test
        run: pytest --cov=fpl --cov-report=term-missing
```

---

## `.github/workflows/weekly-refresh.yml`
*43 lines*

```yaml
name: weekly refresh

# The vaastav archive stopped weekly updates after 2024-25, so this job IS the
# in-season data source. It runs before each gameweek deadline, refreshes data,
# regenerates projections and publishes the dashboard.
on:
  schedule:
    - cron: "0 9 * * 4"   # Thursdays 09:00 UTC, ahead of Friday deadlines
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[dev,llm]"

      # Phases 01-07 fill these in. Kept as explicit steps so the shape of the
      # weekly job is visible from day one.
      - name: Snapshot live data
        run: echo "phase 01 - fpl snapshot"
      - name: Build features
        run: echo "phase 02 - fpl features"
      - name: Predict
        run: echo "phase 04 - fpl predict"
      - name: Optimise squad
        run: echo "phase 05 - fpl optimise"
      - name: Explain picks
        run: echo "phase 06 - fpl explain"
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
      - name: Publish
        run: echo "phase 07 - fpl publish"
```

---

## `.gitignore`
*29 lines*

```text
# data is regenerated, never committed
data/bronze/
data/silver/
data/gold/
*.parquet
*.csv
!src/fpl/entity/mappings/*.csv
!tests/fixtures/*.csv

# secrets
.env
.env.*

# python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
.ipynb_checkpoints/

# os / editor
.DS_Store
.vscode/
.idea/
```

---

**Total: 1076 lines across 14 files.**
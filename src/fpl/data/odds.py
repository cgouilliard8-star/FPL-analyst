"""Bookmaker odds as features: what the market thinks each match will look like.

Closing odds absorb team news, rotation and form better than any rolling window can,
which is why every serious projection service uses them. The source is
football-data.co.uk: one CSV per season of results with a dozen bookmakers' prices
(``mmz4281/<yyzz>/E0.csv``), and ``fixtures.csv`` with the same columns for the
matches still to be played. Both are free and need no key.

From the 1X2 and over/under 2.5 prices each match yields, per side: the chance of
winning, drawing and losing, and -- by solving a two-team Poisson model so that the
win probability and the over-2.5 probability both match the market -- an implied
expected goals for and against, and hence a clean-sheet probability. Those are the
numbers a defender's or striker's projection actually turns on.

Files are stored in ``data/external/odds_<season>.csv`` and committed, so a source
that is down keeps its last copy; when a season is missing the features are NaN and
the model falls back to what it knows.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fpl.config import CURRENT_SEASON, PROJECT_ROOT, TRAIN_SEASONS
from fpl.entity.resolve import canonical_team, load_team_aliases

log = logging.getLogger(__name__)

EXTERNAL = PROJECT_ROOT / "data" / "external"
BASE = "https://www.football-data.co.uk"
# The same files by other names, tried in turn: the site sits behind a bot filter
# that has answered 503 to plain clients for days at a time.
MIRRORS = (
    "https://www.football-data.co.uk",
    "https://football-data.co.uk",
    "http://www.football-data.co.uk",
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.football-data.co.uk/englandm.php",
}
TIMEOUT = 30
DIVISION = "E0"
ODDS_API = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"

ODDS_FEATURES = (
    "odds_win", "odds_draw", "odds_lose", "odds_xg", "odds_xgc", "odds_cs", "odds_over25",
)  # fmt: skip
COLUMNS = ["season", "kickoff_date", "home", "away", *ODDS_FEATURES]

# Preferred price columns, first available wins: Pinnacle closing, market average, Bet365.
_1X2 = (("PSCH", "PSCD", "PSCA"), ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"))
_OU = (("AvgC>2.5", "AvgC<2.5"), ("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5"))


def _season_code(season: str) -> str:
    start = season.split("-")[0]
    return start[2:] + str(int(start) + 1)[2:]


def odds_path(season: str) -> Path:
    return EXTERNAL / f"odds_{season}.csv"


def _implied(row: pd.Series, columns: tuple[str, ...]) -> list[float] | None:
    try:
        prices = [float(row[c]) for c in columns]
    except (KeyError, TypeError, ValueError):
        return None
    if any(not np.isfinite(p) or p <= 1.0 for p in prices):
        return None
    inverse = [1.0 / p for p in prices]
    total = sum(inverse)
    return [x / total for x in inverse]


def _first_implied(row: pd.Series, options) -> list[float] | None:
    for columns in options:
        probs = _implied(row, columns)
        if probs is not None:
            return probs
    return None


def _poisson_cdf(k: int, lam: float) -> float:
    return sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k + 1))


def _p_home_win(lh: float, la: float, cap: int = 12) -> float:
    ph = [math.exp(-lh) * lh**i / math.factorial(i) for i in range(cap)]
    pa = [math.exp(-la) * la**j / math.factorial(j) for j in range(cap)]
    return sum(ph[i] * pa[j] for i in range(cap) for j in range(i))


def implied_goals(p_home: float, p_over25: float) -> tuple[float, float]:
    """Home and away expected goals such that a two-team Poisson model reproduces the
    market's home-win and over-2.5 probabilities.

    Total goals come from the over/under line alone (Poisson total); the split
    between the sides is then found by bisection on the home-win probability.
    """
    lo, hi = 0.2, 6.0
    for _ in range(60):  # total λ: P(total > 2.5) = 1 - CDF(2)
        mid = (lo + hi) / 2
        if 1.0 - _poisson_cdf(2, mid) < p_over25:
            lo = mid
        else:
            hi = mid
    total = (lo + hi) / 2
    lo, hi = 0.02, total - 0.02
    for _ in range(60):  # home share: P(home > away) = p_home
        mid = (lo + hi) / 2
        if _p_home_win(mid, total - mid) < p_home:
            lo = mid
        else:
            hi = mid
    home = (lo + hi) / 2
    return home, total - home


def parse_football_data(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Per-side rows with the market's view of each match."""
    aliases = load_team_aliases()
    rows = []
    for _, r in raw.iterrows():
        if "Div" in raw.columns and str(r.get("Div", DIVISION)) != DIVISION:
            continue
        home = canonical_team(str(r.get("HomeTeam", "")), aliases)
        away = canonical_team(str(r.get("AwayTeam", "")), aliases)
        if home is None or away is None:
            continue
        one = _first_implied(r, _1X2)
        ou = _first_implied(r, _OU)
        if one is None:
            continue
        p_h, p_d, p_a = one
        p_over = ou[0] if ou is not None else 0.52  # league-average share of 3+ goal games
        lh, la = implied_goals(p_h, p_over)
        date = pd.to_datetime(str(r.get("Date", "")), dayfirst=True, errors="coerce")
        base = {"season": season, "kickoff_date": date.date() if pd.notna(date) else None}
        rows.append({**base, "home": home, "away": away, "side": "home", "team": home,
                     "odds_win": p_h, "odds_draw": p_d, "odds_lose": p_a, "odds_xg": lh,
                     "odds_xgc": la, "odds_cs": math.exp(-la), "odds_over25": p_over})  # fmt: skip
        rows.append({**base, "home": home, "away": away, "side": "away", "team": away,
                     "odds_win": p_a, "odds_draw": p_d, "odds_lose": p_h, "odds_xg": la,
                     "odds_xgc": lh, "odds_cs": math.exp(-lh), "odds_over25": p_over})  # fmt: skip
    return pd.DataFrame(rows, columns=[*COLUMNS, "side", "team"])


def fetch_odds(
    seasons: tuple[str, ...] = (*TRAIN_SEASONS, CURRENT_SEASON),
    *,
    session: requests.Session | None = None,
    upcoming: bool = True,
) -> dict[str, int]:
    """Download each season's results file (and the upcoming fixtures file) and store
    the raw CSVs. Returns rows stored per season; a season that cannot be fetched
    keeps whatever is already on disk."""
    session = session or requests.Session()
    for key, value in HEADERS.items():
        session.headers.setdefault(key, value)
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    stored: dict[str, int] = {}
    for season in seasons:
        text = _fetch_csv(session, f"mmz4281/{_season_code(season)}/{DIVISION}.csv", "HomeTeam")
        if text is None:
            log.warning("odds %s: every mirror failed; keeping the stored file", season)
            continue
        odds_path(season).write_text(text)
        stored[season] = text.count("\n") - 1
        log.info("odds %s: %d matches", season, stored[season])
    if upcoming:
        text = _fetch_csv(session, "fixtures.csv", "HomeTeam")
        if text is not None:
            (EXTERNAL / "odds_fixtures.csv").write_text(text)
            stored["fixtures"] = text.count("\n") - 1
        else:
            log.warning("odds fixtures: every mirror failed")
    return stored


def _fetch_csv(session: requests.Session, path: str, marker: str) -> str | None:
    """One CSV from the first mirror that answers with the real thing. Every failure
    is logged with its status so the Actions log says *why* the odds are missing."""
    for base in MIRRORS:
        url = f"{base}/{path}"
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code != 200:
                log.warning("odds: %s -> HTTP %s", url, r.status_code)
                continue
            text = r.content.decode("utf-8", errors="replace")
            if marker not in text.splitlines()[0]:
                log.warning("odds: %s -> not a CSV (%s...)", url, text[:60].replace("\n", " "))
                continue
            return text
        except Exception as e:  # noqa: BLE001
            log.warning("odds: %s -> %s", url, e)
    return None


def fetch_live_odds(
    season: str = CURRENT_SEASON,
    *,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> int:
    """Today's prices for the coming Premier League matches from The Odds API (free
    tier: 500 requests a month; this is one request), stored beside the results files
    as ``odds_live_<season>.csv`` and merged with anything already there, so a
    history of pre-match prices accumulates on its own even while football-data.co.uk
    is unreachable. Returns the number of matches stored. Needs ``ODDS_API_KEY``."""
    import os

    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        log.info("odds live: no ODDS_API_KEY, skipping")
        return 0
    session = session or requests.Session()
    params = {
        "regions": "uk,eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "apiKey": api_key,
    }
    r = session.get(ODDS_API, params=params, timeout=TIMEOUT)
    if r.status_code != 200:
        log.warning("odds live: HTTP %s (%s)", r.status_code, r.text[:120])
        return 0
    rows = parse_odds_api(r.json())
    if not rows:
        log.warning("odds live: no usable matches in the reply")
        return 0
    frame = pd.DataFrame(rows)
    frame["season"] = season
    path = EXTERNAL / f"odds_live_{season}.csv"
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([frame, old], ignore_index=True)
    # keep the latest price seen for each match
    frame = frame.drop_duplicates(["home", "away"], keep="first")
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    log.info("odds live: %d matches stored (%s)", len(rows), path.name)
    return len(rows)


def parse_odds_api(payload: list[dict]) -> list[dict]:
    """Average each match's 1X2 and over/under 2.5 prices across the bookmakers
    quoted, as one row per match in football-data.co.uk's column names."""
    aliases = load_team_aliases()
    out = []
    for event in payload:
        home = canonical_team(event.get("home_team", ""), aliases)
        away = canonical_team(event.get("away_team", ""), aliases)
        if home is None or away is None:
            log.warning(
                "odds live: unknown club in %s v %s", event.get("home_team"), event.get("away_team")
            )
            continue
        h, d, a, over, under = [], [], [], [], []
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                prices = {o.get("name"): o for o in market.get("outcomes", [])}
                if market.get("key") == "h2h" and {
                    event["home_team"],
                    event["away_team"],
                    "Draw",
                } <= set(prices):
                    h.append(prices[event["home_team"]]["price"])
                    a.append(prices[event["away_team"]]["price"])
                    d.append(prices["Draw"]["price"])
                elif market.get("key") == "totals":
                    o = prices.get("Over")
                    u = prices.get("Under")
                    if o and u and float(o.get("point", 0)) == 2.5:
                        over.append(o["price"])
                        under.append(u["price"])
        if not h:
            continue
        row = {
            "Date": pd.Timestamp(event["commence_time"]).strftime("%d/%m/%Y"),
            "HomeTeam": event["home_team"],
            "AwayTeam": event["away_team"],
            "home": home,
            "away": away,
            "AvgH": sum(h) / len(h),
            "AvgD": sum(d) / len(d),
            "AvgA": sum(a) / len(a),
            "books": len(h),
        }
        if over:
            row["Avg>2.5"] = sum(over) / len(over)
            row["Avg<2.5"] = sum(under) / len(under)
        out.append(row)
    return out


def load_odds(seasons: tuple[str, ...] = (*TRAIN_SEASONS, CURRENT_SEASON)) -> pd.DataFrame:
    """Per-side odds features for every stored match, plus the upcoming fixtures
    (attributed to the current season)."""
    parts = []
    for season in seasons:
        path = odds_path(season)
        if path.exists():
            raw = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
            parts.append(parse_football_data(raw, season))
    upcoming = EXTERNAL / "odds_fixtures.csv"
    if upcoming.exists():
        raw = pd.read_csv(upcoming, encoding="utf-8", encoding_errors="replace")
        parts.append(parse_football_data(raw, CURRENT_SEASON))
    live = EXTERNAL / f"odds_live_{CURRENT_SEASON}.csv"
    if live.exists():  # The Odds API prices, in the same columns; results files win
        raw = pd.read_csv(live)
        parts.append(parse_football_data(raw, CURRENT_SEASON))
    if not parts:
        return pd.DataFrame(columns=[*COLUMNS, "side", "team"])
    frame = pd.concat(parts, ignore_index=True)
    # a fixture that has since been played appears in both files: keep the result file's
    return frame.drop_duplicates(["season", "home", "away", "side"], keep="first")


def attach_odds(frame: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Join odds features onto per-fixture player rows by (season, home, away).

    A pairing meets once per season at each venue, so that key is unique and no date
    arithmetic across time zones is needed. Rows with no market are left NaN.
    """
    out = frame.copy()
    for column in ODDS_FEATURES:
        out[column] = np.nan
    if odds.empty or not {"season", "team", "opponent", "was_home"} <= set(out.columns):
        return out
    home = np.where(out["was_home"].astype(bool), out["team"], out["opponent"])
    away = np.where(out["was_home"].astype(bool), out["opponent"], out["team"])
    keys = pd.DataFrame(
        {
            "season": out["season"].to_numpy(),
            "home": home,
            "away": away,
            "team": out["team"].to_numpy(),
        }
    )
    lookup = odds[["season", "home", "away", "team", *ODDS_FEATURES]]
    merged = keys.merge(lookup, on=["season", "home", "away", "team"], how="left")
    for column in ODDS_FEATURES:
        out[column] = merged[column].to_numpy()
    matched = int(out["odds_win"].notna().sum())
    log.info("odds attached to %d of %d fixture rows", matched, len(out))
    return out

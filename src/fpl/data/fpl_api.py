"""Live data from the official FPL API, snapshotted at a point in time.

The API is mutable: prices drift daily, ``chance_of_playing`` and ``news`` are
rewritten as press conferences happen. So every pull is written to bronze as an
immutable, timestamped JSON file named for the gameweek it was taken ahead of, and
the pipeline reads snapshots -- never the API directly.

Three endpoints are used:

* ``bootstrap-static/`` -- players (price, availability, news, set-piece order),
  teams, and the gameweek calendar with deadlines.
* ``fixtures/`` -- every match of the season with kickoff and result.
* ``element-summary/{id}/`` -- one row per fixture per player for the current
  season, in exactly the shape the historical archive uses.

Network is only touched by ``fetch_snapshot``. Everything else works on the saved
JSON, which is what the tests use.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from fpl.config import BRONZE, CURRENT_SEASON, FPL_API_BASE, ensure_data_dirs
from fpl.entity.resolve import ELEMENT_TYPE_TO_POSITION

log = logging.getLogger(__name__)

LIVE_DIR = BRONZE / "live"
TIMEOUT = 30
WORKERS = 8

# The subset of bootstrap fields the pipeline uses. Everything else is dropped at
# capture time so snapshots stay small enough to keep forever.
ELEMENT_FIELDS = (
    "id", "code", "web_name", "first_name", "second_name", "team", "element_type",
    "now_cost", "selected_by_percent", "status", "chance_of_playing_next_round",
    "chance_of_playing_this_round", "news", "news_added", "total_points", "minutes",
    "form", "ep_next", "ep_this", "points_per_game", "expected_goals",
    "expected_assists", "expected_goals_conceded", "penalties_order",
    "corners_and_indirect_freekicks_order", "direct_freekicks_order",
    "cost_change_start", "transfers_in_event", "transfers_out_event", "value_season",
    "event_points",
)  # fmt: skip

# FPL availability codes. Anything not listed is treated as unavailable.
STATUS_AVAILABLE = "a"
STATUS_DOUBTFUL = "d"
STATUS_UNAVAILABLE = {"i", "s", "u", "n"}  # injured, suspended, unavailable, not in squad


# --------------------------------------------------------------------------- fetch


def _get(session: requests.Session, path: str) -> dict | list:
    response = session.get(f"{FPL_API_BASE}/{path}", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_snapshot(*, session: requests.Session | None = None, workers: int = WORKERS) -> dict:
    """Pull everything the pipeline needs from the live API into one dict."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "fpl-analyst/0.1 (+https://github.com)")

    bootstrap = _get(session, "bootstrap-static/")
    fixtures = _get(session, "fixtures/")
    elements = [{k: e.get(k) for k in ELEMENT_FIELDS} for e in bootstrap["elements"]]

    def summary(element_id: int) -> tuple[int, list]:
        return element_id, _get(session, f"element-summary/{element_id}/")["history"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        history = dict(pool.map(summary, [e["id"] for e in elements]))

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "total_players": bootstrap.get("total_players"),
        "events": [
            {
                k: e.get(k)
                for k in (
                    "id",
                    "name",
                    "deadline_time",
                    "finished",
                    "is_current",
                    "is_next",
                    "average_entry_score",
                    "highest_score",
                    "most_captained",
                )
            }  # fmt: skip
            for e in bootstrap["events"]
        ],
        "teams": [
            {k: t[k] for k in ("id", "name", "short_name", "strength")} for t in bootstrap["teams"]
        ],
        "elements": elements,
        "fixtures": [
            {
                k: f.get(k)
                for k in (
                    "id",
                    "event",
                    "kickoff_time",
                    "team_h",
                    "team_a",
                    "team_h_score",
                    "team_a_score",
                    "finished",
                    "team_h_difficulty",
                    "team_a_difficulty",
                )
            }  # fmt: skip
            for f in fixtures
        ],
        "history": {str(k): v for k, v in history.items()},
    }


# ------------------------------------------------------------------------ snapshots


def next_gameweek(snapshot: dict) -> dict:
    """The gameweek being predicted: the next one, or the current if none is marked."""
    events = snapshot["events"]
    upcoming = [e for e in events if e.get("is_next")] or [e for e in events if e.get("is_current")]
    if not upcoming:
        raise ValueError("snapshot has no current or next gameweek")
    return upcoming[0]


def snapshot_path(snapshot: dict, season: str = CURRENT_SEASON) -> Path:
    gameweek = next_gameweek(snapshot)["id"]
    stamp = snapshot["captured_at"][:19].replace(":", "").replace("-", "")
    return LIVE_DIR / f"snapshot_{season}_gw{gameweek}_{stamp}.json"


def save_snapshot(snapshot: dict, season: str = CURRENT_SEASON) -> Path:
    """Write an immutable snapshot. Never overwrites: a new pull is a new file."""
    ensure_data_dirs()
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(snapshot, season)
    if path.exists():
        raise FileExistsError(f"{path} already exists; snapshots are immutable")
    path.write_text(json.dumps(snapshot))
    log.info("saved %s", path.name)
    return path


def load_latest_snapshot(season: str = CURRENT_SEASON) -> dict:
    """The most recent snapshot on disk for the season."""
    candidates = sorted(LIVE_DIR.glob(f"snapshot_{season}_gw*.json"))
    if not candidates:
        raise FileNotFoundError(f"no live snapshot for {season}; run `fpl snapshot`")
    return json.loads(candidates[-1].read_text())


# ----------------------------------------------------------------- transformation


def _team_names(snapshot: dict) -> dict[int, str]:
    return {t["id"]: t["name"] for t in snapshot["teams"]}


def _fixture_gameweeks(snapshot: dict) -> dict[int, int | None]:
    return {f["id"]: f["event"] for f in snapshot["fixtures"]}


def snapshot_to_gameweeks(
    snapshot: dict, season: str = CURRENT_SEASON, *, before_gameweek: int | None = None
) -> pd.DataFrame:
    """Finished fixtures of the current season, one row per player per fixture.

    Output matches the archive's ``merged_gw.csv`` columns, so the same silver and
    feature code runs on it unchanged. ``xP`` is left null: the API does not keep
    its historical expected-points figures.

    ``before_gameweek`` drops everything from that gameweek onward, which is how a
    backtest reconstructs what was knowable at an earlier deadline.
    """
    names = _team_names(snapshot)
    by_id = {e["id"]: e for e in snapshot["elements"]}

    rows: list[dict] = []
    for element_id, history in snapshot["history"].items():
        element = by_id.get(int(element_id))
        if element is None:
            continue
        for h in history:
            if before_gameweek is not None and h["round"] >= before_gameweek:
                continue
            row = dict(h)
            row["element"] = int(element_id)
            row["code"] = element["code"]
            row["name"] = f"{element['first_name']} {element['second_name']}".strip()
            row["position"] = ELEMENT_TYPE_TO_POSITION.get(element["element_type"], "UNK")
            row["team"] = names[element["team"]]
            row["GW"] = h["round"]
            row["xP"] = None
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["code", "season", "GW", "kickoff_time", "round"])
    frame["season"] = season
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    for column in (
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "influence", "creativity", "threat", "ict_index",
    ):  # fmt: skip
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop(columns=["round"])


def availability_multiplier(status: str, chance: float | None) -> float:
    """How much of a projection survives the availability flag.

    FPL's own flag is authoritative for "will he be on the pitch": it is set by the
    game after press conferences and medical updates, which is exactly the news the
    rolling features cannot see. A stated percentage is used as-is; an unavailable
    status with no percentage is zero; a clean status is one.
    """
    if status in STATUS_UNAVAILABLE:
        return 0.0 if chance is None else max(0.0, min(1.0, chance / 100.0))
    if status == STATUS_DOUBTFUL:
        return 0.75 if chance is None else max(0.0, min(1.0, chance / 100.0))
    return 1.0


def snapshot_to_upcoming(
    snapshot: dict,
    season: str = CURRENT_SEASON,
    *,
    gameweeks: list[int] | None = None,
    as_of_gameweek: int | None = None,
) -> pd.DataFrame:
    """One row per player per fixture for the gameweeks to be projected.

    By default that is the next gameweek only. Passing ``gameweeks`` projects a run
    of them (the fixture list is known for the whole season), which is what the
    multi-week horizon and the transfer planner use. Rows carry fixture context
    (opponent, venue, kickoff), price and ownership, and the availability signal.
    Outcome columns are absent -- there is nothing to leak.

    ``as_of_gameweek`` reconstructs an earlier deadline for a backtest: price and
    ownership are taken from the player's fixture row in that gameweek rather than
    from today's registry, and availability is unknown (1.0) because the flags are
    not archived. The current registry is used only for identity.
    """
    names = _team_names(snapshot)
    if gameweeks is None:
        gameweeks = [as_of_gameweek] if as_of_gameweek else [next_gameweek(snapshot)["id"]]
    wanted = set(gameweeks)
    fixtures = [f for f in snapshot["fixtures"] if f["event"] in wanted]

    # The archive stores ownership as a manager COUNT; bootstrap only offers a
    # percentage. Feeding the percentage through as if it were a count put every live
    # player at the bottom of the ownership scale and depressed all projections.
    # Prefer the count from the player's most recent fixture row; fall back to
    # percentage x total managers.
    latest_selected: dict[int, float] = {}
    as_of_value: dict[int, float] = {}
    for element_id, history in snapshot.get("history", {}).items():
        eid = int(element_id)
        if as_of_gameweek:
            prior = [h for h in history if h["round"] < as_of_gameweek]
            at = [h for h in history if h["round"] == as_of_gameweek]
            if at:
                as_of_value[eid] = float(at[0].get("value") or 0)
                latest_selected[eid] = float(at[0].get("selected") or 0)
            elif prior:
                as_of_value[eid] = float(prior[-1].get("value") or 0)
                latest_selected[eid] = float(prior[-1].get("selected") or 0)
        elif history:
            latest_selected[eid] = float(history[-1].get("selected") or 0)
    total_players = float(snapshot.get("total_players") or 0)
    by_team: dict[int, list[dict]] = {}
    for f in fixtures:
        by_team.setdefault(f["team_h"], []).append(f)
        by_team.setdefault(f["team_a"], []).append(f)

    rows = []
    for e in snapshot["elements"]:
        position = ELEMENT_TYPE_TO_POSITION.get(e["element_type"])
        if position not in ("GK", "DEF", "MID", "FWD"):
            continue
        if as_of_gameweek:
            value = as_of_value.get(e["id"])
            if value is None:
                continue  # not in the game at that deadline
            availability, status, chance, news = 1.0, "a", None, ""
        else:
            value = e["now_cost"]
            status, chance, news = e["status"], e["chance_of_playing_next_round"], e["news"] or ""
            availability = availability_multiplier(status, chance)
        for f in by_team.get(e["team"], []):
            home = f["team_h"] == e["team"]
            rows.append(
                {
                    "season": season,
                    "GW": f["event"],
                    "element": e["id"],
                    "code": e["code"],
                    "name": f"{e['first_name']} {e['second_name']}".strip(),
                    "web_name": e["web_name"],
                    "position": position,
                    "team": names[e["team"]],
                    "fixture": f["id"],
                    "opponent_team": f["team_a"] if home else f["team_h"],
                    "was_home": home,
                    "kickoff_time": f["kickoff_time"],
                    "value": value,
                    "selected": latest_selected.get(
                        e["id"], float(e["selected_by_percent"] or 0) / 100.0 * total_players
                    ),
                    "selected_by_percent": float(e["selected_by_percent"] or 0),
                    "status": status,
                    "chance_of_playing": chance,
                    "news": news,
                    "availability": availability,
                    "fpl_ep_next": float(e["ep_next"] or 0) if not as_of_gameweek else 0.0,
                }
            )

    frame = pd.DataFrame(rows)
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    log.info(
        "GW%s: %d player-fixture rows, %d flagged as doubtful or out",
        ",".join(str(g) for g in sorted(wanted)),
        len(frame),
        int((frame["availability"] < 1).sum()),
    )
    return frame


def gameweek_averages(snapshot: dict) -> dict[int, dict]:
    """Average and top manager scores per finished gameweek, from the API calendar."""
    return {
        e["id"]: {
            "average": e.get("average_entry_score") or 0,
            "highest": e.get("highest_score"),
            "deadline": e.get("deadline_time"),
        }
        for e in snapshot["events"]
        if e.get("finished")
    }


def team_id_map(snapshot: dict, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """The live season's (season, team_id) -> name mapping, same shape as the archive's."""
    return pd.DataFrame(
        [{"season": season, "team_id": t["id"], "team_name": t["name"]} for t in snapshot["teams"]]
    )

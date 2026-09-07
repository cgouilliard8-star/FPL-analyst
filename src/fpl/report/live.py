"""Produce the live dashboard payload: every player's projection for the next
gameweek, the best squad money can buy, and the metadata the page needs to say how
fresh it is.

The browser does the team-rating arithmetic itself from this file, so the page works
as static hosting with nothing behind it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fpl.config import CURRENT_SEASON, PROJECT_ROOT
from fpl.data.fpl_api import load_latest_snapshot
from fpl.models.combine import CONTRIBUTIONS
from fpl.models.predict import project_gameweek
from fpl.optimise.squad import pick_squad

log = logging.getLogger(__name__)

SITE_DATA = PROJECT_ROOT / "site" / "data"
LIVE_PATH = SITE_DATA / "live.json"


def _optimal(projections: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    pool = projections.rename(columns={"expected_points": "projected"}).assign(
        price_tenths=lambda d: (d["price"] * 10).round().astype(int), actual=0.0
    )
    squad = pick_squad(pool[["code", "position", "team", "price_tenths", "projected", "actual"]])
    points = float(
        squad.loc[squad["is_starter"], "projected"].sum()
        + squad.loc[squad["is_captain"], "projected"].sum()
    )
    return squad, points


SEASON_STATS = (
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "bonus", "bps", "saves", "starts", "yellow_cards", "red_cards", "own_goals",
    "penalties_missed", "penalties_saved", "expected_goals", "expected_assists",
    "expected_goals_conceded", "defensive_contribution", "influence", "creativity",
    "threat", "ict_index",
)  # fmt: skip


def _season_totals(snapshot: dict) -> dict[int, dict]:
    """Season-to-date totals per player, summed from the fixture rows.

    The picker shows these the way the FPL app does; they are context for the
    manager, not model inputs.
    """
    totals: dict[int, dict] = {}
    by_id = {e["id"]: e for e in snapshot["elements"]}
    for element_id, history in snapshot.get("history", {}).items():
        element = by_id.get(int(element_id))
        if element is None:
            continue
        acc = dict.fromkeys(SEASON_STATS, 0.0)
        last_gw_points = 0
        for h in history:
            for stat in SEASON_STATS:
                acc[stat] += float(h.get(stat) or 0)
            last_gw_points = int(h.get("total_points") or 0)
        acc["last_gw_points"] = last_gw_points
        acc["form"] = float(element.get("form") or 0)
        acc["points_per_game"] = float(element.get("points_per_game") or 0)
        acc["games"] = len(history)
        totals[element["code"]] = {k: round(v, 2) for k, v in acc.items()}
    return totals


def _player_record(row: pd.Series) -> dict:
    return {
        "code": int(row["code"]),
        "id": int(row["element"]),
        "name": row["full_name"],
        "web_name": row["web_name"],
        "position": row["position"],
        "team": row["team"],
        "price": round(float(row["price"]), 1),
        "ep": round(float(row["expected_points"]), 2),
        "raw_ep": round(float(row["raw_expected_points"]), 2),
        "fpl_ep": round(float(row["fpl_ep_next"]), 1),
        "availability": round(float(row["availability"]), 2),
        "status": row["status"],
        "chance": None if pd.isna(row["chance_of_playing"]) else int(row["chance_of_playing"]),
        "news": row["news"] or "",
        "p60": round(float(row["p_60"]), 2),
        "opponent": row["opponent"],
        "home": bool(row["is_home"]),
        "fixtures": int(row["fixtures_this_gw"]),
        "owned_pct": round(float(row["selected_by_percent"]), 1),
        "breakdown": {term: round(float(row[term]), 2) for term in CONTRIBUTIONS},
    }


def build_live(snapshot: dict | None = None, *, explain: bool = True) -> dict:
    snapshot = load_latest_snapshot() if snapshot is None else snapshot
    projections = project_gameweek(snapshot)
    squad, optimal_points = _optimal(projections)

    rationales: dict[int, str] = {}
    if explain:
        from fpl.explain.generate import explain_frame

        starters = squad.loc[squad["is_starter"], "code"]
        rows = projections[projections["code"].isin(starters)]
        explained = explain_frame(rows)
        rationales = dict(zip(explained["code"].astype(int), explained["rationale"], strict=True))

    totals = _season_totals(snapshot)
    players = []
    for _, row in projections.iterrows():
        record = _player_record(row)
        record["season"] = totals.get(record["code"], {})
        if record["code"] in rationales:
            record["rationale"] = rationales[record["code"]]
        players.append(record)

    gameweek = int(projections["gameweek"].iloc[0])
    return {
        "meta": {
            "season": CURRENT_SEASON,
            "gameweek": gameweek,
            "last_gameweek": gameweek - 1,
            "deadline": projections["deadline"].iloc[0],
            "captured_at": snapshot["captured_at"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "players": len(players),
            "flagged": int((projections["availability"] < 1).sum()),
            "optimal_points": round(optimal_points, 2),
            "clubs": {t["name"]: t["short_name"] for t in snapshot["teams"]},
        },
        "optimal": {
            "starters": [int(c) for c in squad.loc[squad["is_starter"], "code"]],
            "bench": [int(c) for c in squad.loc[~squad["is_starter"], "code"]],
            "captain": int(squad.loc[squad["is_captain"], "code"].iloc[0]),
            "spend": round(int(squad["price_tenths"].sum()) / 10, 1),
            "projected": round(optimal_points, 2),
        },
        "players": players,
    }


def write_live(payload: dict, path: Path = LIVE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    log.info("wrote %s (%d players)", path, len(payload["players"]))
    return path

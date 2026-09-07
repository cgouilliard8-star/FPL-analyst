"""Produce the live dashboard payload: every player's projection for the next
gameweek and the four after it, the best squad money can buy, the club strength table
the projections lean on, the season replay, and the metadata the page needs to say
how fresh it is.

The browser does the team-rating arithmetic itself from this file, so the page works
as static hosting with nothing behind it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fpl.config import CURRENT_SEASON, PROJECT_ROOT, TRANSFER_HIT
from fpl.data.fpl_api import gameweek_averages, load_latest_snapshot
from fpl.models.combine import CONTRIBUTIONS
from fpl.models.predict import HORIZON_WEIGHTS, MAX_HORIZON, project_horizon
from fpl.optimise.squad import pick_squad

log = logging.getLogger(__name__)

SITE_DATA = PROJECT_ROOT / "site" / "data"
LIVE_PATH = SITE_DATA / "live.json"
SEASON_SIM_PATH = PROJECT_ROOT / "data" / "gold" / "season_sim.json"


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


def _fixture_record(row: pd.Series) -> dict:
    def _int(v):
        return None if pd.isna(v) else int(v)

    def _flt(v, nd=2):
        return None if pd.isna(v) else round(float(v), nd)

    return {
        "gw": int(row["GW"]),
        "opp": row["all_opponents"],
        "home": bool(row["is_home"]),
        "n": int(row["fixtures_this_gw"]),
        "ep": _flt(row["expected_points"]),
        "raw_ep": _flt(row["raw_expected_points"]),
        "avail": _flt(row["availability"]),
        "p60": _flt(row["p_60"]),
        "opp_att": _int(row["opp_att_rank"]),
        "opp_def": _int(row["opp_def_rank"]),
        "opp_xg": _flt(row["opp_xg_r5"]),
        "opp_xgc": _flt(row["opp_xgc_r5"]),
        "w": float(row["weight"]),
        "breakdown": {term: _flt(row[term]) for term in CONTRIBUTIONS},
    }


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
        "ep1": round(float(row["ep1"]), 2),
        "ep3": round(float(row["ep3"]), 2),
        "ep5": round(float(row["ep5"]), 2),
        "raw_ep": round(float(row["raw_expected_points"]), 2),
        "own_att": None if pd.isna(row["own_att_rank"]) else int(row["own_att_rank"]),
        "own_def": None if pd.isna(row["own_def_rank"]) else int(row["own_def_rank"]),
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


def _team_record(row: pd.Series) -> dict:
    return {
        "team": row["team"],
        "att_rank": int(row["att_rank"]),
        "def_rank": int(row["def_rank"]),
        "xg_r5": round(float(row["team_xg_r5"]), 2),
        "xgc_r5": round(float(row["team_xgc_r5"]), 2),
        "xg_r10": round(float(row["team_xg_r10"]), 2),
        "xgc_r10": round(float(row["team_xgc_r10"]), 2),
    }


def load_season_sim(path: Path = SEASON_SIM_PATH) -> dict | None:
    """The cached season replay, if one has been run (it takes minutes, so it is not
    recomputed on every refresh)."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_live(
    snapshot: dict | None = None, *, explain: bool = True, season_sim: dict | None = None
) -> dict:
    snapshot = load_latest_snapshot() if snapshot is None else snapshot
    projections, fixtures, teams = project_horizon(snapshot, horizon=MAX_HORIZON)
    squad, optimal_points = _optimal(projections)
    optimal_by = {}
    for metric in ("ep1", "ep3", "ep5"):
        _, pts = _optimal(
            projections.drop(columns=["expected_points"]).rename(
                columns={metric: "expected_points"}
            )
        )
        optimal_by[metric] = round(pts, 2)
    runs = {int(code): group for code, group in fixtures.groupby("code")}

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
        record["fixtures"] = [_fixture_record(f) for _, f in runs[record["code"]].iterrows()]
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
            "optimal_by": optimal_by,
            "horizon_weights": list(HORIZON_WEIGHTS),
            "transfer_hit": TRANSFER_HIT,
            "clubs": {t["name"]: t["short_name"] for t in snapshot["teams"]},
            "averages": gameweek_averages(snapshot),
        },
        "teams": [_team_record(t) for _, t in teams.iterrows()],
        "backtest": load_season_sim() if season_sim is None else season_sim,
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

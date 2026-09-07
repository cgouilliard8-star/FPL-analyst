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

from fpl.config import CURRENT_SEASON, HORIZON_WEIGHTS, MAX_HORIZON, PROJECT_ROOT, TRANSFER_HIT
from fpl.data.fpl_api import gameweek_averages, load_latest_snapshot, snapshot_to_gameweeks
from fpl.data.schedule import EUROPEAN, load_schedule
from fpl.data.silver import load_silver
from fpl.entity.resolve import canonical_team, load_team_aliases
from fpl.models.combine import CONTRIBUTIONS
from fpl.models.predict import project_horizon
from fpl.optimise.rate import rate_squad
from fpl.optimise.squad import pick_squad
from fpl.report.insights import fixture_insights, match_results

log = logging.getLogger(__name__)

SITE_DATA = PROJECT_ROOT / "site" / "data"
LIVE_PATH = SITE_DATA / "live.json"
SEASON_SIM_PATH = PROJECT_ROOT / "data" / "gold" / "season_sim.json"


BUDGETS = (100, 105, 110, 115, 120, 125, 130)  # £m: the page picks the nearest at or below yours


def _optimal(projections: pd.DataFrame, budget: float = 100.0) -> tuple[pd.DataFrame, float]:
    pool = projections.rename(columns={"expected_points": "projected"}).assign(
        price_tenths=lambda d: (d["price"] * 10).round().astype(int), actual=0.0
    )
    squad = pick_squad(
        pool[["code", "position", "team", "price_tenths", "projected", "actual"]],
        budget_tenths=int(round(budget * 10)),
    )
    points = float(
        squad.loc[squad["is_starter"], "projected"].sum()
        + squad.loc[squad["is_captain"], "projected"].sum()
    )
    return squad, points


def _optimal_record(
    squad: pd.DataFrame, projections: pd.DataFrame, fixtures: pd.DataFrame, metric: str
) -> dict:
    """The MILP squad, rated the way the page rates a manager's squad (per-gameweek
    elevens plus bench cover) so the two are comparable."""
    codes = [int(c) for c in squad["code"]]
    rated = rate_squad(
        projections[projections["code"].isin(codes)], metric=metric, fixtures=fixtures
    )
    return {
        "starters": rated["starters"],
        "bench": rated["bench"],
        "captain": rated["captain"],
        "vice": rated["vice"],
        "formation": rated["formation"],
        "spend": round(int(squad["price_tenths"].sum()) / 10, 1),
        "projected": rated["points"],
        "next_points": rated["next_points"],
    }


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
        "busy": _int(row.get("other_games_7d", 0)) or 0,  # cup/European games in the week before
        "euro": bool(row.get("euro_midweek", 0)),
        "next4": bool(row.get("other_game_next_4d", 0)),
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
        "pens": _opt_int(row.get("penalties_order")),
        "corners": _opt_int(row.get("corners_order")),
        "freekicks": _opt_int(row.get("freekicks_order")),
        "transfers_in": int(row.get("transfers_in_event") or 0),
        "transfers_out": int(row.get("transfers_out_event") or 0),
        "price_change": round(float(row.get("cost_change_start") or 0) / 10, 1),
        "value_season": float(row.get("value_season") or 0),
        "minutes_share": round(float(row.get("minutes_share_r5") or 0), 2),
    }


def _opt_int(v) -> int | None:
    return None if v is None or pd.isna(v) else int(v)


def _band(rank: int, n: int = 20) -> int:
    """1 (easiest) to 5 (hardest) from a rank where 1 is the strongest."""
    return max(1, min(5, 5 - (rank - 1) * 5 // n))


def _team_record(row: pd.Series) -> dict:
    att, dfn = int(row["att_rank"]), int(row["def_rank"])
    return {
        "team": row["team"],
        "att_rank": att,
        "def_rank": dfn,
        "att_score": round(float(row["att_score"]), 2),
        "def_score": round(float(row["def_score"]), 2),
        # how hard this club is to face: for defenders, how much it scores; for
        # attackers, how little it concedes (rank 1 = tightest defence = hardest)
        "hard_for_def": _band(att),
        "hard_for_att": _band(dfn),
        "xg_ew": round(float(row["team_xg_ew"]), 2),
        "xgc_ew": round(float(row["team_xgc_ew"]), 2),
        "xg_r5": round(float(row["team_xg_r5"]), 2),
        "xgc_r5": round(float(row["team_xgc_r5"]), 2),
        "xg_r38": round(float(row["team_xg_r38"]), 2),
        "xgc_r38": round(float(row["team_xgc_r38"]), 2),
    }


def _club_schedules(snapshot: dict, season: str, first_gw: int, horizon: int = 8) -> dict:
    """Each club's coming matches in every competition: the league fixtures from the
    snapshot and the cup/European ties from the stored schedule."""
    aliases = load_team_aliases()
    names = {t["id"]: canonical_team(t["name"], aliases) for t in snapshot["teams"]}
    out: dict[str, dict] = {n: {"europe": None, "matches": []} for n in names.values()}
    for f in snapshot["fixtures"]:
        if not f.get("event") or f["event"] < first_gw or f["event"] >= first_gw + horizon:
            continue
        h, a = names[f["team_h"]], names[f["team_a"]]
        when = f.get("kickoff_time")
        out[h]["matches"].append(
            {"comp": "PL", "gw": f["event"], "opp": a, "home": True, "at": when}
        )
        out[a]["matches"].append(
            {"comp": "PL", "gw": f["event"], "opp": h, "home": False, "at": when}
        )
    schedule = load_schedule((season,))
    if not schedule.empty:
        now = pd.Timestamp.now(tz="UTC")
        for r in schedule.itertuples(index=False):
            for me, opp, home in ((r.home, r.away, True), (r.away, r.home, False)):
                if me not in out:
                    continue
                if r.competition in EUROPEAN:
                    out[me]["europe"] = r.competition
                if r.kickoff_time >= now - pd.Timedelta(days=1):
                    out[me]["matches"].append(
                        {"comp": r.competition, "gw": None, "opp": opp, "home": home,
                         "at": r.kickoff_time.strftime("%Y-%m-%dT%H:%M:%SZ")}
                    )  # fmt: skip
    for club in out.values():
        club["matches"].sort(key=lambda m: m["at"] or "")
        club["matches"] = club["matches"][:14]
    return out


def _next_fixtures(snapshot: dict, gameweek: int) -> list[dict]:
    aliases = load_team_aliases()
    names = {t["id"]: canonical_team(t["name"], aliases) for t in snapshot["teams"]}
    return [
        {"gw": gameweek, "home": names[f["team_h"]], "away": names[f["team_a"]],
         "kickoff": f.get("kickoff_time")}
        for f in sorted(snapshot["fixtures"], key=lambda f: f.get("kickoff_time") or "")
        if f.get("event") == gameweek
    ]  # fmt: skip


def _results_to_date(snapshot: dict, season: str) -> pd.DataFrame:
    archive = load_silver()
    live = snapshot_to_gameweeks(snapshot, season)
    if not live.empty:
        aliases = load_team_aliases()
        live = live.copy()
        live["team"] = live["team"].map(lambda n: canonical_team(n, aliases) or n)
    frames = [archive] + ([live] if not live.empty else [])
    return match_results(pd.concat(frames, ignore_index=True, sort=False))


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
    # The reference squads: the best the money can buy at each budget and horizon,
    # rated exactly as a manager's squad is so the rating is like for like.
    optimal_by: dict[str, float] = {}
    optimal_squads: dict[str, dict[str, dict]] = {}
    for metric in ("ep1", "ep3", "ep5"):
        on_metric = projections.drop(columns=["expected_points"]).rename(
            columns={metric: "expected_points"}
        )
        optimal_squads[metric] = {}
        for budget in BUDGETS:
            best, _ = _optimal(on_metric, budget)
            optimal_squads[metric][str(budget)] = _optimal_record(
                best, projections, fixtures, metric
            )
        optimal_by[metric] = optimal_squads[metric]["100"]["projected"]
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
    results = _results_to_date(snapshot, CURRENT_SEASON)
    insights = fixture_insights(_next_fixtures(snapshot, gameweek), results)
    return {
        "meta": {
            "schema": 2,
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
        "schedules": _club_schedules(snapshot, CURRENT_SEASON, gameweek),
        "insights": insights,
        "optimal_squads": optimal_squads,
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

"""Replay the season from scratch: build a squad at GW1 with the information a
manager had at that deadline, then play each gameweek with FPL's transfer rules and
score it against what actually happened.

This is the honest test of the whole pipeline -- projections, horizon weighting,
squad optimiser and transfer suggester together -- rather than of any one model.
Every gameweek is projected ``as_of`` its own deadline, so nothing later leaks in.

Transfer rules followed: one free transfer per gameweek from GW2, bankable up to
five, and none in GW1 (the squad is built there). The planner never takes a points
hit; it makes its best move when that move is worth at least ``MIN_GAIN`` weighted
points over the horizon, and rolls the transfer otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from fpl.config import CURRENT_SEASON, XI_MAX, XI_MIN
from fpl.data.fpl_api import snapshot_to_gameweeks
from fpl.models.predict import project_horizon
from fpl.optimise.rate import _best_eleven, _to_players, captain_order, suggest_transfers
from fpl.optimise.squad import pick_squad

log = logging.getLogger(__name__)

MAX_FREE_TRANSFERS = 5
MIN_GAIN = 1.0  # weighted horizon points a move must add before it is worth making
BENCH_ORDER_METRIC = "expected_points"


@dataclass
class GameweekResult:
    gameweek: int
    points: int
    projected: float
    average: int | None
    highest: int | None
    free_transfers: int
    transfers: list[dict]
    squad: list[int]
    starters: list[int]
    captain: int
    bench: list[int]
    value: float
    bank: float
    autosubs: list[dict] = field(default_factory=list)


def _actuals(snapshot: dict, season: str) -> pd.DataFrame:
    """Points and minutes each player actually recorded per gameweek."""
    rows = snapshot_to_gameweeks(snapshot, season)
    return rows.groupby(["code", "GW"])[["total_points", "minutes"]].sum()


def _initial_squad(players: pd.DataFrame, metric: str) -> tuple[list[int], float]:
    pool = players.assign(
        projected=players[metric],
        price_tenths=(players["price"] * 10).round().astype(int),
        actual=0.0,
    )[["code", "position", "team", "price_tenths", "projected", "actual"]]
    squad = pick_squad(pool)
    codes = [int(c) for c in squad["code"]]
    spend = int(squad["price_tenths"].sum()) / 10
    return codes, spend


def _formation_ok(counts: dict[str, int]) -> bool:
    return all(XI_MIN[p] <= counts.get(p, 0) <= XI_MAX[p] for p in XI_MIN) and (
        sum(counts.values()) == 11
    )


def _play(
    squad_rows: list[dict], actual: pd.DataFrame, gameweek: int
) -> tuple[int, list[int], int, list[int], list[dict], float]:
    """Pick the eleven and captain on projections, then score what happened, with
    FPL's automatic substitutions for anyone who did not play."""
    eleven = _best_eleven(squad_rows)
    by_code = {int(p["code"]): p for p in squad_rows}

    def played(code: int) -> bool:
        return (code, gameweek) in actual.index and actual.loc[(code, gameweek), "minutes"] > 0

    def points(code: int) -> int:
        return (
            int(actual.loc[(code, gameweek), "total_points"])
            if (code, gameweek) in actual.index
            else 0
        )

    starters = list(eleven.starters)
    bench_gk = [c for c in eleven.bench if by_code[c]["position"] == "GK"]
    bench_out = sorted(
        (c for c in eleven.bench if by_code[c]["position"] != "GK"),
        key=lambda c: -by_code[c][BENCH_ORDER_METRIC],
    )
    autosubs: list[dict] = []
    for missing in [c for c in starters if not played(c)]:
        pos = by_code[missing]["position"]
        candidates = bench_gk if pos == "GK" else bench_out
        for sub in list(candidates):
            if not played(sub):
                continue
            trial = [c for c in starters if c != missing] + [sub]
            counts: dict[str, int] = {}
            for c in trial:
                counts[by_code[c]["position"]] = counts.get(by_code[c]["position"], 0) + 1
            if _formation_ok(counts):
                starters = trial
                candidates.remove(sub)
                autosubs.append({"out": missing, "in": sub})
                break

    captain = eleven.captain
    if not played(captain):  # armband passes to the vice: next in armband order
        others = [by_code[c] for c in eleven.starters if c != captain]
        captain = int(captain_order(others, lambda p: p["expected_points"])[0]["code"])
    total = sum(points(c) for c in starters) + (points(captain) if captain in starters else 0)
    bench = [c for c in by_code if c not in starters]
    return total, starters, captain, bench, autosubs, eleven.points


def simulate_season(
    snapshot: dict,
    season: str = CURRENT_SEASON,
    *,
    gameweeks: tuple[int, ...] = (1, 2, 3),
    horizon: int = 5,
    metric: str = "ep5",
    min_gain: float = MIN_GAIN,
) -> dict:
    """Build a team at ``gameweeks[0]`` and play through the rest; return the log."""
    actual = _actuals(snapshot, season)
    events = {e["id"]: e for e in snapshot["events"]}
    squad: list[int] = []
    bank = 0.0
    free = 0
    results: list[GameweekResult] = []
    prices: dict[int, float] = {}

    for gw in gameweeks:
        players, fixtures, _ = project_horizon(snapshot, season, horizon=horizon, as_of_gameweek=gw)
        players = players[players["price"] > 0].reset_index(drop=True)
        transfers: list[dict] = []
        if not squad:
            squad, spend = _initial_squad(players, metric)
            bank = round(100.0 - spend, 1)
            log.info("GW%s: built squad for £%.1fm, £%.1fm in the bank", gw, spend, bank)
        else:
            free = min(MAX_FREE_TRANSFERS, free + 1)
            # Players who left the game keep their last known price so the squad is
            # still fifteen; they project nothing and will be sold.
            own = players[players["code"].isin(squad)]
            missing = set(squad) - set(own["code"])
            if missing:
                filler = pd.DataFrame(
                    [{**{c: None for c in players.columns}, "code": c, "price": prices.get(c, 4.0)}
                     for c in missing]
                )  # fmt: skip
                own = pd.concat([own, filler], ignore_index=True)
            value = float(own["price"].sum())
            moves = suggest_transfers(
                own.fillna({metric: 0.0, "expected_points": 0.0, "availability": 0.0}),
                players,
                bank=bank,
                top_n=3,
                metric=metric,
                team_value=value,
                free_transfers=free,
                fixtures=fixtures,
            )
            moves = [m for m in moves if m["transfers"] <= free and m["gain"] >= min_gain]
            if moves:
                best = moves[0]
                for out_code, in_code in zip(best["out"], best["in"], strict=True):
                    squad[squad.index(out_code)] = in_code
                bank = round(bank - best["cost_change"], 1)
                free -= best["transfers"]
                names = dict(zip(players["code"], players["web_name"], strict=True))
                transfers.append(
                    {
                        "out": best["out"],
                        "in": best["in"],
                        "out_names": [names.get(c, "?") for c in best["out"]],
                        "in_names": [names.get(c, "?") for c in best["in"]],
                        "gain": best["gain"],
                        "cost_change": best["cost_change"],
                    }
                )
                log.info("GW%s: %s -> %s (+%.2f)", gw, transfers[0]["out_names"],
                         transfers[0]["in_names"], best["gain"])  # fmt: skip
            else:
                log.info("GW%s: rolled the transfer (%d banked)", gw, free)

        own = players[players["code"].isin(squad)]
        prices.update(dict(zip(own["code"].astype(int), own["price"].astype(float), strict=True)))
        rows = _to_players(own)  # rated on the next gameweek for lineup and captain
        points, starters, captain, bench, autosubs, projected = _play(rows, actual, gw)
        event = events.get(gw, {})
        results.append(
            GameweekResult(
                gameweek=gw,
                points=points,
                projected=round(projected, 2),
                average=event.get("average_entry_score"),
                highest=event.get("highest_score"),
                free_transfers=free,
                transfers=transfers,
                squad=list(squad),
                starters=starters,
                captain=captain,
                bench=bench,
                value=round(float(own["price"].sum()), 1),
                bank=bank,
                autosubs=autosubs,
            )
        )
        log.info("GW%s: %d points (average %s, highest %s)", gw, points,
                 event.get("average_entry_score"), event.get("highest_score"))  # fmt: skip

    return {
        "season": season,
        "gameweeks": [r.__dict__ for r in results],
        "total": sum(r.points for r in results),
        "average_total": sum(r.average or 0 for r in results),
        "highest_total": sum(r.highest or 0 for r in results),
        "rules": {
            "free_transfer_per_gameweek": 1,
            "max_banked": MAX_FREE_TRANSFERS,
            "hits_taken": 0,
            "min_gain": min_gain,
            "metric": metric,
            "horizon": horizon,
        },
        "note": (
            "Injury and suspension flags are not archived, so every player was treated "
            "as available at each past deadline; a live manager would have known better."
        ),
    }


__all__ = ["simulate_season", "GameweekResult"]

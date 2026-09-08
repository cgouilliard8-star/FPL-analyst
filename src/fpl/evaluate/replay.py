"""Replay a whole archived season under FPL's transfer rules: the model as manager
against the crowd as manager, on real points.

Three gameweeks say nothing; thirty-three say something. This uses the cached
walk-forward predictions (every one made before its gameweek, refit weekly) and
plays the season from gameweek 6 with the same rules as ``season_sim``: build a
£100m squad, then one free transfer a week, bankable to five, never a hit, lineup
and captain re-picked each week, scored with automatic substitutions.

Two managers play the same season:

* **model** -- ranks players by the projection.
* **crowd** -- ranks players by ownership at that deadline: the template team, which
  tracks the average manager closely because it *is* what the average manager owns.

The archive carries no per-gameweek average score, so the crowd manager is the
stand-in for "the average"; on the same rules and the same squad-building code the
comparison is like for like.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from fpl.config import PROJECT_ROOT
from fpl.evaluate.season_sim import MAX_FREE_TRANSFERS, MIN_GAIN, _play
from fpl.features.build import load_features
from fpl.optimise.rate import _to_players, suggest_transfers
from fpl.optimise.squad import pick_squad

log = logging.getLogger(__name__)

PREDICTIONS = PROJECT_ROOT / "data" / "gold" / "predictions"


@dataclass
class Week:
    gameweek: int
    points: int
    transfers: list[str]
    captain: str


def _pool_for(gameweek: int, frame: pd.DataFrame, manager: str) -> pd.DataFrame:
    rows = frame[frame["GW"] == gameweek].copy()
    score = rows["predicted"] if manager == "model" else rows["ownership"] / 1e5
    rows["expected_points"] = score.fillna(0.0)
    rows["availability"] = 1.0
    return rows[
        ["code", "web_name", "position", "team", "price", "expected_points", "availability"]
    ].drop_duplicates("code")


def _pick_initial(pool: pd.DataFrame) -> list[int]:
    milp = pool.assign(
        projected=pool["expected_points"],
        price_tenths=(pool["price"] * 10).round().astype(int),
        actual=0.0,
    )[["code", "position", "team", "price_tenths", "projected", "actual"]]
    return [int(c) for c in pick_squad(milp)["code"]]


def replay_season(
    season: str = "2024-25",
    model: str = "component",
    *,
    manager: str = "model",
    first_gameweek: int = 6,
    min_gain: float = MIN_GAIN,
) -> dict:
    """Play ``season`` from ``first_gameweek`` as ``manager``; return the log."""
    preds = pd.read_parquet(PREDICTIONS / f"{model}_{season}.parquet")
    feats = load_features()
    feats = feats[feats["season"] == season][
        ["code", "GW", "full_name", "position", "team", "price", "ownership", "minutes", "total_points"]
    ].rename(columns={"full_name": "web_name"})  # fmt: skip
    frame = feats.merge(
        preds.rename(columns={"gameweek": "GW"})[["code", "GW", "predicted"]],
        on=["code", "GW"],
        how="left",
    )
    frame = frame[frame["price"] > 0]
    actual = frame.groupby(["code", "GW"])[["total_points", "minutes"]].sum()
    gameweeks = sorted(int(g) for g in preds["gameweek"].unique() if g >= first_gameweek)

    squad: list[int] = []
    bank, free = 0.0, 0
    weeks: list[Week] = []
    last_seen: dict[int, dict] = {}  # a player's identity/price when he last had a fixture

    def complete(pool: pd.DataFrame) -> pd.DataFrame:
        """The squad as fifteen rows even in a blank week: absent players keep their
        last known position, club and price and project nothing."""
        own = pool[pool["code"].isin(squad)]
        missing = [c for c in squad if c not in set(own["code"])]
        if missing:
            filler = pd.DataFrame(
                [{**last_seen.get(c, {"web_name": "?", "position": "MID", "team": "?", "price": 4.0}),
                  "code": c, "expected_points": 0.0, "availability": 0.0} for c in missing]
            )  # fmt: skip
            own = pd.concat([own, filler], ignore_index=True)
        return own

    for gw in gameweeks:
        pool = _pool_for(gw, frame, manager)
        for r in pool.itertuples(index=False):
            last_seen[int(r.code)] = {
                "web_name": r.web_name,
                "position": r.position,
                "team": r.team,
                "price": float(r.price),
            }
        names = {c: v["web_name"] for c, v in last_seen.items()}
        moves: list[str] = []
        if not squad:
            squad = _pick_initial(pool)
            bank = round(100.0 - float(pool.set_index("code").loc[squad, "price"].sum()), 1)
        else:
            free = min(MAX_FREE_TRANSFERS, free + 1)
            own = complete(pool)
            value = float(own["price"].sum())
            options = suggest_transfers(
                own, pool, bank=bank, top_n=3, team_value=value, free_transfers=free
            )
            options = [m for m in options if m["transfers"] <= free and m["gain"] >= min_gain]
            if options:
                best = options[0]
                for out_code, in_code in zip(best["out"], best["in"], strict=True):
                    squad[squad.index(out_code)] = in_code
                bank = round(bank - best["cost_change"], 1)
                free -= best["transfers"]
                moves = [
                    f"{names.get(o, '?')} -> {names.get(i, '?')}"
                    for o, i in zip(best["out"], best["in"], strict=True)
                ]
        rows = _to_players(complete(pool))
        points, _, captain, _, _, _ = _play(rows, actual, gw)
        weeks.append(Week(gw, points, moves, names.get(captain, "?")))
    total = sum(w.points for w in weeks)
    return {
        "season": season,
        "manager": manager,
        "gameweeks": [w.__dict__ for w in weeks],
        "total": total,
        "per_gameweek": round(total / len(weeks), 1) if weeks else 0.0,
    }


__all__ = ["replay_season"]

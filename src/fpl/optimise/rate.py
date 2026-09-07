"""Rate a manager's own squad and find the moves that lift it most.

Given the fifteen a manager actually owns, this answers two questions: how many
points is that squad projected to score with its best legal eleven, and which single
transfers raise that number the most. The gain of a transfer is measured on the whole
squad -- best eleven and captain re-picked after the swap -- not on the two players
in isolation, because a new signing can change who starts and who wears the armband.

The core works on plain dicts rather than DataFrames: a squad is fifteen rows, a
transfer search evaluates a few thousand candidate squads, and the same logic is
mirrored in the browser where there is no pandas at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import pandas as pd

from fpl.config import MAX_PER_CLUB, SQUAD_QUOTA, SQUAD_SIZE, XI_MAX, XI_MIN, XI_SIZE

POSITIONS = ("GK", "DEF", "MID", "FWD")

# Every legal starting shape: one keeper, and outfield counts within FPL's bounds
# that sum to ten.
FORMATIONS: tuple[dict[str, int], ...] = tuple(
    {"GK": 1, "DEF": d, "MID": m, "FWD": f}
    for d, m, f in product(
        range(XI_MIN["DEF"], XI_MAX["DEF"] + 1),
        range(XI_MIN["MID"], XI_MAX["MID"] + 1),
        range(XI_MIN["FWD"], XI_MAX["FWD"] + 1),
    )
    if 1 + d + m + f == XI_SIZE
)

REQUIRED = ("code", "position", "team", "price", "expected_points")

Player = dict  # code, position, team, price, expected_points, availability?


@dataclass
class Eleven:
    points: float
    starters: list[int]
    captain: int
    formation: dict[str, int]
    bench: list[int] = field(default_factory=list)


def _to_players(frame: pd.DataFrame) -> list[Player]:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    keep = [*REQUIRED, *(c for c in ("availability", "web_name") if c in frame.columns)]
    return frame[keep].to_dict("records")


def _check_squad(squad: list[Player]) -> None:
    if len(squad) != SQUAD_SIZE:
        raise ValueError(f"a squad has {SQUAD_SIZE} players, got {len(squad)}")
    counts: dict[str, int] = {}
    clubs: dict[str, int] = {}
    codes = set()
    for p in squad:
        counts[p["position"]] = counts.get(p["position"], 0) + 1
        clubs[p["team"]] = clubs.get(p["team"], 0) + 1
        codes.add(p["code"])
    if counts != SQUAD_QUOTA:
        raise ValueError(f"squad must be {SQUAD_QUOTA}, got {counts}")
    if len(codes) != SQUAD_SIZE:
        raise ValueError("squad contains the same player twice")
    worst = max(clubs, key=clubs.get)
    if clubs[worst] > MAX_PER_CLUB:
        raise ValueError(f"more than {MAX_PER_CLUB} players from {worst}")


def _best_eleven(squad: list[Player]) -> Eleven:
    """Highest-projected legal eleven; the captain is the top projection in it."""
    by_position: dict[str, list[Player]] = {p: [] for p in POSITIONS}
    for player in squad:
        by_position[player["position"]].append(player)
    for players in by_position.values():
        players.sort(key=lambda p: -p["expected_points"])

    best: Eleven | None = None
    for formation in FORMATIONS:
        chosen = [p for pos, n in formation.items() for p in by_position[pos][:n]]
        captain = max(chosen, key=lambda p: p["expected_points"])
        points = sum(p["expected_points"] for p in chosen) + captain["expected_points"]
        if best is None or points > best.points:
            starters = [int(p["code"]) for p in chosen]
            best = Eleven(
                points=float(points),
                starters=starters,
                captain=int(captain["code"]),
                formation=dict(formation),
            )
    assert best is not None
    starting = set(best.starters)
    best.bench = [int(p["code"]) for p in squad if int(p["code"]) not in starting]
    return best


def best_eleven(squad: pd.DataFrame) -> Eleven:
    players = _to_players(squad)
    _check_squad(players)
    return _best_eleven(players)


def _rating(points: float, optimal_points: float | None) -> float | None:
    if not optimal_points:
        return None
    # Not capped at 100: the optimum is budget-constrained to £100m, and a squad whose
    # value has grown past that can honestly beat it.
    return round(100.0 * points / optimal_points, 1)


def rate_squad(squad: pd.DataFrame, optimal_points: float | None = None) -> dict:
    """Projected points for the squad's best eleven, and a rating against the best
    £100m squad this week (100 = matches it; above 100 is possible for richer squads)."""
    players = _to_players(squad)
    _check_squad(players)
    eleven = _best_eleven(players)
    return {
        "points": round(eleven.points, 2),
        "rating": _rating(eleven.points, optimal_points),
        "starters": eleven.starters,
        "captain": eleven.captain,
        "bench": eleven.bench,
        "formation": eleven.formation,
        "flagged": [int(p["code"]) for p in players if p.get("availability", 1.0) < 1],
    }


def suggest_transfers(
    squad: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    bank: float = 0.0,
    top_n: int = 5,
    optimal_points: float | None = None,
) -> list[dict]:
    """The ``top_n`` single transfers that most raise the squad's projected points.

    A candidate must play the same position as the player leaving, cost no more than
    his price plus the bank, and keep every club at or under the three-player limit.
    Each candidate is scored by re-picking the best eleven of the resulting fifteen.
    Suggestions are ranked by that gain and never repeat a player coming in.
    """
    players = _to_players(squad)
    _check_squad(players)
    base = _best_eleven(players).points
    owned = {p["code"] for p in players}
    clubs: dict[str, int] = {}
    for p in players:
        clubs[p["team"]] = clubs.get(p["team"], 0) + 1

    candidates_by_position: dict[str, list[Player]] = {p: [] for p in POSITIONS}
    for c in _to_players(pool):
        if c["code"] not in owned:
            candidates_by_position[c["position"]].append(c)

    results: list[dict] = []
    for leaving in players:
        rest = [p for p in players if p["code"] != leaving["code"]]
        budget = leaving["price"] + bank + 1e-9
        for arriving in candidates_by_position[leaving["position"]]:
            if arriving["price"] > budget:
                continue
            if (
                arriving["team"] != leaving["team"]
                and clubs.get(arriving["team"], 0) >= MAX_PER_CLUB
            ):
                continue
            new_points = _best_eleven([*rest, arriving]).points
            gain = new_points - base
            if gain <= 1e-9:
                continue
            results.append(
                {
                    "out": int(leaving["code"]),
                    "in": int(arriving["code"]),
                    "gain": round(gain, 2),
                    "points_after": round(new_points, 2),
                    "rating_after": _rating(new_points, optimal_points),
                    "cost_change": round(float(arriving["price"] - leaving["price"]), 1),
                }
            )

    results.sort(key=lambda r: -r["gain"])
    seen_in: set[int] = set()
    ranked: list[dict] = []
    for r in results:
        if r["in"] in seen_in:
            continue
        seen_in.add(r["in"])
        r["rank"] = len(ranked) + 1
        ranked.append(r)
        if len(ranked) == top_n:
            break
    return ranked


def rate_and_suggest(
    codes: list[int],
    projections: pd.DataFrame,
    *,
    bank: float = 0.0,
    top_n: int = 5,
    optimal_points: float | None = None,
) -> dict:
    """Convenience wrapper: from fifteen player codes to a rating and ranked moves."""
    squad = projections[projections["code"].isin(codes)].copy()
    unknown = set(codes) - set(squad["code"])
    if unknown:
        raise ValueError(f"unknown player codes: {sorted(unknown)}")
    rating = rate_squad(squad, optimal_points)
    rating["suggestions"] = suggest_transfers(
        squad, projections, bank=bank, top_n=top_n, optimal_points=optimal_points
    )
    return rating

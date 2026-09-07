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

from fpl.config import (
    MAX_PER_CLUB,
    SQUAD_QUOTA,
    SQUAD_SIZE,
    TRANSFER_HIT,
    XI_MAX,
    XI_MIN,
    XI_SIZE,
)

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

# Projection columns a manager can rate against: the next gameweek alone, or the
# deadline-weighted sum over the next three or five.
METRICS = ("expected_points", "ep1", "ep3", "ep5")

Player = dict  # code, position, team, price, expected_points, availability?


@dataclass
class Eleven:
    points: float
    starters: list[int]
    captain: int
    formation: dict[str, int]
    bench: list[int] = field(default_factory=list)


def _to_players(frame: pd.DataFrame, metric: str = "expected_points") -> list[Player]:
    """Plain rows with ``expected_points`` set to the chosen ``metric`` column."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if metric != "expected_points":
        if metric not in frame.columns:
            raise ValueError(f"projections have no {metric!r} column")
        frame = frame.drop(columns=["expected_points"], errors="ignore").rename(
            columns={metric: "expected_points"}
        )
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


def _best_eleven(squad: list[Player], formation: dict[str, int] | None = None) -> Eleven:
    """Highest-projected legal eleven (within ``formation`` if given); the captain is
    the top projection in it."""
    by_position: dict[str, list[Player]] = {p: [] for p in POSITIONS}
    for player in squad:
        by_position[player["position"]].append(player)
    for players in by_position.values():
        players.sort(key=lambda p: -p["expected_points"])

    best: Eleven | None = None
    for shape in FORMATIONS if formation is None else (formation,):
        chosen = [p for pos, n in shape.items() for p in by_position[pos][:n]]
        if len(chosen) != XI_SIZE:
            continue
        captain = max(chosen, key=lambda p: p["expected_points"])
        points = sum(p["expected_points"] for p in chosen) + captain["expected_points"]
        if best is None or points > best.points:
            starters = [int(p["code"]) for p in chosen]
            best = Eleven(
                points=float(points),
                starters=starters,
                captain=int(captain["code"]),
                formation=dict(shape),
            )
    assert best is not None
    starting = set(best.starters)
    best.bench = [int(p["code"]) for p in squad if int(p["code"]) not in starting]
    return best


def best_eleven(
    squad: pd.DataFrame, formation: dict[str, int] | None = None, metric: str = "expected_points"
) -> Eleven:
    players = _to_players(squad, metric)
    _check_squad(players)
    return _best_eleven(players, formation)


def _rating(points: float, optimal_points: float | None) -> float | None:
    if not optimal_points:
        return None
    # Not capped at 100: the optimum is budget-constrained to £100m, and a squad whose
    # value has grown past that can honestly beat it.
    return round(100.0 * points / optimal_points, 1)


def rate_squad(
    squad: pd.DataFrame, optimal_points: float | None = None, metric: str = "expected_points"
) -> dict:
    """Projected points for the squad's best eleven, and a rating against the best
    £100m squad this week (100 = matches it; above 100 is possible for richer squads)."""
    players = _to_players(squad, metric)
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


BUDGET_CAP = 100.0


def available_budget(
    squad: list[Player], bank: float, team_value: float | None = None, cap: float = BUDGET_CAP
) -> float:
    """Money a manager can spend in total.

    FPL values a squad at its selling prices, which the manager can read off their
    own team page but this model cannot see. So the manager's stated ``team_value``
    wins when given; otherwise the cap applies, or the squad's current value if that
    has grown past it. The bank is added either way.
    """
    if team_value:
        return float(team_value) + bank
    value = sum(p["price"] for p in squad)
    return max(cap, value) + bank


def _legal_after(squad: list[Player], outs: list[Player], ins: list[Player]) -> bool:
    clubs: dict[str, int] = {}
    for p in squad:
        if p not in outs:
            clubs[p["team"]] = clubs.get(p["team"], 0) + 1
    for p in ins:
        clubs[p["team"]] = clubs.get(p["team"], 0) + 1
        if clubs[p["team"]] > MAX_PER_CLUB:
            return False
    return True


def suggest_transfers(
    squad: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    bank: float = 0.0,
    top_n: int = 5,
    optimal_points: float | None = None,
    formation: dict[str, int] | None = None,
    team_value: float | None = None,
    cap: float = BUDGET_CAP,
    funding_candidates: int = 8,
    metric: str = "expected_points",
    free_transfers: int | None = None,
) -> list[dict]:
    """The ``top_n`` moves that most raise the squad's projected points under the cap.

    A move is one transfer, or two when one is not enough on its own: if the best
    replacement for a player costs more than the squad can afford, a second, funding
    transfer elsewhere in the squad is searched for so the pair fits under the cap
    and still raises the team. Every move is scored by re-solving the best eleven of
    the resulting fifteen (within ``formation`` if given); the gain is the change in
    the team's points, and no signing is repeated across the ranked list.

    ``metric`` picks which projection the team is rated on (next gameweek, or the
    weighted next three or five). With ``free_transfers`` given, each transfer beyond
    that number costs ``TRANSFER_HIT`` points, moves are ranked by the gain net of
    that hit, and each carries ``worth_it`` saying whether it still comes out ahead.
    """
    players = _to_players(squad, metric)
    _check_squad(players)
    base = _best_eleven(players, formation).points
    value = float(team_value) if team_value else sum(p["price"] for p in players)
    budget = available_budget(players, bank, team_value, cap)
    owned = {p["code"] for p in players}

    by_position: dict[str, list[Player]] = {p: [] for p in POSITIONS}
    for c in _to_players(pool, metric):
        if c["code"] not in owned:
            by_position[c["position"]].append(c)

    singles: list[dict] = []
    unfunded: list[tuple[Player, Player, float]] = []
    for leaving in players:
        rest = [p for p in players if p["code"] != leaving["code"]]
        for arriving in by_position[leaving["position"]]:
            if not _legal_after(players, [leaving], [arriving]):
                continue
            gain = _best_eleven([*rest, arriving], formation).points - base
            if gain <= 1e-9:
                continue
            new_value = value - leaving["price"] + arriving["price"]
            if new_value <= budget + 1e-9:
                singles.append(_move([leaving], [arriving], gain, base, optimal_points, value))
            else:
                unfunded.append((leaving, arriving, gain))

    # Two-transfer moves: fund the best over-budget upgrades with a downgrade elsewhere.
    doubles: list[dict] = []
    unfunded.sort(key=lambda t: -t[2])
    for leaving, arriving, _ in unfunded[:funding_candidates]:
        shortfall = value - leaving["price"] + arriving["price"] - budget
        after_first = [p for p in players if p["code"] != leaving["code"]] + [arriving]
        best: dict | None = None
        for second_out in after_first:
            if second_out["code"] == arriving["code"]:
                continue
            for second_in in by_position[second_out["position"]]:
                if second_in["code"] == arriving["code"]:
                    continue
                if second_out["price"] - second_in["price"] < shortfall - 1e-9:
                    continue
                if not _legal_after(after_first, [second_out], [second_in]):
                    continue
                trial = [p for p in after_first if p["code"] != second_out["code"]] + [second_in]
                gain = _best_eleven(trial, formation).points - base
                if gain <= 1e-9:
                    continue
                if best is None or gain > best["gain"]:
                    best = _move(
                        [leaving, second_out],
                        [arriving, second_in],
                        gain,
                        base,
                        optimal_points,
                        value,
                    )
        if best:
            doubles.append(best)

    for r in singles + doubles:
        _apply_hit(r, free_transfers)
    results = sorted(singles + doubles, key=lambda r: -r["net"])
    seen_in: set[int] = set()
    ranked: list[dict] = []
    for r in results:
        if any(i in seen_in for i in r["in"]):
            continue
        seen_in.update(r["in"])
        r["rank"] = len(ranked) + 1
        ranked.append(r)
        if len(ranked) == top_n:
            break
    return ranked


def _move(
    outs: list[Player], ins: list[Player], gain: float, base: float,
    optimal_points: float | None, value: float,
) -> dict:  # fmt: skip
    new_points = base + gain
    cost = sum(p["price"] for p in ins) - sum(p["price"] for p in outs)
    return {
        "out": [int(p["code"]) for p in outs],
        "in": [int(p["code"]) for p in ins],
        "transfers": len(outs),
        "gain": round(gain, 2),
        "points_after": round(new_points, 2),
        "rating_after": _rating(new_points, optimal_points),
        "cost_change": round(cost, 1),
        "value_after": round(value + cost, 1),
    }


def _apply_hit(move: dict, free_transfers: int | None) -> None:
    """Charge the points hit for transfers beyond the free ones and record the net."""
    if free_transfers is None:
        hit = 0
    else:
        hit = TRANSFER_HIT * max(0, move["transfers"] - int(free_transfers))
    move["hit"] = hit
    move["net"] = round(move["gain"] - hit, 2)
    move["worth_it"] = move["net"] > 1e-9


def rate_and_suggest(
    codes: list[int],
    projections: pd.DataFrame,
    *,
    bank: float = 0.0,
    team_value: float | None = None,
    top_n: int = 5,
    optimal_points: float | None = None,
    metric: str = "expected_points",
    free_transfers: int | None = None,
) -> dict:
    """Convenience wrapper: from fifteen player codes to a rating and ranked moves."""
    squad = projections[projections["code"].isin(codes)].copy()
    unknown = set(codes) - set(squad["code"])
    if unknown:
        raise ValueError(f"unknown player codes: {sorted(unknown)}")
    rating = rate_squad(squad, optimal_points, metric)
    rating["suggestions"] = suggest_transfers(
        squad, projections, bank=bank, top_n=top_n, optimal_points=optimal_points,
        team_value=team_value, metric=metric, free_transfers=free_transfers,
    )  # fmt: skip
    rating["budget"] = round(available_budget(_to_players(squad), bank, team_value), 1)
    rating["value"] = round(float(squad["price"].sum()), 1)
    return rating

"""Rate a manager's own squad and find the moves that lift it most.

Given the fifteen a manager actually owns, this answers two questions: how many
points is that squad projected to score, and which transfers raise that number the
most. The gain of a transfer is measured on the whole squad -- best eleven and
captain re-picked after the swap, for every gameweek in the horizon -- not on the two
players in isolation, because a new signing can change who starts and who wears the
armband, and a bench player is worth what he covers, not what he scores on the bench.

Scoring a squad
---------------
For each gameweek ``k`` in the horizon the best legal eleven is picked on that
gameweek's projections (the manager's chosen formation is honoured for the next
gameweek only; later weeks are free, since the lineup can be changed), the top
projection in it is doubled as captain, and expected bench cover is added: the
chance that at least ``j`` starters miss out, times what the ``j``-th bench player
would bring on. The gameweeks are then summed with ``HORIZON_WEIGHTS``.

The core works on plain dicts rather than DataFrames: a squad is fifteen rows, a
transfer search evaluates a few thousand candidate squads, and the same logic is
mirrored in the browser where there is no pandas at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import pandas as pd

from fpl.config import (
    HORIZON_WEIGHTS,
    MAX_PER_CLUB,
    METRIC_HORIZON,
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
METRICS = tuple(METRIC_HORIZON)
BUDGET_CAP = 100.0

Player = dict  # code, position, team, price, expected_points, eps, plays, availability?


@dataclass
class Eleven:
    points: float
    starters: list[int]
    captain: int
    formation: dict[str, int]
    bench: list[int] = field(default_factory=list)
    vice: int | None = None
    cover: float = 0.0


@dataclass
class Score:
    points: float  # weighted over the horizon, bench cover included
    lineups: list[Eleven]  # one per gameweek, index 0 = next gameweek


def play_probability(availability: float, p_60: float | None) -> float:
    """Chance a player takes the pitch at all, for bench-cover arithmetic.

    ``p_60`` is the model's chance of a 60-minute appearance after availability;
    the mapping stretches it so a rotation risk still usually plays some minutes.
    """
    avail = max(0.0, min(1.0, float(availability)))
    if p_60 is None or pd.isna(p_60) or avail <= 0:
        return avail
    raw = min(1.0, float(p_60) / avail)
    return avail * min(1.0, 0.35 + 0.65 * raw)


def _to_players(
    frame: pd.DataFrame, metric: str = "expected_points", fixtures: pd.DataFrame | None = None
) -> list[Player]:
    """Plain rows with ``expected_points`` set to the chosen ``metric`` column, plus
    per-gameweek ``eps`` and play probabilities when ``fixtures`` is given."""
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
    extras = [c for c in ("availability", "web_name", "p_60") if c in frame.columns]
    players = frame[[*REQUIRED, *extras]].to_dict("records")

    horizon = METRIC_HORIZON[metric]
    runs: dict[int, dict[int, tuple[float, float]]] = {}
    if fixtures is not None and horizon > 1:
        cols = ["code", "offset", "expected_points", "availability", "p_60"]
        for code, offset, ep, avail, p60 in fixtures[cols].itertuples(index=False):
            if 0 <= int(offset) < horizon:
                runs.setdefault(int(code), {})[int(offset)] = (
                    float(ep),
                    play_probability(avail, p60),
                )
    for p in players:
        p["code"] = int(p["code"])
        avail = float(p.get("availability", 1.0))
        if pd.isna(avail):
            avail = 1.0
        p["availability"] = avail
        run = runs.get(p["code"])
        if run:
            p["eps"] = [run.get(k, (0.0, 0.0))[0] for k in range(horizon)]
            p["plays"] = [run.get(k, (0.0, 0.0))[1] for k in range(horizon)]
        else:
            p["eps"] = [float(p["expected_points"])]
            p["plays"] = [play_probability(avail, p.get("p_60"))]
    return players


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


PRUNE_MARGIN = 0.5  # weighted points; bench cover can add at most about this much


def _total(p: Player) -> float:
    eps = p.get("eps") or [float(p["expected_points"])]
    return sum(w * e for w, e in zip(HORIZON_WEIGHTS, eps, strict=False))


def _ep(p: Player, k: int) -> float:
    eps = p.get("eps")
    if eps is None:
        return float(p["expected_points"]) if k == 0 else 0.0
    return eps[k] if k < len(eps) else 0.0


def _play(p: Player, k: int) -> float:
    plays = p.get("plays")
    if plays is None:
        return float(p.get("availability", 1.0))
    return plays[k] if k < len(plays) else 0.0


def _at_least(probabilities: list[float]) -> list[float]:
    """``out[j]`` = chance that at least ``j`` of the events happen (independent)."""
    dist = [1.0]
    for q in probabilities:
        nxt = [0.0] * (len(dist) + 1)
        for i, v in enumerate(dist):
            nxt[i] += v * (1 - q)
            nxt[i + 1] += v * q
        dist = nxt
    out = []
    tail = 1.0
    for j in range(len(dist)):
        out.append(tail)
        tail -= dist[j]
    return out


def _bench_cover(squad: list[Player], eleven: Eleven, k: int) -> float:
    """Expected points from automatic substitutions in gameweek ``k``."""
    by_code = {p["code"]: p for p in squad}
    starters = [by_code[c] for c in eleven.starters]
    bench = [by_code[c] for c in eleven.bench]
    gk_out = [p for p in starters if p["position"] == "GK"]
    outfield = [p for p in starters if p["position"] != "GK"]
    bench_gk = [p for p in bench if p["position"] == "GK"]
    bench_out = sorted((p for p in bench if p["position"] != "GK"), key=lambda p: -_ep(p, k))

    cover = 0.0
    if gk_out and bench_gk:
        cover += (1 - _play(gk_out[0], k)) * _ep(bench_gk[0], k)
    misses = _at_least([1 - _play(p, k) for p in outfield])
    for j, sub in enumerate(bench_out, start=1):
        if j < len(misses):
            cover += misses[j] * _ep(sub, k)
    return cover


def _best_eleven(
    squad: list[Player], formation: dict[str, int] | None = None, k: int = 0
) -> Eleven:
    """Highest-projected legal eleven for gameweek ``k`` (within ``formation`` if
    given); the captain is the top projection in it, the vice the next."""
    by_position: dict[str, list[Player]] = {p: [] for p in POSITIONS}
    for player in squad:
        by_position[player["position"]].append(player)
    for players in by_position.values():
        players.sort(key=lambda p: -_ep(p, k))

    best: Eleven | None = None
    for shape in FORMATIONS if formation is None else (formation,):
        chosen = [p for pos, n in shape.items() for p in by_position[pos][:n]]
        if len(chosen) != XI_SIZE:
            continue
        ranked = sorted(chosen, key=lambda p: -_ep(p, k))
        captain, vice = ranked[0], ranked[1]
        points = sum(_ep(p, k) for p in chosen) + _ep(captain, k)
        if best is None or points > best.points:
            best = Eleven(
                points=float(points),
                starters=[int(p["code"]) for p in chosen],
                captain=int(captain["code"]),
                vice=int(vice["code"]),
                formation=dict(shape),
            )
    assert best is not None
    starting = set(best.starters)
    rest = [p for p in squad if int(p["code"]) not in starting]
    # Bench order as FPL wants it: keeper first, then outfield by projection.
    rest.sort(key=lambda p: (p["position"] != "GK", -_ep(p, k)))
    best.bench = [int(p["code"]) for p in rest]
    best.cover = _bench_cover(squad, best, k)
    return best


def _horizon(squad: list[Player]) -> int:
    return max(len(p.get("eps") or [1]) for p in squad)


def _score(squad: list[Player], formation: dict[str, int] | None = None) -> Score:
    """Weighted points over the horizon, re-picking the eleven each gameweek."""
    lineups = []
    total = 0.0
    for k in range(_horizon(squad)):
        eleven = _best_eleven(squad, formation if k == 0 else None, k)
        lineups.append(eleven)
        total += HORIZON_WEIGHTS[k] * (eleven.points + eleven.cover)
    return Score(points=total, lineups=lineups)


def _points(squad: list[Player], formation: dict[str, int] | None = None) -> float:
    return _score(squad, formation).points


def best_eleven(
    squad: pd.DataFrame,
    formation: dict[str, int] | None = None,
    metric: str = "expected_points",
    fixtures: pd.DataFrame | None = None,
) -> Eleven:
    """The next gameweek's best eleven (captain and bench order included)."""
    players = _to_players(squad, metric, fixtures)
    _check_squad(players)
    return _best_eleven(players, formation, 0)


def _rating(points: float, optimal_points: float | None) -> float | None:
    if not optimal_points:
        return None
    # Not capped at 100: the optimum is budget-constrained to £100m, and a squad whose
    # value has grown past that can honestly beat it.
    return round(100.0 * points / optimal_points, 1)


def _lineup_record(k: int, e: Eleven) -> dict:
    return {
        "offset": k,
        "points": round(e.points, 2),
        "cover": round(e.cover, 2),
        "starters": e.starters,
        "captain": e.captain,
        "vice": e.vice,
        "bench": e.bench,
        "formation": e.formation,
    }


def rate_squad(
    squad: pd.DataFrame,
    optimal_points: float | None = None,
    metric: str = "expected_points",
    fixtures: pd.DataFrame | None = None,
    formation: dict[str, int] | None = None,
) -> dict:
    """Projected points for the squad over the metric's horizon, and a rating against
    the best £100m squad (100 = matches it; above 100 is possible for richer squads).

    The lineup fields describe the next gameweek; ``lineups`` carries every
    gameweek's recommended eleven so the page can show how the bench gets used."""
    players = _to_players(squad, metric, fixtures)
    _check_squad(players)
    score = _score(players, formation)
    nxt = score.lineups[0]
    return {
        "points": round(score.points, 2),
        "next_points": round(nxt.points, 2),
        "bench_cover": round(nxt.cover, 2),
        "rating": _rating(score.points, optimal_points),
        "starters": nxt.starters,
        "captain": nxt.captain,
        "vice": nxt.vice,
        "bench": nxt.bench,
        "formation": nxt.formation,
        "lineups": [_lineup_record(k, e) for k, e in enumerate(score.lineups)],
        "flagged": [int(p["code"]) for p in players if p.get("availability", 1.0) < 1],
    }


def available_budget(
    squad: list[Player], bank: float, team_value: float | None = None, cap: float = BUDGET_CAP
) -> float:
    """Money a manager can spend in total.

    FPL values a squad at its selling prices, which the manager can read off their
    own team page but this model cannot see. So the manager's stated ``team_value``
    wins when given -- a value above what the fifteen on the pitch cost is genuine
    headroom, one below it means the squad is over budget and must be trimmed.
    Without it the cap applies, or the squad's current value if that has grown past
    the cap. The bank is added either way.
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
    fixtures: pd.DataFrame | None = None,
) -> list[dict]:
    """The ``top_n`` moves that most raise the squad's projected points under the cap.

    A move is one transfer, or two when one is not enough on its own: if the best
    replacement for a player costs more than the squad can afford, a second, funding
    transfer elsewhere in the squad is searched for so the pair fits under the cap
    and still raises the team. Every move is scored by re-solving the squad over the
    horizon (within ``formation`` for the next gameweek); the gain is the change in
    the team's points, and no signing is repeated across the ranked list.

    ``metric`` picks the horizon (next gameweek, or the weighted next three or
    five; ``fixtures`` supplies the per-gameweek projections that make the
    multi-week case honest). With ``free_transfers`` given, each transfer beyond
    that number costs ``TRANSFER_HIT`` points, moves are ranked by the gain net of
    that hit, and each carries ``worth_it``.

    If the squad is already over budget, only moves that bring it back under are
    offered -- even ones that cost points -- because the team is not legal as it
    stands; those carry ``fixes_budget``.
    """
    players = _to_players(squad, metric, fixtures)
    _check_squad(players)
    base = _points(players, formation)
    value = sum(p["price"] for p in players)
    budget = available_budget(players, bank, team_value, cap)
    deficit = value > budget + 1e-9
    owned = {p["code"] for p in players}

    by_position: dict[str, list[Player]] = {p: [] for p in POSITIONS}
    for c in _to_players(pool, metric, fixtures):
        if c["code"] not in owned:
            by_position[c["position"]].append(c)

    singles: list[dict] = []
    unfunded: list[tuple[Player, Player, float]] = []
    for leaving in players:
        rest = [p for p in players if p["code"] != leaving["code"]]
        for arriving in by_position[leaving["position"]]:
            if not _legal_after(players, [leaving], [arriving]):
                continue
            new_value = value - leaving["price"] + arriving["price"]
            fits = new_value <= budget + 1e-9
            if deficit:
                if arriving["price"] >= leaving["price"]:
                    continue  # over budget: only downgrades are on the table
            elif _total(arriving) < _total(leaving) - PRUNE_MARGIN:
                continue  # cannot lift the team: he projects less in every sense
            gain = _points([*rest, arriving], formation) - base
            if fits and (gain > 1e-9 or deficit):
                singles.append(_move([leaving], [arriving], gain, base, optimal_points, value))
            elif not fits and (gain > 1e-9 or deficit):
                unfunded.append((leaving, arriving, gain))

    # Two-transfer moves: fund the best over-budget upgrades with a downgrade elsewhere.
    doubles: list[dict] = []
    if deficit:  # the first transfer must do most of the saving; rank by price cut
        unfunded.sort(key=lambda t: (t[1]["price"] - t[0]["price"], -t[2]))
    else:
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
                gain = _points(trial, formation) - base
                if gain <= 1e-9 and not deficit:
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
        r["fixes_budget"] = deficit
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
    fixtures: pd.DataFrame | None = None,
    formation: dict[str, int] | None = None,
) -> dict:
    """Convenience wrapper: from fifteen player codes to a rating and ranked moves."""
    squad = projections[projections["code"].isin(codes)].copy()
    unknown = set(codes) - set(squad["code"])
    if unknown:
        raise ValueError(f"unknown player codes: {sorted(unknown)}")
    rating = rate_squad(squad, optimal_points, metric, fixtures, formation)
    rating["suggestions"] = suggest_transfers(
        squad, projections, bank=bank, top_n=top_n, optimal_points=optimal_points,
        team_value=team_value, metric=metric, free_transfers=free_transfers,
        fixtures=fixtures, formation=formation,
    )  # fmt: skip
    rating["budget"] = round(available_budget(_to_players(squad), bank, team_value), 1)
    rating["value"] = round(float(squad["price"].sum()), 1)
    return rating

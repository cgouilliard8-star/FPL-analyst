"""Multi-gameweek transfer planning as one mixed-integer program.

The single-week suggester asks "which swap lifts the squad most right now?"; the
plan asks "which sequence of swaps, hits and holds gives the most points over the
run?". Banking a free transfer this week to make a double next week, taking a hit
now for a player with three soft fixtures, selling a man the week *before* his run
turns -- none of that is visible one week at a time. This is the formulation the
open-source FPL solvers (sertalpbilal/FPL-Optimization-Tools and its descendants)
made standard, written against this project's projections and rules:

* a squad, a lineup and a captain for every gameweek of the horizon;
* transfers link one week's squad to the next; free transfers bank up to five and
  every transfer beyond them costs ``TRANSFER_HIT`` points;
* the budget is the squad's value plus the bank (prices are held fixed);
* the objective is the same team-points measure the rating uses -- the eleven, the
  captain doubled, the bench at ``BENCH_WEIGHT`` -- weighted by ``HORIZON_WEIGHTS``,
  less the hits, less a hair per transfer so the solver does not churn for nothing.

Solved with CBC through pulp on a pruned pool: the current fifteen plus the best
few by weighted projection in each position and the cheapest playing options, which
is where every plan worth making lives. Well under a minute on a laptop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pulp

from fpl.config import (
    BENCH_WEIGHT,
    HORIZON_WEIGHTS,
    MAX_PER_CLUB,
    SQUAD_QUOTA,
    SQUAD_SIZE,
    TRANSFER_HIT,
    XI_MAX,
    XI_MIN,
    XI_SIZE,
)

log = logging.getLogger(__name__)

MAX_FREE = 5
CANDIDATES_PER_POSITION = 8
ENABLERS_PER_POSITION = 2  # cheapest players who actually play: the money-makers
CHURN_PENALTY = 0.05  # weighted points a transfer must be worth to be made at all


@dataclass
class Week:
    gameweek: int
    transfers_in: list[dict] = field(default_factory=list)
    transfers_out: list[dict] = field(default_factory=list)
    hit: int = 0
    free_after: int = 0
    starters: list[int] = field(default_factory=list)
    bench: list[int] = field(default_factory=list)
    captain: int | None = None
    points: float = 0.0  # eleven + captain + weighted bench, this week, unweighted


@dataclass
class Plan:
    weeks: list[Week]
    objective: float  # weighted team points over the run, hits and churn deducted
    hold: float  # the same measure for standing still
    status: str


def _ep(p: dict, k: int) -> float:
    eps = p.get("eps") or []
    return float(eps[k]) if k < len(eps) else 0.0


def _pool(squad: list[dict], pool: list[dict], horizon: int, weights) -> list[dict]:
    owned = {p["code"] for p in squad}
    worth = lambda p: sum(weights[k] * _ep(p, k) for k in range(horizon))  # noqa: E731
    chosen = list(squad)
    for position in SQUAD_QUOTA:
        others = [p for p in pool if p["position"] == position and p["code"] not in owned]
        others.sort(key=worth, reverse=True)
        chosen.extend(others[:CANDIDATES_PER_POSITION])
        playing = [p for p in others[CANDIDATES_PER_POSITION:] if _ep(p, 0) > 1.0]
        playing.sort(key=lambda p: p["price"])
        chosen.extend(playing[:ENABLERS_PER_POSITION])
    seen, out = set(), []
    for p in chosen:
        if p["code"] not in seen:
            seen.add(p["code"])
            out.append(p)
    return out


def plan_transfers(
    squad: list[dict],
    pool: list[dict],
    *,
    bank: float,
    free_transfers: int,
    first_gameweek: int,
    horizon: int | None = None,
    weights: tuple[float, ...] = HORIZON_WEIGHTS,
    hit: int = TRANSFER_HIT,
    bench_weight: float = BENCH_WEIGHT,
    time_limit: int = 30,
) -> Plan:
    """The best sequence of transfers over ``horizon`` gameweeks for this squad.

    Players are dicts with ``code, position, team, price, eps`` (per-gameweek
    projections from ``first_gameweek``). ``squad`` is the current fifteen.
    """
    horizon = horizon or len(weights)
    weeks = range(horizon)
    players = _pool(squad, pool, horizon, weights)
    by_code = {p["code"]: p for p in players}
    owned = {p["code"] for p in squad}
    budget = sum(p["price"] for p in squad) + bank

    prob = pulp.LpProblem("fpl_plan", pulp.LpMaximize)
    codes = [p["code"] for p in players]
    sq = pulp.LpVariable.dicts("sq", (codes, weeks), cat="Binary")
    xi = pulp.LpVariable.dicts("xi", (codes, weeks), cat="Binary")
    cap = pulp.LpVariable.dicts("cap", (codes, weeks), cat="Binary")
    tin = pulp.LpVariable.dicts("tin", (codes, weeks), cat="Binary")
    tout = pulp.LpVariable.dicts("tout", (codes, weeks), cat="Binary")
    ft = pulp.LpVariable.dicts("ft", weeks, lowBound=0, upBound=MAX_FREE, cat="Integer")
    paid = pulp.LpVariable.dicts("paid", weeks, lowBound=0, cat="Integer")

    points = {}
    for w in weeks:
        points[w] = pulp.lpSum(
            _ep(by_code[c], w) * (xi[c][w] + cap[c][w] + bench_weight * (sq[c][w] - xi[c][w]))
            for c in codes
        )
    prob += pulp.lpSum(
        weights[w] * points[w]
        - hit * paid[w]
        - CHURN_PENALTY * pulp.lpSum(tin[c][w] for c in codes)
        for w in weeks
    )

    for w in weeks:
        prob += pulp.lpSum(sq[c][w] for c in codes) == SQUAD_SIZE
        prob += pulp.lpSum(xi[c][w] for c in codes) == XI_SIZE
        prob += pulp.lpSum(cap[c][w] for c in codes) == 1
        prob += pulp.lpSum(by_code[c]["price"] * sq[c][w] for c in codes) <= budget + 1e-6
        for c in codes:
            prob += xi[c][w] <= sq[c][w]
            prob += cap[c][w] <= xi[c][w]
            if by_code[c]["position"] not in ("MID", "FWD"):
                prob += cap[c][w] == 0  # the armband is a bet on a ceiling
            previous = (1 if c in owned else 0) if w == 0 else sq[c][w - 1]
            prob += sq[c][w] - previous == tin[c][w] - tout[c][w]
            prob += tin[c][w] + tout[c][w] <= 1
        for position, quota in SQUAD_QUOTA.items():
            members = [c for c in codes if by_code[c]["position"] == position]
            prob += pulp.lpSum(sq[c][w] for c in members) == quota
            prob += pulp.lpSum(xi[c][w] for c in members) >= XI_MIN[position]
            prob += pulp.lpSum(xi[c][w] for c in members) <= XI_MAX[position]
        for club in {p["team"] for p in players}:
            members = [c for c in codes if by_code[c]["team"] == club]
            prob += pulp.lpSum(sq[c][w] for c in members) <= MAX_PER_CLUB
        # free transfers: what is left after this week's moves, plus one, capped at five
        n_transfers = pulp.lpSum(tin[c][w] for c in codes)
        prob += paid[w] >= n_transfers - ft[w]
        if w == 0:
            prob += ft[0] == min(MAX_FREE, max(0, free_transfers))
        if w + 1 in weeks:
            prob += ft[w + 1] <= ft[w] - n_transfers + paid[w] + 1
            prob += ft[w + 1] <= MAX_FREE

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=0.005))
    status_name = pulp.LpStatus[status]
    if status_name not in ("Optimal", "Not Solved"):
        raise RuntimeError(f"transfer plan failed: {status_name}")

    weeks_out: list[Week] = []
    for w in weeks:
        chosen = [c for c in codes if sq[c][w].value() > 0.5]
        starters = [c for c in chosen if xi[c][w].value() > 0.5]
        captain = next((c for c in chosen if cap[c][w].value() > 0.5), None)
        bench = sorted([c for c in chosen if c not in starters], key=lambda c: -_ep(by_code[c], w))
        week = Week(
            gameweek=first_gameweek + w,
            transfers_in=[by_code[c] for c in codes if tin[c][w].value() > 0.5],
            transfers_out=[by_code[c] for c in codes if tout[c][w].value() > 0.5],
            hit=int(round(paid[w].value() or 0)) * hit,
            free_after=int(round(ft[w + 1].value())) if w + 1 in weeks else 0,
            starters=starters,
            bench=bench,
            captain=captain,
            points=float(pulp.value(points[w])),
        )
        weeks_out.append(week)
    objective = float(pulp.value(prob.objective))
    hold = _hold_value(squad, horizon, weights, bench_weight)
    return Plan(weeks=weeks_out, objective=objective, hold=hold, status=status_name)


def _hold_value(squad: list[dict], horizon: int, weights, bench_weight: float) -> float:
    """The same objective for making no transfers: best eleven and captain each week."""
    from fpl.optimise.rate import _best_eleven

    total = 0.0
    for k in range(horizon):
        eleven = _best_eleven(squad, None, k)
        starters = set(eleven.starters)
        pts = sum(_ep(p, k) for p in squad if p["code"] in starters)
        pts += _ep(next(p for p in squad if p["code"] == eleven.captain), k)
        pts += bench_weight * sum(_ep(p, k) for p in squad if p["code"] not in starters)
        total += weights[k] * pts
    return total

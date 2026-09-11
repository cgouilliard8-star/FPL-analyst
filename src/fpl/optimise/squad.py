"""Squad selection as a mixed-integer program.

Projections are a ranked table; a squad is an answer. The difference is the
constraints: £100m, two keepers, five defenders, five midfielders, three forwards,
at most three players from any one club, and a legal eleven inside that fifteen.

Greedily taking the highest projections violates all of them at once, which is why
this is solved rather than sorted.
"""

from __future__ import annotations

import logging

import pandas as pd
import pulp

from fpl.config import (
    BENCH_WEIGHT,
    BUDGET_TENTHS,
    MAX_PER_CLUB,
    SQUAD_QUOTA,
    SQUAD_SIZE,
    XI_MAX,
    XI_MIN,
    XI_SIZE,
)

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("code", "position", "team", "price_tenths", "projected")


def _validate(players: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(players.columns)
    if missing:
        raise ValueError(f"player pool is missing {sorted(missing)}")
    if players["code"].duplicated().any():
        raise ValueError("player pool contains duplicate codes")


def pick_squad(
    players: pd.DataFrame,
    *,
    budget_tenths: int = BUDGET_TENTHS,
    max_per_club: int = MAX_PER_CLUB,
    solver: pulp.LpSolver | None = None,
) -> pd.DataFrame:
    """Choose the fifteen that maximise projected points inside every FPL rule.

    The objective is the starting eleven plus a captain, with the bench counted at
    ``BENCH_WEIGHT`` -- about the chance a substitute comes on. Counting the bench in
    full would buy an expensive bench that never plays; counting it at nothing left
    money unspent and four £4.5m players who could never cover an absence. The
    captain must be a midfielder or forward: his points are a bet on a ceiling.
    """
    _validate(players)
    pool = players.reset_index(drop=True)

    problem = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", pool.index, cat="Binary")
    starter = pulp.LpVariable.dicts("start", pool.index, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", pool.index, cat="Binary")

    # A captain scores double, so his projection is counted twice; the bench counts
    # for the chance it is needed.
    problem += pulp.lpSum(
        pool.loc[i, "projected"]
        * (starter[i] + captain[i] + BENCH_WEIGHT * (squad[i] - starter[i]))
        for i in pool.index
    )
    for i in pool.index[~pool["position"].isin(["MID", "FWD"])]:
        problem += captain[i] == 0

    problem += pulp.lpSum(squad.values()) == SQUAD_SIZE
    problem += pulp.lpSum(starter.values()) == XI_SIZE
    problem += pulp.lpSum(captain.values()) == 1
    problem += (
        pulp.lpSum(pool.loc[i, "price_tenths"] * squad[i] for i in pool.index) <= budget_tenths
    )

    for i in pool.index:
        problem += starter[i] <= squad[i]  # cannot start a player you do not own
        problem += captain[i] <= starter[i]  # the captain must be in the eleven

    for position, quota in SQUAD_QUOTA.items():
        members = pool.index[pool["position"] == position]
        problem += pulp.lpSum(squad[i] for i in members) == quota
        problem += pulp.lpSum(starter[i] for i in members) >= XI_MIN[position]
        problem += pulp.lpSum(starter[i] for i in members) <= XI_MAX[position]

    for club in pool["team"].unique():
        members = pool.index[pool["team"] == club]
        problem += pulp.lpSum(squad[i] for i in members) <= max_per_club

    status = problem.solve(solver or pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"squad optimisation failed: {pulp.LpStatus[status]}")

    chosen = pool.loc[[i for i in pool.index if squad[i].value() > 0.5]].copy()
    chosen["is_starter"] = [starter[i].value() > 0.5 for i in chosen.index]
    chosen["is_captain"] = [captain[i].value() > 0.5 for i in chosen.index]
    return chosen.sort_values(
        ["is_starter", "is_captain", "projected"], ascending=[False, False, False]
    ).reset_index(drop=True)


def squad_points(squad: pd.DataFrame, actual_column: str = "actual") -> float:
    """Points the chosen eleven actually scored, with the captain doubled."""
    starters = squad[squad["is_starter"]]
    total = starters[actual_column].sum()
    total += squad.loc[squad["is_captain"], actual_column].sum()  # captain counts twice
    return float(total)


def build_pool(predictions: pd.DataFrame, features: pd.DataFrame, gameweek: int) -> pd.DataFrame:
    """Join projections to the price and club needed to make them selectable."""
    context = features.loc[
        features["GW"] == gameweek, ["code", "team", "value", "position"]
    ].drop_duplicates("code")

    pool = (
        predictions[predictions["gameweek"] == gameweek]
        .merge(context, on="code", how="inner", suffixes=("", "_ctx"))
        .rename(columns={"predicted": "projected", "value": "price_tenths"})
    )
    return pool[["code", "position", "team", "price_tenths", "projected", "actual"]]


def backtest(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    budget_tenths: int = BUDGET_TENTHS,
) -> pd.DataFrame:
    """Pick a fresh optimal squad each gameweek and record what it scored.

    Rebuilding the squad every week isolates the question this project is about --
    how good are the projections -- from transfer planning, which is a separate
    optimisation and is out of scope for v1.
    """
    rows = []
    for gameweek in sorted(predictions["gameweek"].unique()):
        pool = build_pool(predictions, features, gameweek)
        if len(pool) < SQUAD_SIZE:
            log.warning("GW%s: only %d selectable players, skipping", gameweek, len(pool))
            continue
        squad = pick_squad(pool, budget_tenths=budget_tenths)
        rows.append(
            {
                "gameweek": gameweek,
                "points": squad_points(squad),
                "projected": float(
                    squad.loc[squad["is_starter"], "projected"].sum()
                    + squad.loc[squad["is_captain"], "projected"].sum()
                ),
                "captain": int(squad.loc[squad["is_captain"], "code"].iloc[0]),
                "spend": int(squad["price_tenths"].sum()),
            }
        )
    return pd.DataFrame(rows)

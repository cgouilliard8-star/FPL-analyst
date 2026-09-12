"""A whole season as a manager would play it, on projections made before each deadline.

This is the laboratory for the question that matters -- *how many points does the
model make over a season?* -- rather than how well it ranks players. It runs on the
cached walk-forward projections (``fpl backtest``, with the horizon rows the harness
writes alongside), so a full season of decisions replays in a minute or two and
every strategy idea can be measured before it is believed:

* the squad is built at the first gameweek and then managed with FPL's rules --
  one free transfer a week banking to five, a four-point hit beyond, the budget
  frozen at squad value plus bank;
* transfers come from the multi-week solver (``optimise.plan``) or the one-week
  suggester;
* the chips are played by explicit rules: Wildcard when the rebuilt squad beats the
  current one by enough over the run (or the window is closing), Free Hit when the
  best one-week squad beats ours by enough in a week no other week in the horizon
  beats, Bench Boost when the bench is worth enough in such a week, Triple Captain
  likewise for the captain;
* the lineup, captain and bench order are the model's; FPL's automatic
  substitutions apply; hits and chips score as they do in the game.

Two managers can play: **model** (projections) and **crowd** (ownership at each
deadline: the template team, the closest stand-in for the average manager the
archive allows). Real injury flags are not archived, so neither manager sees them;
the projections' own start probabilities are the only guide, as for a manager who
never read the news.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fpl.config import BENCH_WEIGHT, HORIZON_WEIGHTS, MAX_PER_CLUB, PROJECT_ROOT, TRANSFER_HIT
from fpl.data.core_insights import AVERAGES_PATH as AVERAGES
from fpl.evaluate.season_sim import MAX_FREE_TRANSFERS, MIN_GAIN, _play
from fpl.features.build import load_features
from fpl.optimise.plan import plan_transfers
from fpl.optimise.rate import _best_eleven, _ep, suggest_transfers
from fpl.optimise.squad import pick_squad

log = logging.getLogger(__name__)

PREDICTIONS = PROJECT_ROOT / "data" / "gold" / "predictions"


@dataclass(frozen=True)
class Strategy:
    """Everything a manager decides beyond the projections themselves."""

    planner: str = "milp"  # or "greedy"
    hits: bool = True
    hit_cost: float = 8.0  # what the planner believes a hit costs: twice the real 4,
    # because the gain it sees is the largest of many noisy estimates
    chips: bool = True
    wc_min_gain: float = 16.0  # weighted run points a wildcard must add
    wc_late_gain: float = 4.0  # ...or, in the last weeks of its window, this much
    wc_settle: int = 3  # no wildcard this soon after the squad was built
    fh_min_gain: float = 20.0  # points the one-week squad must add, this week
    bb_min: float = 14.0  # bench points this week
    tc_min: float = 10.0  # captain's projection this week
    bench_weight: float = BENCH_WEIGHT
    weights: tuple[float, ...] = HORIZON_WEIGHTS
    min_gain: float = MIN_GAIN  # greedy planner only


BASELINE = Strategy()
NO_CHIPS = Strategy(chips=False)
GREEDY = Strategy(planner="greedy", hits=False, chips=False)


@dataclass
class Week:
    gameweek: int
    points: int
    transfers: list[str] = field(default_factory=list)
    captain: str = "?"
    chip: str | None = None
    hit: int = 0
    bank: float = 0.0
    projected: float = 0.0


def chip_windows(season: str) -> list[tuple[str, int, int]]:
    """Which chips exist and when: (chip, first gameweek, last gameweek).

    Until 2024-25 there was one of each plus a second wildcard from GW20; from
    2025-26 every chip comes twice, one set for each half of the season.
    """
    start = int(season[:4])
    if start >= 2025:
        halves = [(2, 19), (20, 38)]
        return [
            (chip, a, b) for a, b in halves for chip in ("wildcard", "freehit", "bboost", "3xc")
        ]
    return [
        ("wildcard", 2, 19),
        ("wildcard", 20, 38),
        ("freehit", 2, 38),
        ("bboost", 2, 38),
        ("3xc", 2, 38),
    ]


# ---------------------------------------------------------------- projections
def load_season(season: str, model: str = "component") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The per-gameweek pool with ``eps``/``plays``/``haul`` per player, and actuals."""
    preds = pd.read_parquet(PREDICTIONS / f"{model}_{season}.parquet")
    hpath = PREDICTIONS / f"{model}_{season}_horizon.parquet"
    ahead = pd.read_parquet(hpath) if hpath.exists() else None
    feats = load_features()
    feats = feats[feats["season"] == season][
        ["code", "GW", "full_name", "position", "team", "price", "ownership", "minutes",
         "total_points", "fixtures_this_gw"]
    ].rename(columns={"full_name": "web_name"})  # fmt: skip
    feats = feats[feats["price"] > 0].drop_duplicates(["code", "GW"])
    actual = feats.groupby(["code", "GW"])[["total_points", "minutes"]].sum()
    return _assemble(preds, ahead, feats), actual


def _assemble(preds: pd.DataFrame, ahead: pd.DataFrame | None, feats: pd.DataFrame) -> pd.DataFrame:
    horizon = len(HORIZON_WEIGHTS)
    now = preds.rename(columns={"gameweek": "GW"})
    for col in ("p_play", "p_haul"):  # older prediction files carry points only
        if col not in now.columns:
            now[col] = float("nan")
    now = now[["code", "GW", "predicted", "p_play", "p_haul"]].assign(k=0)
    parts = [now]
    if ahead is not None:
        parts.append(
            ahead.rename(columns={"gameweek": "GW"})[["code", "GW", "k", "predicted", "p_play"]]
        )
    long = pd.concat(parts, ignore_index=True)
    long = long[long["k"] < horizon]
    long = long.drop_duplicates(["code", "GW", "k"])
    wide = long.pivot(  # noqa: PD010 - pivot_table drops an all-NaN column
        index=["code", "GW"], columns="k", values=["predicted", "p_play"]
    )
    empty = pd.Series(float("nan"), index=wide.index)
    eps = [wide["predicted"].get(k, empty).fillna(0.0) for k in range(horizon)]
    plays = [wide["p_play"].get(k, empty).fillna(0.0) for k in range(horizon)]
    table = pd.DataFrame(index=wide.index)
    table["eps"] = list(zip(*[e.to_numpy() for e in eps], strict=True))
    table["plays"] = list(zip(*[p.to_numpy() for p in plays], strict=True))
    haul = now.set_index(["code", "GW"])["p_haul"]
    table["haul"] = haul.reindex(table.index).fillna(0.0).to_numpy()
    table = table.reset_index()
    pool = feats.merge(table, on=["code", "GW"], how="left")
    pool["eps"] = pool["eps"].apply(lambda v: list(v) if isinstance(v, tuple) else [0.0] * horizon)
    pool["plays"] = pool["plays"].apply(
        lambda v: list(v) if isinstance(v, tuple) else [0.0] * horizon
    )
    pool["haul"] = pool["haul"].fillna(0.0)
    return pool


def _rows(pool: pd.DataFrame, gw: int, manager: str, last_seen: dict[int, dict]) -> list[dict]:
    """The players available at ``gw`` as plain dicts the optimisers understand."""
    rows = pool[pool["GW"] == gw]
    out = []
    for r in rows.itertuples(index=False):
        eps = list(r.eps)
        if manager == "crowd":
            own = float(r.ownership or 0.0) / 1e5
            eps = [own] * len(eps)
        p = {
            "code": int(r.code), "web_name": r.web_name, "position": r.position, "team": r.team,
            "price": float(r.price), "expected_points": float(eps[0]), "eps": eps,
            "plays": list(r.plays), "haul": float(r.haul), "availability": 1.0,
        }  # fmt: skip
        last_seen[p["code"]] = p
        out.append(p)
    return out


# ---------------------------------------------------------------- valuation
def week_value(squad: list[dict], k: int, bench_weight: float) -> float:
    eleven = _best_eleven(squad, None, k)
    starters = set(eleven.starters)
    return eleven.points + bench_weight * sum(_ep(p, k) for p in squad if p["code"] not in starters)


def _captain_ep(squad: list[dict], k: int) -> float:
    """The projected points of the man who would wear the armband in week ``k``."""
    eleven = _best_eleven(squad, None, k)
    captain = next((p for p in squad if p["code"] == eleven.captain), None)
    return _ep(captain, k) if captain else 0.0


def run_value(squad: list[dict], weights, bench_weight: float, start: int = 0) -> float:
    return sum(
        weights[k - start] * week_value(squad, k, bench_weight) for k in range(start, len(weights))
    )


def _solve_squad(pool: list[dict], budget: float, score) -> list[dict]:
    frame = pd.DataFrame(
        [
            {"code": p["code"], "position": p["position"], "team": p["team"],
             "price_tenths": int(round(p["price"] * 10)), "projected": score(p), "actual": 0.0}
            for p in pool
        ]
    )  # fmt: skip
    chosen = pick_squad(frame, budget_tenths=int(round(budget * 10)))
    by = {p["code"]: p for p in pool}
    return [by[int(c)] for c in chosen["code"]]


def _fix_club_limit(
    squad: list[dict], rows: list[dict], bank: float, free: int
) -> tuple[list[dict], float, list[tuple[int, int]], int]:
    """A mid-season move can leave four men from one club in a squad that was legal
    when it was picked; FPL makes the manager sell one. The lowest-projected of them
    goes for the best replacement the money allows, on a free transfer if one is
    left and for a hit otherwise."""
    forced: list[tuple[int, int]] = []
    hit = 0
    while True:
        counts: dict[str, int] = {}
        for p in squad:
            counts[p["team"]] = counts.get(p["team"], 0) + 1
        over = [club for club, n in counts.items() if n > MAX_PER_CLUB]
        if not over:
            return squad, bank, forced, hit
        club = over[0]
        seller = min((p for p in squad if p["team"] == club), key=lambda p: _ep(p, 0))
        owned = {p["code"] for p in squad}
        options = [
            p
            for p in rows
            if p["position"] == seller["position"]
            and p["code"] not in owned
            and counts.get(p["team"], 0) < MAX_PER_CLUB
            and p["price"] <= seller["price"] + bank + 1e-6
        ]
        if not options:
            return squad, bank, forced, hit  # nothing affordable: leave it (rare)
        buyer = max(options, key=lambda p: _ep(p, 0))
        squad = [p for p in squad if p["code"] != seller["code"]] + [buyer]
        bank = round(bank + seller["price"] - buyer["price"], 1)
        forced.append((seller["code"], buyer["code"]))
        if len(forced) > free:
            hit += TRANSFER_HIT


# ---------------------------------------------------------------- the season
def play_season(
    season: str,
    *,
    manager: str = "model",
    strategy: Strategy = BASELINE,
    model: str = "component",
    first_gameweek: int = 6,
    pool: pd.DataFrame | None = None,
    actual: pd.DataFrame | None = None,
    noise: float = 0.0,
    seed: int = 0,
) -> dict:
    """Play ``season`` from ``first_gameweek``; return the week-by-week log.

    ``noise`` scales every projection by an independent ``N(1, noise)`` factor per
    player-gameweek: one season under one strategy is a single noisy draw (a squad
    that diverges at one decision can finish forty points apart), so strategies are
    compared as averages over several seeds rather than single runs.
    """
    rng = np.random.default_rng(seed)
    if pool is None or actual is None:
        pool, actual = load_season(season, model)
    if manager == "crowd":
        strategy = GREEDY  # the template manager: one free transfer a week, no chips
    gameweeks = sorted(int(g) for g in pool["GW"].unique() if g >= first_gameweek)
    weights = strategy.weights
    bw = strategy.bench_weight
    chips = {i: w for i, w in enumerate(chip_windows(season))} if strategy.chips else {}
    used: set[int] = set()
    squad: list[dict] = []
    bank, free = 0.0, 0
    last_seen: dict[int, dict] = {}
    weeks: list[Week] = []
    hits_total = 0

    def available(chip: str, gw: int) -> int | None:
        for i, (name, a, b) in chips.items():
            if name == chip and i not in used and a <= gw <= b:
                return i
        return None

    def window_end(idx: int) -> int:
        return chips[idx][2]

    def refresh(current: list[dict], rows: list[dict]) -> list[dict]:
        """The squad with this gameweek's projections (a player without a fixture
        keeps his identity and price and projects nothing)."""
        by = {p["code"]: p for p in rows}
        blank = {"eps": [0.0] * len(weights), "plays": [0.0] * len(weights), "haul": 0.0,
                 "expected_points": 0.0}  # fmt: skip
        return [by.get(p["code"], {**last_seen.get(p["code"], p), **blank}) for p in current]

    for gw in gameweeks:
        rows = _rows(pool, gw, manager, last_seen)
        if noise:
            for p in rows:
                factor = float(rng.normal(1.0, noise))
                p["eps"] = [e * factor for e in p["eps"]]
                p["expected_points"] = p["eps"][0]
        names = {p["code"]: p["web_name"] for p in rows}
        by_code = {p["code"]: p for p in rows}
        chip: str | None = None
        moves: list[str] = []
        hit = 0
        temporary: list[dict] | None = None  # a Free Hit squad, for this week only

        weighted = lambda p: sum(weights[k] * _ep(p, k) for k in range(len(weights)))  # noqa: E731
        if not squad:
            squad = _solve_squad(rows, 100.0, weighted)
            bank = round(100.0 - sum(p["price"] for p in squad), 1)
        else:
            free = min(MAX_FREE_TRANSFERS, free + 1)
            squad = refresh(squad, rows)
            squad, bank, forced, forced_hit = _fix_club_limit(squad, rows, bank, free)
            if forced:
                free = max(0, free - len(forced)) if forced_hit == 0 else 0
                hit += forced_hit
                hits_total += forced_hit
                moves.extend(
                    f"{names.get(o, '?')} -> {names.get(i, '?')} (club limit)" for o, i in forced
                )
            value = sum(p["price"] for p in squad)
            budget = value + bank
            # -- wildcard: rebuild when the rebuilt squad is worth enough more
            wc = available("wildcard", gw)
            if wc is not None and gw - gameweeks[0] < strategy.wc_settle:
                wc = None
            if wc is not None:
                rebuilt = _solve_squad(rows, budget, weighted)
                gain = run_value(rebuilt, weights, bw) - run_value(squad, weights, bw)
                closing = gw >= window_end(wc) - 2
                log.debug("GW%s wildcard gain %.1f (closing=%s)", gw, gain, closing)
                if gain >= strategy.wc_min_gain or (closing and gain >= strategy.wc_late_gain):
                    chip = "wildcard"
                    used.add(wc)
                    changed = len(
                        [p for p in rebuilt if p["code"] not in {q["code"] for q in squad}]
                    )
                    moves.append(f"wildcard: {changed} changes")
                    squad = rebuilt
                    bank = round(budget - sum(p["price"] for p in squad), 1)
                    free = 1
            # -- free hit: this week's best squad, if this is the week to spend it
            if chip is None and available("freehit", gw) is not None:
                here = _solve_squad(rows, budget, lambda p: _ep(p, 0))
                gain = week_value(here, 0, 0.0) - week_value(squad, 0, 0.0)
                later = 0.0
                for k in range(1, len(weights)):
                    alt = _solve_squad(rows, budget, lambda p, k=k: _ep(p, k))
                    later = max(later, week_value(alt, k, 0.0) - week_value(squad, k, 0.0))
                log.debug("GW%s free hit gain %.1f, best later %.1f", gw, gain, later)
                if gain >= strategy.fh_min_gain and gain >= later:
                    chip = "freehit"
                    used.add(available("freehit", gw))
                    temporary = here
                    moves.append("free hit")
            # -- ordinary transfers
            if chip is None:
                if strategy.planner == "milp":
                    try:
                        plan = plan_transfers(
                            squad, rows, bank=bank, free_transfers=free, first_gameweek=gw,
                            weights=weights, bench_weight=bw,
                            hit=int(strategy.hit_cost) if strategy.hits else 10_000,
                        )  # fmt: skip
                        first = plan.weeks[0]
                        outs = [p["code"] for p in first.transfers_out]
                        ins = [p["code"] for p in first.transfers_in]
                        paid = max(0, len(ins) - free)
                        extra = paid * TRANSFER_HIT if strategy.hits else 0
                    except Exception as exc:  # noqa: BLE001
                        log.warning("GW%s: solver failed (%s); holding", gw, exc)
                        outs, ins, extra = [], [], 0
                else:
                    own = pd.DataFrame(squad)
                    options = suggest_transfers(
                        own, pd.DataFrame(rows), bank=bank, top_n=3, team_value=value,
                        free_transfers=free, metric="expected_points",
                    )  # fmt: skip
                    options = [
                        m
                        for m in options
                        if m["transfers"] <= free and m["gain"] >= strategy.min_gain
                    ]
                    outs = options[0]["out"] if options else []
                    ins = options[0]["in"] if options else []
                    extra = 0
                if ins:
                    cost = sum(by_code[c]["price"] for c in ins) - sum(
                        next(p["price"] for p in squad if p["code"] == c) for c in outs
                    )
                    squad = [p for p in squad if p["code"] not in outs] + [by_code[c] for c in ins]
                    bank = round(bank - cost, 1)
                    free = max(0, free - len(ins)) if extra == 0 else 0
                    moves.extend(
                        f"{names.get(o, '?')} -> {names.get(i, '?')}"
                        for o, i in zip(outs, ins, strict=True)
                    )
                    hit += extra
                    hits_total += extra

        playing = temporary if temporary is not None else squad
        # -- bench boost / triple captain: one chip a week, and only in the best week
        eleven = _best_eleven(playing, None, 0)
        starters = set(eleven.starters)
        if chip is None and available("bboost", gw) is not None:
            bench_now = sum(_ep(p, 0) for p in playing if p["code"] not in starters)
            later = max(
                (sum(_ep(p, k) for p in playing if p["code"] not in set(_best_eleven(playing, None, k).starters))
                 for k in range(1, len(weights))),
                default=0.0,
            )  # fmt: skip
            log.debug("GW%s bench %.1f, best later %.1f", gw, bench_now, later)
            if bench_now >= strategy.bb_min and bench_now >= later:
                chip = "bboost"
                used.add(available("bboost", gw))
        if chip is None and available("3xc", gw) is not None:
            cap_now = _captain_ep(playing, 0)
            later = max((_captain_ep(playing, k) for k in range(1, len(weights))), default=0.0)
            log.debug("GW%s captain %.1f, best later %.1f", gw, cap_now, later)
            if cap_now >= strategy.tc_min and cap_now >= later:
                chip = "3xc"
                used.add(available("3xc", gw))

        points, starters_played, captain, bench_codes, _, projected = _play(playing, actual, gw)
        if chip == "bboost":
            points += sum(
                int(actual.loc[(c, gw), "total_points"]) if (c, gw) in actual.index else 0
                for c in bench_codes
            )
        if chip == "3xc" and captain in starters_played and (captain, gw) in actual.index:
            points += int(actual.loc[(captain, gw), "total_points"])
        points -= hit
        weeks.append(
            Week(
                gw,
                int(points),
                moves,
                names.get(captain, "?"),
                chip,
                hit,
                bank,
                round(projected, 1),
            )
        )

    total = sum(w.points for w in weeks)
    return {
        "season": season,
        "manager": manager,
        "strategy": dict(strategy.__dict__),
        "gameweeks": [w.__dict__ for w in weeks],
        "total": total,
        "per_gameweek": round(total / len(weeks), 1) if weeks else 0.0,
        "hits": hits_total,
        "chips": [(w.gameweek, w.chip) for w in weeks if w.chip],
    }


def load_averages(season: str) -> dict[int, int]:
    """FPL's own average score per gameweek (``average_entry_score``), where archived."""
    if not AVERAGES.exists():
        return {}
    frame = pd.read_csv(AVERAGES)
    frame = frame[frame["season"] == season]
    return {int(r.GW): int(r.average) for r in frame.itertuples()}


def replay(season: str, strategy: Strategy = BASELINE, model: str = "component") -> dict:
    """The page's season replay: the model manager with the full rule set, the crowd
    (template) manager with one free transfer a week, and the real average where
    the archive has it, all over the same gameweeks."""
    pool, actual = load_season(season, model)
    out = {
        "model": play_season(season, strategy=strategy, model=model, pool=pool, actual=actual),
        "crowd": play_season(
            season, manager="crowd", strategy=GREEDY, model=model, pool=pool, actual=actual
        ),
    }
    averages = load_averages(season)
    weeks = [w["gameweek"] for w in out["model"]["gameweeks"]]
    if averages and all(gw in averages for gw in weeks):
        pts = [averages[gw] for gw in weeks]
        out["average"] = {
            "season": season,
            "manager": "average",
            "gameweeks": [{"gameweek": gw, "points": p} for gw, p in zip(weeks, pts, strict=True)],
            "total": sum(pts),
            "per_gameweek": round(sum(pts) / len(pts), 1),
        }
    return out


def compare(season: str, strategies: dict[str, Strategy], model: str = "component") -> pd.DataFrame:
    """Several strategies on one season, loaded once."""
    pool, actual = load_season(season, model)
    out = []
    for name, strategy in strategies.items():
        r = play_season(season, strategy=strategy, model=model, pool=pool, actual=actual)
        out.append(
            {
                "strategy": name,
                "total": r["total"],
                "per_gw": r["per_gameweek"],
                "hits": r["hits"],
                "chips": r["chips"],
            }
        )
    return pd.DataFrame(out)


__all__ = [
    "Strategy",
    "BASELINE",
    "NO_CHIPS",
    "GREEDY",
    "play_season",
    "compare",
    "replay",
    "load_averages",
    "chip_windows",
    "load_season",
]

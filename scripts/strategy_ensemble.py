"""Strategies compared as averages over noisy re-runs, not single seasons.

    python scripts/strategy_ensemble.py 2024-25 [seeds] [noise] [strategies] [tag] [model]

One replay of one season is a single draw: two strategies that differ by forty
points may just have diverged at one coin-flip transfer. Each strategy is played
``seeds`` times with every projection scaled by an independent N(1, noise) factor,
and the mean and spread are reported against the crowd manager and, where FPL's own
average is archived, the real average. Saved to ``logs/ensemble_<season>.csv``.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from fpl.config import PROJECT_ROOT
from fpl.evaluate.season import GREEDY, Strategy, load_averages, load_season, play_season

STRATEGIES: dict[str, Strategy] = {
    "greedy_1ft": GREEDY,
    "greedy_chips": Strategy(planner="greedy", hits=False, chips=True),
    "milp_nohits_nochips": Strategy(hits=False, chips=False),
    "milp_nohits_chips": Strategy(hits=False, chips=True),
    "milp_hit8_chips": Strategy(hit_cost=8, chips=True),
    "milp_hit4_chips": Strategy(hit_cost=4, chips=True),
    # the tuned rules (module defaults) and what each knob costs
    "tuned": Strategy(),
    "tuned_nohits": Strategy(hits=False),
    "tuned_hit4": Strategy(hit_cost=4),
    "tuned_hit12": Strategy(hit_cost=12),
    "loose_chips": Strategy(wc_min_gain=10, wc_late_gain=2, wc_settle=0, fh_min_gain=12, tc_min=9),
    "tuned_flat": Strategy(weights=(1.0, 1.0, 1.0, 1.0, 1.0)),
    "tuned_bb10": Strategy(bb_min=10),
}


def main() -> None:
    season = sys.argv[1] if len(sys.argv) > 1 else "2024-25"
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else list(STRATEGIES)
    model = sys.argv[6] if len(sys.argv) > 6 else "component"
    pool, actual = load_season(season, model)
    averages = load_averages(season)
    crowd = play_season(
        season, manager="crowd", strategy=GREEDY, model=model, pool=pool, actual=actual
    )
    weeks = [w["gameweek"] for w in crowd["gameweeks"]]
    n = len(weeks)
    avg_total = sum(averages[gw] for gw in weeks) if all(gw in averages for gw in weeks) else None
    print(
        f"{season} GW{weeks[0]}-{weeks[-1]} ({n} weeks): crowd {crowd['total']}, average {avg_total}"
    )
    rows = []
    for name in only:
        strategy = STRATEGIES[name]
        totals, hits, chips = [], [], []
        t0 = time.time()
        for seed in range(seeds):
            r = play_season(
                season, strategy=strategy, pool=pool, actual=actual, noise=noise, seed=seed
            )
            totals.append(r["total"])
            hits.append(r["hits"])
            chips.append(" ".join(f"{c}@{gw}" for gw, c in r["chips"]))
            print(f"  {name} seed {seed}: {r['total']} (hits {r['hits']}) {chips[-1]}", flush=True)
        mean = float(np.mean(totals))
        row = {
            "strategy": name,
            "mean": round(mean),
            "sd": round(float(np.std(totals)), 1),
            "min": min(totals),
            "max": max(totals),
            "per_gw": round(mean / n, 1),
            "vs_crowd": round((mean - crowd["total"]) / n, 2),
            "vs_avg": round((mean - avg_total) / n, 2) if avg_total else None,
            "hits": round(float(np.mean(hits)), 1),
            "secs": round(time.time() - t0),
        }
        rows.append(row)
        print(row, flush=True)
    table = pd.DataFrame(rows)
    tag = sys.argv[5] if len(sys.argv) > 5 else "all"
    table.to_csv(PROJECT_ROOT / "logs" / f"ensemble_{season}_{tag}.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

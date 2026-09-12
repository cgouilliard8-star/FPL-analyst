"""Play one archived season under many strategies and tabulate the points.

    python scripts/strategy_grid.py 2024-25 [grid-name]

Runs on the cached walk-forward projections (``fpl backtest`` must have written
``component_<season>.parquet`` and its ``_horizon`` companion). Each strategy takes
20-60 seconds; results are printed as they finish and saved to
``logs/grid_<season>_<grid>.csv``.
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from fpl.config import PROJECT_ROOT
from fpl.evaluate.season import GREEDY, Strategy, load_averages, load_season, play_season

GRIDS: dict[str, dict[str, Strategy]] = {
    "hits": {
        "greedy": GREEDY,
        "milp_nohits": Strategy(hits=False, chips=False),
        "milp_hit4": Strategy(hit_cost=4, chips=False),
        "milp_hit6": Strategy(hit_cost=6, chips=False),
        "milp_hit8": Strategy(hit_cost=8, chips=False),
        "milp_hit12": Strategy(hit_cost=12, chips=False),
    },
    "chips": {
        "chips_hit8": Strategy(hit_cost=8),
        "chips_nohits": Strategy(hits=False),
        "chips_hit8_wc15": Strategy(hit_cost=8, wc_min_gain=15.0),
        "chips_hit8_wc6": Strategy(hit_cost=8, wc_min_gain=6.0),
        "chips_hit8_fh8": Strategy(hit_cost=8, fh_min_gain=8.0),
        "chips_hit8_fh18": Strategy(hit_cost=8, fh_min_gain=18.0),
        "chips_hit8_bb10": Strategy(hit_cost=8, bb_min=10.0),
        "chips_hit8_bb18": Strategy(hit_cost=8, bb_min=18.0),
        "chips_hit8_tc7": Strategy(hit_cost=8, tc_min=7.0),
        "chips_hit8_tc11": Strategy(hit_cost=8, tc_min=11.0),
    },
    "weights": {
        "w_flat5": Strategy(hit_cost=8, weights=(1.0, 1.0, 1.0, 1.0, 1.0)),
        "w_steep": Strategy(hit_cost=8, weights=(1.0, 0.7, 0.5, 0.35, 0.25)),
        "w_default": Strategy(hit_cost=8),
        "w_short3": Strategy(hit_cost=8, weights=(1.0, 0.8, 0.6)),
        "bench0": Strategy(hit_cost=8, bench_weight=0.0),
        "bench30": Strategy(hit_cost=8, bench_weight=0.3),
    },
}


def main() -> None:
    season = sys.argv[1] if len(sys.argv) > 1 else "2024-25"
    grid = sys.argv[2] if len(sys.argv) > 2 else "hits"
    pool, actual = load_season(season)
    averages = load_averages(season)
    crowd = play_season(season, manager="crowd", strategy=GREEDY, pool=pool, actual=actual)
    weeks = [w["gameweek"] for w in crowd["gameweeks"]]
    avg_total = sum(averages.get(gw, 0) for gw in weeks) if averages else None
    print(f"{season} GW{weeks[0]}-{weeks[-1]}: crowd {crowd['total']}, average {avg_total}")
    rows = []
    for name, strategy in GRIDS[grid].items():
        t0 = time.time()
        r = play_season(season, strategy=strategy, pool=pool, actual=actual)
        row = {
            "strategy": name,
            "total": r["total"],
            "per_gw": r["per_gameweek"],
            "vs_crowd": round((r["total"] - crowd["total"]) / len(weeks), 2),
            "vs_avg": round((r["total"] - avg_total) / len(weeks), 2) if avg_total else None,
            "hits": r["hits"],
            "chips": " ".join(f"{c}@{gw}" for gw, c in r["chips"]),
            "secs": round(time.time() - t0),
        }
        rows.append(row)
        print(row, flush=True)
    table = pd.DataFrame(rows)
    out = PROJECT_ROOT / "logs" / f"grid_{season}_{grid}.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

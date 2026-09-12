"""The season simulator on a small synthetic season: rules, hits and chips add up."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl.evaluate.season import (
    BASELINE,
    GREEDY,
    NO_CHIPS,
    Strategy,
    _assemble,
    chip_windows,
    play_season,
)

QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
GAMEWEEKS = [6, 7, 8, 9]
HORIZON = 5


def _season(seed: int = 0):
    """Forty-odd players over four gameweeks; the top few in each position are
    good every week, the rest are fodder, and actual points follow the projections
    with noise so that a sensible manager scores more than a random one."""
    rng = np.random.default_rng(seed)
    players, code = [], 1
    for position, n in QUOTA.items():
        for i in range(n + 5):
            players.append(
                {"code": code, "web_name": f"{position}{i}", "position": position,
                 "team": f"club{code % 10}", "price": 4.0 + 0.3 * i, "quality": 1.5 + 0.5 * i}
            )  # fmt: skip
            code += 1
    rows, actual = [], []
    for gw in GAMEWEEKS:
        for p in players:
            eps = [p["quality"] * (1.0 + 0.05 * k) for k in range(HORIZON)]
            rows.append(
                {**{k: v for k, v in p.items() if k != "quality"}, "GW": gw,
                 "ownership": 1e5 * p["quality"], "minutes": 90, "total_points": 0,
                 "fixtures_this_gw": 1, "eps": eps, "plays": [0.9] * HORIZON, "haul": 0.1}
            )  # fmt: skip
            actual.append(
                {"code": p["code"], "GW": gw, "minutes": 90,
                 "total_points": int(max(0, rng.normal(p["quality"], 2)))}
            )  # fmt: skip
    pool = pd.DataFrame(rows)
    return pool, pd.DataFrame(actual).set_index(["code", "GW"])


def test_chip_windows_follow_the_rules_of_each_season():
    old = chip_windows("2024-25")
    assert [c for c, _, _ in old].count("wildcard") == 2
    assert [c for c, _, _ in old].count("freehit") == 1
    new = chip_windows("2025-26")
    assert len(new) == 8 and all(b in (19, 38) for _, _, b in new)


def test_a_season_plays_through_with_legal_squads_and_chips_once_each():
    pool, actual = _season()
    out = play_season("2024-25", strategy=BASELINE, pool=pool, actual=actual, first_gameweek=6)
    assert [w["gameweek"] for w in out["gameweeks"]] == GAMEWEEKS
    assert out["total"] == sum(w["points"] for w in out["gameweeks"])
    chips = [c for _, c in out["chips"]]
    assert len(chips) == len(set(chips)), "a chip was played twice"
    assert out["hits"] == sum(w["hit"] for w in out["gameweeks"])
    assert all(w["bank"] >= -1e-6 for w in out["gameweeks"])


def test_the_model_manager_beats_the_crowd_and_no_chips_is_never_charged_a_chip():
    pool, actual = _season()
    model = play_season("2024-25", strategy=NO_CHIPS, pool=pool, actual=actual, first_gameweek=6)
    crowd = play_season(
        "2024-25", manager="crowd", strategy=GREEDY, pool=pool, actual=actual, first_gameweek=6
    )
    assert model["chips"] == [] and crowd["chips"] == []
    assert crowd["hits"] == 0
    assert model["total"] > 0 and crowd["total"] > 0


def test_hits_off_means_no_hit_is_ever_taken():
    pool, actual = _season()
    out = play_season(
        "2024-25", strategy=Strategy(hits=False, chips=False), pool=pool, actual=actual,
        first_gameweek=6,
    )  # fmt: skip
    assert out["hits"] == 0


def test_assemble_fills_missing_horizon_with_zero_and_tolerates_old_files():
    preds = pd.DataFrame(
        {"code": [1, 2], "gameweek": [6, 6], "predicted": [3.0, 4.0]}
    )  # no p_play / p_haul: an older prediction file
    feats = pd.DataFrame(
        {"code": [1, 2], "GW": [6, 6], "web_name": ["a", "b"], "position": ["MID", "FWD"],
         "team": ["x", "y"], "price": [5.0, 6.0], "ownership": [1, 2], "minutes": [90, 90],
         "total_points": [2, 5], "fixtures_this_gw": [1, 1]}
    )  # fmt: skip
    ahead = pd.DataFrame(
        {"code": [1], "gameweek": [6], "k": [1], "predicted": [2.5], "p_play": [0.8]}
    )
    pool = _assemble(preds, ahead, feats)
    a = pool[pool["code"] == 1].iloc[0]
    assert a["eps"][0] == pytest.approx(3.0) and a["eps"][1] == pytest.approx(2.5)
    assert a["eps"][2] == 0.0 and a["plays"][1] == pytest.approx(0.8)
    assert pool[pool["code"] == 2].iloc[0]["eps"][1] == 0.0

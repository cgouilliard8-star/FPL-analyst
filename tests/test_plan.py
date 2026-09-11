"""The multi-week transfer solver on a small, hand-checkable pool."""

from __future__ import annotations

import pytest

from fpl.optimise.plan import plan_transfers

QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def _pool():
    players, code = [], 1
    for position, n in QUOTA.items():
        for i in range(n + 4):
            eps = [2.0 + 0.1 * i] * 5
            players.append(
                {"code": code, "web_name": f"{position}{i}", "position": position,
                 "team": f"club{code % 9}", "price": 4.5 + 0.1 * i, "eps": eps}
            )  # fmt: skip
            code += 1
    return players


def _squad(pool):
    out = []
    for position, n in QUOTA.items():
        out.extend([p for p in pool if p["position"] == position][:n])
    return out


def test_a_dead_player_is_sold_for_a_live_one_within_the_free_transfer():
    pool = _pool()
    squad = _squad(pool)
    dead = [p for p in squad if p["position"] == "MID"][0]
    dead["eps"] = [0.0] * 5
    star = [p for p in pool if p["position"] == "MID" and p not in squad][0]
    star["eps"] = [8.0] * 5
    plan = plan_transfers(squad, pool, bank=20.0, free_transfers=1, first_gameweek=10)
    first = plan.weeks[0]
    assert [p["code"] for p in first.transfers_out] == [dead["code"]]
    assert [p["code"] for p in first.transfers_in] == [star["code"]]
    assert first.hit == 0
    assert plan.objective > plan.hold
    assert first.captain == star["code"]  # the armband goes to the best attacker
    assert len(first.starters) == 11 and len(first.bench) == 4


def test_no_worthwhile_move_means_no_move():
    pool = _pool()
    squad = _squad(pool)
    for p in squad:
        p["eps"] = [9.0] * 5  # already the best fifteen by a mile
    plan = plan_transfers(squad, pool, bank=0.0, free_transfers=2, first_gameweek=3)
    assert all(not w.transfers_in for w in plan.weeks)
    assert plan.objective == pytest.approx(plan.hold, abs=0.5)


def test_hits_are_charged_and_free_transfers_bank():
    pool = _pool()
    squad = _squad(pool)
    for p in squad:
        if p["position"] == "FWD":
            p["eps"] = [0.0] * 5
    for p in pool:
        if p["position"] == "FWD" and p not in squad:
            p["eps"] = [9.0] * 5
    plan = plan_transfers(squad, pool, bank=30.0, free_transfers=1, first_gameweek=1)
    first = plan.weeks[0]
    assert len(first.transfers_in) >= 2 and first.hit >= 4
    assert first.free_after == 1  # a hit resets you to one free transfer

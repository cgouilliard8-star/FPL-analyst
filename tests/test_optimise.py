"""The optimiser must obey FPL's rules exactly. A squad that breaks one is worthless."""

import pandas as pd
import pytest

from fpl.config import BUDGET_TENTHS, MAX_PER_CLUB, SQUAD_QUOTA, SQUAD_SIZE, XI_MAX, XI_MIN, XI_SIZE
from fpl.optimise.squad import pick_squad, squad_points


def make_pool(n_per_club: int = 6) -> pd.DataFrame:
    """A synthetic league: 10 clubs, every position, varied price and projection."""
    rows, code = [], 0
    positions = ["GK", "DEF", "DEF", "MID", "MID", "FWD"][:n_per_club]
    for club in range(10):
        for slot, position in enumerate(positions):
            code += 1
            rows.append(
                {
                    "code": code,
                    "position": position,
                    "team": f"club{club}",
                    "price_tenths": 40 + slot * 5 + club,
                    "projected": (code % 7) + slot * 0.5,
                    "actual": float(code % 5),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def squad() -> pd.DataFrame:
    return pick_squad(make_pool())


def test_squad_has_exactly_fifteen(squad):
    assert len(squad) == SQUAD_SIZE


def test_squad_respects_positional_quota(squad):
    counts = squad["position"].value_counts().to_dict()
    assert counts == SQUAD_QUOTA


def test_squad_is_within_budget(squad):
    assert squad["price_tenths"].sum() <= BUDGET_TENTHS


def test_no_more_than_three_from_one_club(squad):
    assert squad["team"].value_counts().max() <= MAX_PER_CLUB


def test_starting_eleven_is_eleven(squad):
    assert squad["is_starter"].sum() == XI_SIZE


def test_formation_is_legal(squad):
    starters = squad[squad["is_starter"]]["position"].value_counts().to_dict()
    for position in ("GK", "DEF", "MID", "FWD"):
        count = starters.get(position, 0)
        assert XI_MIN[position] <= count <= XI_MAX[position], f"{position}: {count}"


def test_exactly_one_captain_and_he_starts(squad):
    assert squad["is_captain"].sum() == 1
    assert squad.loc[squad["is_captain"], "is_starter"].all()


def cheapest_legal_spend(pool: pd.DataFrame) -> int:
    """Lower bound on what any legal squad must cost, for choosing test budgets."""
    return int(
        sum(
            pool[pool["position"] == position]
            .nsmallest(quota, "price_tenths")["price_tenths"]
            .sum()
            for position, quota in SQUAD_QUOTA.items()
        )
    )


def test_tighter_budget_produces_a_cheaper_squad():
    """A binding budget constraint must actually bind."""
    pool = make_pool()
    floor = cheapest_legal_spend(pool)
    generous = pick_squad(pool, budget_tenths=BUDGET_TENTHS)
    assert generous["price_tenths"].sum() > floor, "fixture must leave room to economise"

    tight = floor + 10
    frugal = pick_squad(pool, budget_tenths=tight)
    assert frugal["price_tenths"].sum() <= tight
    assert frugal["price_tenths"].sum() < generous["price_tenths"].sum()


def test_impossible_budget_is_reported_not_silently_ignored():
    """Below the cheapest legal squad there is no answer, and pretending otherwise
    would hand the user an invalid team."""
    pool = make_pool()
    with pytest.raises(RuntimeError, match="Infeasible"):
        pick_squad(pool, budget_tenths=cheapest_legal_spend(pool) - 50)


def test_squad_points_doubles_the_captain(squad):
    starters = squad[squad["is_starter"]]
    captain_score = squad.loc[squad["is_captain"], "actual"].iloc[0]
    assert squad_points(squad) == pytest.approx(starters["actual"].sum() + captain_score)


def test_pool_missing_columns_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        pick_squad(pd.DataFrame([{"code": 1, "position": "GK"}]))


def test_duplicate_players_are_rejected():
    pool = make_pool()
    with pytest.raises(ValueError, match="duplicate"):
        pick_squad(pd.concat([pool, pool.head(1)], ignore_index=True))

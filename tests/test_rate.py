"""Team rating and transfer suggestions must obey the rules and reward the team."""

import pandas as pd
import pytest

from fpl.optimise import rate


def pool() -> pd.DataFrame:
    rows, code = [], 0
    for club in range(8):
        for position, n in (("GK", 2), ("DEF", 3), ("MID", 3), ("FWD", 2)):
            for i in range(n):
                code += 1
                rows.append(
                    {
                        "code": code,
                        "web_name": f"p{code}",
                        "position": position,
                        "team": f"club{club}",
                        "price": 4.0 + i + club * 0.3,
                        "expected_points": 1.0 + (code * 7 % 11) / 3 + i,
                        "availability": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def legal_squad(frame: pd.DataFrame) -> pd.DataFrame:
    picks, clubs = [], {}
    for position, quota in (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        for _, row in frame[frame["position"] == position].iterrows():
            if len([p for p in picks if p["position"] == position]) == quota:
                break
            if clubs.get(row["team"], 0) >= 3:
                continue
            picks.append(row)
            clubs[row["team"]] = clubs.get(row["team"], 0) + 1
    return pd.DataFrame(picks)


def test_every_formation_is_legal():
    for f in rate.FORMATIONS:
        assert sum(f.values()) == 11
        assert f["GK"] == 1
    assert len(rate.FORMATIONS) == 8


def test_best_eleven_captain_is_the_top_projection_not_the_keeper():
    squad = legal_squad(pool())
    eleven = rate.best_eleven(squad)
    top = squad.loc[squad["code"].isin(eleven.starters)].sort_values("expected_points").iloc[-1]
    assert eleven.captain == int(top["code"])
    assert squad.loc[squad["code"] == eleven.captain, "position"].iloc[0] != "GK" or (
        squad["expected_points"].max() == top["expected_points"]
    )


def test_best_eleven_points_include_the_captain_twice():
    squad = legal_squad(pool())
    eleven = rate.best_eleven(squad)
    starters = squad[squad["code"].isin(eleven.starters)]
    captain_pts = float(squad.loc[squad["code"] == eleven.captain, "expected_points"].iloc[0])
    assert eleven.points == pytest.approx(starters["expected_points"].sum() + captain_pts)
    assert len(eleven.starters) == 11 and len(eleven.bench) == 4


def test_rating_is_relative_to_optimal_and_uncapped():
    squad = legal_squad(pool())
    r = rate.rate_squad(squad, optimal_points=rate.best_eleven(squad).points / 2)
    assert r["rating"] == pytest.approx(200.0)


def test_illegal_squads_are_rejected():
    squad = legal_squad(pool())
    with pytest.raises(ValueError, match="15 players"):
        rate.best_eleven(squad.head(14))
    four_same_club = squad.copy()
    four_same_club.loc[four_same_club.index[:4], "team"] = "club0"
    with pytest.raises(ValueError, match="more than 3"):
        rate.best_eleven(four_same_club)


def test_suggestions_respect_position_budget_and_club_limit():
    frame = pool()
    squad = legal_squad(frame)
    moves = rate.suggest_transfers(squad, frame, bank=0.5, top_n=5)
    clubs = squad["team"].value_counts().to_dict()
    by_code = frame.set_index("code")
    for m in moves:
        out, inc = by_code.loc[m["out"]], by_code.loc[m["in"]]
        assert out["position"] == inc["position"]
        assert inc["price"] <= out["price"] + 0.5 + 1e-9
        if inc["team"] != out["team"]:
            assert clubs.get(inc["team"], 0) < 3
        assert m["in"] not in set(squad["code"])


def test_suggestions_are_ranked_by_team_gain_and_distinct():
    frame = pool()
    squad = legal_squad(frame)
    moves = rate.suggest_transfers(squad, frame, bank=5.0, top_n=5)
    assert len(moves) == 5
    assert [m["rank"] for m in moves] == [1, 2, 3, 4, 5]
    gains = [m["gain"] for m in moves]
    assert gains == sorted(gains, reverse=True)
    assert len({m["in"] for m in moves}) == 5
    base = rate.best_eleven(squad).points
    for m in moves:
        assert m["points_after"] == pytest.approx(base + m["gain"], abs=0.02)


def test_gain_is_measured_on_the_whole_team():
    """Swapping a bench player for a slightly better bench player gains nothing."""
    frame = pool()
    squad = legal_squad(frame)
    eleven = rate.best_eleven(squad)
    bench_code = eleven.bench[0]
    bench = squad[squad["code"] == bench_code].iloc[0]
    # a candidate marginally better than the bench player but still below the XI
    weakest_starter = squad[
        squad["code"].isin(eleven.starters) & (squad["position"] == bench["position"])
    ]["expected_points"].min()
    candidate = bench.copy()
    candidate["code"] = 9999
    candidate["team"] = "club9"
    candidate["expected_points"] = min(bench["expected_points"] + 0.1, weakest_starter - 0.05)
    moves = rate.suggest_transfers(squad, pd.DataFrame([candidate]), bank=10.0)
    assert moves == []


def test_rate_and_suggest_rejects_unknown_codes():
    with pytest.raises(ValueError, match="unknown player codes"):
        rate.rate_and_suggest([9001, 9002, 9003], pool())

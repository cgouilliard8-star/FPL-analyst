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


def test_suggestions_respect_position_cap_and_club_limit():
    frame = pool()
    squad = legal_squad(frame)
    value = float(squad["price"].sum())
    moves = rate.suggest_transfers(squad, frame, bank=0.5, top_n=5)
    clubs = squad["team"].value_counts().to_dict()
    by_code = frame.set_index("code")
    budget = rate.available_budget(squad.to_dict("records"), 0.5)
    for m in moves:
        assert len(m["out"]) == len(m["in"]) == m["transfers"]
        for out_code, in_code in zip(m["out"], m["in"], strict=True):
            assert by_code.loc[out_code, "position"] == by_code.loc[in_code, "position"]
            assert in_code not in set(squad["code"])
        assert m["value_after"] <= budget + 1e-6
        assert m["value_after"] == pytest.approx(value + m["cost_change"], abs=0.05)
        if m["transfers"] == 1:
            inc, out = by_code.loc[m["in"][0]], by_code.loc[m["out"][0]]
            if inc["team"] != out["team"]:
                assert clubs.get(inc["team"], 0) < 3


def test_suggestions_are_ranked_by_team_gain_and_distinct():
    frame = pool()
    squad = legal_squad(frame)
    moves = rate.suggest_transfers(squad, frame, bank=5.0, top_n=5)
    assert len(moves) == 5
    assert [m["rank"] for m in moves] == [1, 2, 3, 4, 5]
    gains = [m["gain"] for m in moves]
    assert gains == sorted(gains, reverse=True)
    signings = [c for m in moves for c in m["in"]]
    assert len(signings) == len(set(signings))
    base = rate.best_eleven(squad).points
    for m in moves:
        assert m["points_after"] == pytest.approx(base + m["gain"], abs=0.02)


def test_over_cap_upgrade_is_funded_by_a_second_transfer():
    """A star signing the bank cannot cover must arrive as a two-transfer move."""
    frame = pool()
    squad = legal_squad(frame)
    star = frame.iloc[0].copy()
    star["code"], star["team"], star["price"], star["expected_points"] = 9001, "club9", 7.5, 40.0
    star["web_name"] = "star"
    enriched = pd.concat([frame, star.to_frame().T], ignore_index=True)
    enriched["price"] = enriched["price"].astype(float)
    enriched["expected_points"] = enriched["expected_points"].astype(float)
    value = float(squad["price"].sum())
    moves = rate.suggest_transfers(squad, enriched, bank=0.0, team_value=value, top_n=5)
    star_moves = [m for m in moves if 9001 in m["in"]]
    assert star_moves, "the star should be reachable"
    assert star_moves[0]["transfers"] == 2
    assert star_moves[0]["value_after"] <= value + 1e-6


def test_team_value_raises_the_cap():
    frame = pool()
    squad = legal_squad(frame)
    value = float(squad["price"].sum())
    tight = rate.suggest_transfers(squad, frame, bank=0.0, team_value=value, top_n=50)
    loose = rate.suggest_transfers(squad, frame, bank=0.0, team_value=value + 3.0, top_n=50)
    assert max(m["value_after"] for m in loose) > max(m["value_after"] for m in tight)
    assert rate.available_budget(squad.to_dict("records"), 0.5, 102.3) == pytest.approx(102.8)


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


def test_best_eleven_can_be_pinned_to_a_formation():
    squad = legal_squad(pool())
    pinned = rate.best_eleven(squad, {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1})
    assert pinned.formation == {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1}
    assert pinned.points <= rate.best_eleven(squad).points + 1e-9


def test_rate_and_suggest_rejects_unknown_codes():
    with pytest.raises(ValueError, match="unknown player codes"):
        rate.rate_and_suggest([9001, 9002, 9003], pool())

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


def test_metric_switches_the_projection_the_team_is_rated_on():
    frame = pool()
    frame["ep5"] = frame["expected_points"] * 3 + (frame["code"] % 5)  # reshuffles the order
    squad = legal_squad(frame)
    one = rate.rate_squad(squad)
    five = rate.rate_squad(squad, metric="ep5")
    assert five["points"] > one["points"]
    assert five["points"] == pytest.approx(rate.best_eleven(squad, metric="ep5").points)
    with pytest.raises(ValueError, match="metric"):
        rate.rate_squad(squad, metric="ep9")


def test_transfers_beyond_the_free_ones_cost_four_points_and_rank_by_net():
    frame = pool()
    squad = legal_squad(frame)
    free = rate.suggest_transfers(squad, frame, bank=50.0, top_n=5)
    charged = rate.suggest_transfers(squad, frame, bank=50.0, top_n=5, free_transfers=0)
    assert all(m["hit"] == 0 and m["net"] == m["gain"] for m in free)
    assert all(m["hit"] == 4 * m["transfers"] for m in charged)
    assert all(m["net"] == pytest.approx(m["gain"] - m["hit"]) for m in charged)
    assert all(m["worth_it"] == (m["net"] > 0) for m in charged)
    nets = [m["net"] for m in charged]
    assert nets == sorted(nets, reverse=True)
    one_free = rate.suggest_transfers(squad, frame, bank=50.0, top_n=5, free_transfers=1)
    assert all(m["hit"] == 4 * (m["transfers"] - 1) for m in one_free)


def test_season_sim_autosubs_and_vice_captain():
    from fpl.evaluate import season_sim

    squad = legal_squad(pool())
    rows = rate._to_players(squad)
    eleven = rate._best_eleven(rows)
    gw = 1
    # everyone plays and scores 2, except the captain and one starting defender
    starters = squad[squad["code"].isin(eleven.starters)]
    absent_def = int(starters[starters["position"] == "DEF"]["code"].iloc[0])
    actual = pd.DataFrame(
        {
            "code": squad["code"],
            "GW": gw,
            "total_points": [0 if c in (eleven.captain, absent_def) else 2 for c in squad["code"]],
            "minutes": [0 if c in (eleven.captain, absent_def) else 90 for c in squad["code"]],
        }
    ).set_index(["code", "GW"])
    total, played, captain, bench, autosubs, _ = season_sim._play(rows, actual, gw)
    assert captain != eleven.captain and captain in played
    assert {s["out"] for s in autosubs} == {eleven.captain, absent_def}
    assert len(played) == 11 and set(played).isdisjoint(bench)
    # 11 players x 2, plus the vice captain doubled
    assert total == 11 * 2 + 2


def with_runs(frame: pd.DataFrame, horizon: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give every player a per-gameweek run; the total is the ep3 column."""
    rows = []
    for _, p in frame.iterrows():
        for k in range(horizon):
            ep = float(p["expected_points"]) * (1.0 + 0.3 * ((p["code"] + k) % 3 - 1))
            rows.append(
                {
                    "code": p["code"],
                    "offset": k,
                    "expected_points": ep,
                    "availability": 1.0,
                    "p_60": 0.9,
                }
            )
    fixtures = pd.DataFrame(rows)
    weighted = fixtures.assign(
        w=fixtures["expected_points"]
        * fixtures["offset"].map(dict(enumerate(rate.HORIZON_WEIGHTS)))
    )
    frame = frame.copy()
    frame["ep3"] = frame["code"].map(weighted.groupby("code")["w"].sum())
    return frame, fixtures


def test_horizon_scoring_repicks_the_eleven_every_gameweek():
    frame, fixtures = with_runs(pool())
    squad = legal_squad(frame)
    r = rate.rate_squad(squad, metric="ep3", fixtures=fixtures)
    assert len(r["lineups"]) == 3
    # a player benched next week can start later when his fixture is kinder
    starters = [set(lu["starters"]) for lu in r["lineups"]]
    assert any(starters[0] != s for s in starters[1:])
    # the weighted sum of the per-gameweek elevens (plus cover) is the score
    expected = sum(
        rate.HORIZON_WEIGHTS[k] * (lu["points"] + lu["cover"]) for k, lu in enumerate(r["lineups"])
    )
    assert r["points"] == pytest.approx(expected, abs=0.05)


def test_bench_cover_rewards_a_bench_that_would_play():
    frame = pool()
    squad = legal_squad(frame)
    players = rate._to_players(squad)
    eleven = rate._best_eleven(players)
    by = {p["code"]: p for p in players}
    # make one outfield starter a coin flip to play: the bench now matters
    risky = next(c for c in eleven.starters if by[c]["position"] != "GK")
    by[risky]["plays"] = [0.5]
    cover = rate._bench_cover(players, eleven, 0)
    assert cover > 0
    top_sub = max(
        (by[c] for c in eleven.bench if by[c]["position"] != "GK"), key=lambda p: p["eps"][0]
    )
    assert cover == pytest.approx(0.5 * top_sub["eps"][0], abs=0.15)
    assert rate._at_least([0.5, 0.5]) == pytest.approx([1.0, 0.75, 0.25])


def test_team_value_above_squad_value_is_headroom():
    frame = pool()
    squad = legal_squad(frame)
    players = rate._to_players(squad)
    value = sum(p["price"] for p in players)
    assert rate.available_budget(players, 0.0, team_value=value + 20) == pytest.approx(value + 20)
    assert rate.available_budget(players, 1.5, team_value=None) == pytest.approx(
        max(100.0, value) + 1.5
    )
    moves = rate.suggest_transfers(squad, frame, team_value=value + 20, top_n=5)
    assert any(m["cost_change"] > 0.5 for m in moves)


def test_over_budget_squad_only_gets_moves_that_fix_it():
    frame = pool()
    squad = legal_squad(frame)
    players = rate._to_players(squad)
    value = sum(p["price"] for p in players)
    moves = rate.suggest_transfers(squad, frame, team_value=value - 3, bank=0.0, top_n=5)
    assert moves and all(m["fixes_budget"] for m in moves)
    assert all(m["value_after"] <= value - 3 + 1e-9 for m in moves)

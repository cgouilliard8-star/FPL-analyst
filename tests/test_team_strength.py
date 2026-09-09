"""The opponent-adjusted club ratings: sensible, point-in-time, and cheap."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from fpl.models.team_strength import (
    FEATURES,
    attach_team_strength,
    fit_ratings,
    team_matches,
    win_probability,
)


def _league(seed: int = 0, rounds: int = 6) -> pd.DataFrame:
    """A four-club league where Alpha is clearly best and Delta clearly worst."""
    rng = np.random.default_rng(seed)
    quality = {"Alpha": 0.5, "Beta": 0.1, "Gamma": -0.1, "Delta": -0.5}
    rows = []
    fixture = 0
    day = pd.Timestamp("2024-08-17", tz="UTC")
    for r in range(rounds):
        clubs = list(quality)
        rng.shuffle(clubs)
        for h, a in ((clubs[0], clubs[1]), (clubs[2], clubs[3])):
            fixture += 1
            lam_h = np.exp(0.2 + quality[h] - quality[a] + 0.15)
            lam_a = np.exp(0.2 + quality[a] - quality[h])
            gh, ga = rng.poisson(lam_h), rng.poisson(lam_a)
            for team, opp, home in ((h, a, True), (a, h, False)):
                for player in range(3):  # three player rows per side
                    rows.append({
                        "season": "2024-25", "GW": r + 1, "fixture": fixture, "team": team,
                        "opponent": opp, "was_home": home, "kickoff_time": day,
                        "minutes": 90, "team_h_score": gh, "team_a_score": ga,
                        "expected_goals": (lam_h if home else lam_a) / 3,
                        "expected_goals_conceded": lam_a if home else lam_h,
                        "code": hash((team, player)) % 10_000,
                    })  # fmt: skip
        day += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_ratings_order_the_clubs_correctly():
    silver = _league(rounds=30)
    ratings = fit_ratings(team_matches(silver), silver["kickoff_time"].max() + pd.Timedelta(days=1))
    assert ratings.att["Alpha"] > ratings.att["Delta"]
    assert ratings.dfn["Alpha"] > ratings.dfn["Delta"]
    assert ratings.home > 0
    for_, against = ratings.expected_goals("Alpha", "Delta", home=True)
    assert for_ > against
    assert 0 < win_probability(for_, against) < 1
    assert win_probability(2.0, 0.5) > 0.5 > win_probability(0.5, 2.0)


def test_features_use_only_earlier_matches():
    silver = _league(rounds=8)
    out = attach_team_strength(silver)
    assert set(FEATURES) <= set(out.columns)
    # the first gameweek has nothing before it: league-average ratings, no NaN
    first = out[out["GW"] == 1]
    assert first["ts_att"].abs().max() == 0.0
    assert out[list(FEATURES)].notna().all().all()
    # rewriting the last gameweek's results must not move any earlier gameweek's features
    corrupted = silver.copy()
    last = corrupted["GW"] == corrupted["GW"].max()
    corrupted.loc[last, ["team_h_score", "team_a_score"]] = 9
    corrupted.loc[last, "expected_goals"] = 5.0
    again = attach_team_strength(corrupted)
    earlier = out["GW"] < out["GW"].max()
    pd.testing.assert_frame_equal(
        out.loc[earlier, list(FEATURES)].reset_index(drop=True),
        again.loc[earlier, list(FEATURES)].reset_index(drop=True),
    )


def test_fit_is_fast():
    silver = _league(rounds=40)
    matches = team_matches(silver)
    started = time.perf_counter()
    for _ in range(10):
        fit_ratings(matches, silver["kickoff_time"].max())
    assert time.perf_counter() - started < 2.0

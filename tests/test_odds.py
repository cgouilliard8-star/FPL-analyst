"""Bookmaker prices become per-side probabilities and Poisson-implied goals."""

import math

import pandas as pd
import pytest

from fpl.data.odds import ODDS_FEATURES, attach_odds, implied_goals, parse_football_data


def test_implied_goals_reproduce_the_market():
    from fpl.data.odds import _p_home_win, _poisson_cdf

    home, away = implied_goals(p_home=0.55, p_over25=0.60)
    assert home > away > 0
    assert 1.0 - _poisson_cdf(2, home + away) == pytest.approx(0.60, abs=1e-3)
    assert _p_home_win(home, away) == pytest.approx(0.55, abs=1e-3)


def test_parse_football_data_gives_each_side_its_own_view():
    raw = pd.DataFrame(
        [
            {"Div": "E0", "Date": "12/09/2026", "HomeTeam": "Man United", "AwayTeam": "Man City",
             "PSCH": 3.4, "PSCD": 3.6, "PSCA": 2.1, "Avg>2.5": 1.7, "Avg<2.5": 2.2},
            {"Div": "E0", "Date": "12/09/2026", "HomeTeam": "Chelsea", "AwayTeam": "Hull",
             "B365H": 1.3, "B365D": 5.5, "B365A": 9.0, "B365>2.5": 1.6, "B365<2.5": 2.4},
            {"Div": "E1", "Date": "12/09/2026", "HomeTeam": "Leeds", "AwayTeam": "Hull",
             "B365H": 2.0, "B365D": 3.0, "B365A": 4.0},
        ]
    )  # fmt: skip
    out = parse_football_data(raw, "2026-27")
    assert len(out) == 4  # two matches, two sides each; the E1 row is ignored
    united = out[(out["team"] == "Man Utd")].iloc[0]
    city = out[(out["team"] == "Man City")].iloc[0]
    assert united["home"] == "Man Utd" and united["away"] == "Man City"
    assert united["odds_win"] == pytest.approx(city["odds_lose"])
    assert united["odds_win"] < city["odds_win"]  # City are favourites
    assert united["odds_xg"] == pytest.approx(city["odds_xgc"])
    assert united["odds_cs"] == pytest.approx(math.exp(-united["odds_xgc"]))
    chelsea = out[out["team"] == "Chelsea"].iloc[0]
    assert chelsea["odds_win"] > 0.7 and chelsea["odds_xg"] > chelsea["odds_xgc"]


def test_attach_odds_joins_by_venue_and_leaves_unquoted_rows_nan():
    raw = pd.DataFrame([{"Div": "E0", "Date": "12/09/2026", "HomeTeam": "Man United", "AwayTeam": "Man City",
                         "B365H": 3.4, "B365D": 3.6, "B365A": 2.1}])  # fmt: skip
    odds = parse_football_data(raw, "2026-27")
    rows = pd.DataFrame(
        [
            {"season": "2026-27", "team": "Man City", "opponent": "Man Utd", "was_home": False},
            {"season": "2026-27", "team": "Man Utd", "opponent": "Man City", "was_home": True},
            {"season": "2026-27", "team": "Arsenal", "opponent": "Spurs", "was_home": True},
        ]
    )
    out = attach_odds(rows, odds)
    assert set(ODDS_FEATURES) <= set(out.columns)
    assert out.loc[0, "odds_win"] > out.loc[1, "odds_win"]
    assert pd.isna(out.loc[2, "odds_win"])


def test_odds_api_reply_is_parsed_into_football_data_columns():
    from fpl.data.odds import parse_odds_api

    payload = [
        {
            "home_team": "Manchester City",
            "away_team": "Tottenham Hotspur",
            "commence_time": "2026-09-12T14:00:00Z",
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Manchester City", "price": 1.4},
                                {"name": "Tottenham Hotspur", "price": 7.0},
                                {"name": "Draw", "price": 5.0},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 2.5, "price": 1.6},
                                {"name": "Under", "point": 2.5, "price": 2.3},
                            ],
                        },
                    ]
                }
            ],
        },
        {
            "home_team": "Nowhere FC",
            "away_team": "Arsenal",
            "commence_time": "2026-09-12T14:00:00Z",
            "bookmakers": [],
        },
    ]
    rows = parse_odds_api(payload)
    assert len(rows) == 1
    row = rows[0]
    assert (row["home"], row["away"]) == ("Man City", "Spurs")
    assert row["AvgH"] == 1.4 and row["Avg>2.5"] == 1.6 and row["Date"] == "12/09/2026"


def test_market_is_folded_into_the_ratings_only_where_priced():
    import numpy as np
    import pandas as pd

    from fpl.models.predict import _fold_market

    rows = pd.DataFrame(
        {
            "ts_xg_for": [1.0, 1.0],
            "odds_xg": [2.0, np.nan],
            "ts_cs": [0.3, 0.3],
            "odds_cs": [0.5, np.nan],
        }
    )
    out = _fold_market(rows, weight=0.5)
    assert out["ts_xg_for"].tolist() == [1.5, 1.0]
    assert out["ts_cs"].tolist() == [0.4, 0.3]

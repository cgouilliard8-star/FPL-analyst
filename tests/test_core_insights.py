"""Parsing the FPL-Core-Insights layout into player-gameweek actions."""

from __future__ import annotations

import pandas as pd

from fpl.data import core_insights as ci


def _write_season(root, season_dir):
    gw = root / "data" / season_dir / "By Gameweek" / "GW1"
    gw.mkdir(parents=True)
    pd.DataFrame(
        {
            "gameweek": [1, 1, 0],
            "match_id": ["m1", "cup", "friendly"],
            "tournament": ["prem", "efl-cup", "prem"],
            "home_team": [3, 3, 3],
            "away_team": [7, 8, 9],
        }
    ).to_csv(gw / "matches.csv", index=False)
    pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2],
            "match_id": ["m1", "cup", "m1", "m1"],  # the cup match and a duplicate row
            "minutes_played": [90, 45, 20, 20],
            "start_min": [0, 0, 70, 70],
            "total_shots": [3, 9, 1, 1],
            "tackles_won": [2, 0, 0, 0],
            "interceptions": [1, 0, 0, 0],
            "blocks": [0, 0, 1, 1],
            "clearances": [4, 0, 0, 0],
        }
    ).to_csv(gw / "playermatchstats.csv", index=False)
    pd.DataFrame({"player_id": [1, 2], "player_code": [1001, 1002]}).to_csv(
        root / "data" / season_dir / "players.csv", index=False
    )
    pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "average_entry_score": [54, 51, 51, 0],
            "highest_score": [140, 130, 130, 0],
            "finished": [True, True, True, False],
        }
    ).to_csv(root / "data" / season_dir / "gameweek_summaries.csv", index=False)


def test_league_matches_only_keyed_by_fpl_code(tmp_path, monkeypatch):
    _write_season(tmp_path, "2025-2026")
    monkeypatch.setattr(ci, "CORE_DIR", tmp_path / "out")
    written = ci.fetch_core_insights(("2025-26", "2022-23"), source=tmp_path)
    assert written == {"2025-26": 2}  # the cup match and the duplicate are gone
    stats = ci.load_core_stats(("2025-26",))
    starter = stats[stats["code"] == 1001].iloc[0]
    assert starter["GW"] == 1 and starter["ci_shots"] == 3 and starter["ci_def_actions"] == 7
    assert starter["ci_started"] == 1
    sub = stats[stats["code"] == 1002].iloc[0]
    assert sub["ci_started"] == 0 and sub["ci_def_actions"] == 1 and sub["ci_matches"] == 1
    averages = pd.read_csv(tmp_path / "out" / "averages.csv")
    assert averages["GW"].tolist() == [1, 2]  # finished gameweeks, once each
    assert averages["average"].tolist() == [54, 51]


def test_missing_season_is_empty_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "CORE_DIR", tmp_path / "none")
    assert ci.load_core_stats(("2022-23",)).empty

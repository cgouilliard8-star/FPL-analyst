"""Cup and European fixtures become congestion features without touching the past."""

import pandas as pd

from fpl.data import schedule as sch


def test_fixturedownload_rows_keep_only_premier_league_clubs():
    records = [
        {"HomeTeam": "Arsenal", "AwayTeam": "Bayern Munich", "DateUtc": "2026-09-16 19:00:00Z"},
        {"HomeTeam": "Real Madrid", "AwayTeam": "Liverpool", "DateUtc": "2026-09-17 19:00:00Z"},
        {"HomeTeam": "PSG", "AwayTeam": "Bayern Munich", "DateUtc": "2026-09-17 19:00:00Z"},
        {"HomeTeam": "Manchester City", "AwayTeam": "Inter", "DateUtc": "not a date"},
    ]
    frame = sch.parse_fixturedownload(records, "2026-27", "UCL")
    assert list(frame["home"]) == ["Arsenal", "Real Madrid"]
    assert list(frame["away"]) == ["Bayern Munich", "Liverpool"]
    assert frame["kickoff_time"].dt.tz is not None


def test_sportsdb_rows_parse_timestamps():
    payload = {
        "events": [
            {"strHomeTeam": "Tottenham Hotspur", "strAwayTeam": "Doncaster", "strTimestamp": "2026-09-23T19:45:00"},
            {"strHomeTeam": "Bayern", "strAwayTeam": "Inter", "strTimestamp": "2026-09-23T19:45:00"},
        ]
    }  # fmt: skip
    frame = sch.parse_sportsdb(payload, "2026-27", "EFL")
    assert list(frame["home"]) == ["Spurs"]


def test_congestion_counts_matches_around_each_kickoff():
    schedule = pd.DataFrame(
        {
            "season": ["2026-27"] * 3,
            "competition": ["UCL", "UCL", "EFL"],
            "kickoff_time": pd.to_datetime(
                ["2026-09-16 19:00Z", "2026-09-17 19:00Z", "2026-09-23 19:45Z"], utc=True
            ),
            "home": ["Arsenal", "Real Madrid", "Arsenal"],
            "away": ["Bayern Munich", "Liverpool", "Port Vale"],
        }
    )
    fixtures = pd.DataFrame(
        {
            "team": ["Arsenal", "Liverpool", "Chelsea", "Arsenal", "Arsenal"],
            "kickoff_time": pd.to_datetime(
                ["2026-09-19 14:00Z", "2026-09-20 15:30Z", "2026-09-19 14:00Z",
                 "2026-09-13 14:00Z", "2026-09-26 14:00Z"],
                utc=True,
            ),
        }
    )  # fmt: skip
    out = sch.congestion_features(fixtures, schedule)
    assert list(out["other_games_7d"]) == [1, 1, 0, 0, 1]
    assert list(out["euro_midweek"]) == [1, 1, 0, 0, 0]  # the cup tie is not European
    assert list(out["other_game_next_4d"]) == [0, 0, 0, 1, 0]
    assert out["days_since_any_match"].iloc[0] == pytest_approx(2.79)
    assert pd.isna(out["days_since_any_match"].iloc[2])


def pytest_approx(x):
    import pytest

    return pytest.approx(x, abs=0.01)


def test_empty_schedule_degrades_to_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(sch, "EXTERNAL", tmp_path)
    assert sch.load_schedule(("2026-27",)).empty
    fixtures = pd.DataFrame(
        {"team": ["Arsenal"], "kickoff_time": pd.to_datetime(["2026-09-19 14:00Z"], utc=True)}
    )
    out = sch.congestion_features(fixtures, sch.load_schedule(("2026-27",)))
    assert int(out["other_games_7d"].iloc[0]) == 0 and pd.isna(out["days_since_any_match"].iloc[0])


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(sch, "EXTERNAL", tmp_path)
    frame = pd.DataFrame(
        {
            "season": ["2026-27"], "competition": ["UEL"],
            "kickoff_time": pd.to_datetime(["2026-09-24 17:45Z"], utc=True),
            "home": ["Aston Villa"], "away": ["Porto"],
        }
    )  # fmt: skip
    sch.save_schedule(frame, "2026-27")
    back = sch.load_schedule(("2026-27",))
    assert back["kickoff_time"].iloc[0] == frame["kickoff_time"].iloc[0]
    assert back["home"].iloc[0] == "Aston Villa"

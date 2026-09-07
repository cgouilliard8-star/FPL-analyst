"""Match results, form and head-to-head lines come out of per-player fixture rows."""

import pandas as pd

from fpl.report.insights import fixture_insights, head_to_head, match_results, team_form


def rows():
    # Two seasons of one fixture pairing plus one other match, two players per side.
    data = []
    matches = [
        ("2024-25", 1, 1, "Spurs", "Chelsea", 1, 2, "2024-08-17T14:00:00Z"),
        ("2024-25", 20, 200, "Chelsea", "Spurs", 4, 3, "2025-01-05T16:30:00Z"),
        ("2025-26", 3, 30, "Spurs", "Chelsea", 0, 0, "2025-08-30T14:00:00Z"),
        ("2025-26", 4, 40, "Spurs", "Everton", 2, 0, "2025-09-13T14:00:00Z"),
    ]
    for season, gw, fixture, home, away, hg, ag, when in matches:
        for team, was_home in ((home, True), (away, False)):
            for i in range(2):
                data.append(
                    {
                        "season": season, "GW": gw, "fixture": fixture, "team": team,
                        "was_home": was_home, "team_h_score": hg, "team_a_score": ag,
                        "kickoff_time": when, "code": hash((team, i)) % 10000,
                    }
                )  # fmt: skip
    return pd.DataFrame(data)


def test_match_results_collapse_player_rows_to_one_per_match():
    results = match_results(rows())
    assert len(results) == 4
    first = results.iloc[0]
    assert (first["home"], first["away"], first["home_goals"], first["away_goals"]) == (
        "Spurs", "Chelsea", 1, 2,
    )  # fmt: skip


def test_head_to_head_is_from_the_home_sides_view():
    results = match_results(rows())
    h2h = head_to_head(results, "Spurs", "Chelsea")
    assert h2h["meetings"] == 3
    assert (h2h["home_wins"], h2h["draws"], h2h["away_wins"]) == (0, 1, 2)
    assert h2h["home_clean_sheets"] == 1 and h2h["both_scored"] == 2
    assert h2h["last"][-1]["score"] == "0-0"
    assert h2h["last"][1]["venue"] == "A"  # the Chelsea home game, seen from Spurs


def test_form_and_lines():
    results = match_results(rows())
    form = team_form(results, "Spurs", 3)
    assert form["form"] == "LDW" and form["clean_sheets"] == 2 and form["scored"] == 5
    out = fixture_insights(
        [{"gw": 5, "home": "Spurs", "away": "Chelsea", "kickoff": None}], results
    )
    assert out[0]["h2h"]["meetings"] == 3
    assert any("Last 3 meetings" in line for line in out[0]["lines"])
    assert any("Spurs form" in line for line in out[0]["lines"])
    assert head_to_head(results, "Spurs", "Arsenal") == {"meetings": 0}

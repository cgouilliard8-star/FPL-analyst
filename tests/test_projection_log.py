"""The live model is scored on projections written down before each deadline."""

import json

import pandas as pd
import pytest

from fpl.report import projection_log


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(projection_log, "PROJECTION_DIR", tmp_path)
    return tmp_path


def players(gameweek: int, n: int = 220) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "code": i + 1, "web_name": f"p{i}", "gameweek": gameweek,
                "position": ["GK", "DEF", "MID", "FWD"][i % 4],
                "team": f"club{i % 20}", "price": 4.0 + (i % 8),
                "ep1": (i % 11) / 2, "ep5": (i % 11) * 2.0,
                "availability": 1.0, "p_60": 0.9,
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


def test_saving_replaces_the_earlier_copy_of_the_same_gameweek(log_dir):
    projection_log.save_projections(players(4), captured_at="2026-09-07T06:00:00Z")
    projection_log.save_projections(players(4), captured_at="2026-09-08T06:00:00Z")
    files = sorted(p.name for p in log_dir.glob("*.csv"))
    assert files == ["2026-27_gw04.csv"]  # one file per gameweek, latest wins
    saved = pd.read_csv(log_dir / files[0])
    assert saved["captured_at"].iloc[0] == "2026-09-08T06:00:00Z"
    assert {"code", "ep1", "ep5", "availability", "p_60", "gameweek"} <= set(saved.columns)


def snapshot_with(gameweek: int, points: dict[int, int]) -> dict:
    history = {
        str(code): [
            {"round": gameweek, "total_points": pts, "minutes": 90 if pts else 0,
             "fixture": 1, "kickoff_time": "2026-09-12T14:00:00Z", "was_home": True,
             "opponent_team": 2, "value": 50, "selected": 100, "team_h_score": 1,
             "team_a_score": 0}
        ]
        for code, pts in points.items()
    }  # fmt: skip
    elements = [
        {"id": code, "code": code, "first_name": "a", "second_name": str(code),
         "element_type": 3, "team": 1, "web_name": str(code)}
        for code in points
    ]  # fmt: skip
    return {
        "events": [{"id": gameweek, "finished": True, "average_entry_score": 50, "highest_score": 120}],
        "elements": elements,
        "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"},
                  {"id": 2, "name": "Chelsea", "short_name": "CHE"}],
        "history": history,
        "fixtures": [],
        "captured_at": "2026-09-13T06:00:00Z",
    }  # fmt: skip


def test_scoring_needs_finished_results_and_reports_rank_correlation(log_dir, monkeypatch):
    frame = players(4)
    projection_log.save_projections(frame)
    # actual points that follow the projection exactly -> perfect rank correlation
    points = dict(zip(frame["code"], (frame["ep1"] * 2).astype(int), strict=True))
    monkeypatch.setattr(
        projection_log, "_actual_points",
        lambda s, season: pd.DataFrame(
            {"code": list(points), "gameweek": 4, "actual": list(points.values()), "minutes": 90}
        ),
    )  # fmt: skip
    out = projection_log.score_forward(snapshot_with(4, points))
    assert out["n"] == 1
    week = out["gameweeks"][0]
    assert week["gameweek"] == 4 and week["spearman"] == pytest.approx(1.0)
    assert week["top10_hits"] == 10 and week["average"] == 50
    assert out["mean_squad_points"] is not None and out["edge"] == pytest.approx(
        out["mean_squad_points"] - 50
    )


def test_nothing_saved_or_nothing_finished_scores_nothing(log_dir, monkeypatch):
    assert projection_log.score_forward(snapshot_with(4, {})) is None  # nothing logged
    monkeypatch.setattr(
        projection_log, "_actual_points",
        lambda s, season: pd.DataFrame({"code": [1], "gameweek": [9], "actual": [5], "minutes": [90]}),
    )  # fmt: skip
    projection_log.save_projections(players(9))
    unfinished = snapshot_with(9, {1: 5})
    unfinished["events"][0]["finished"] = False
    assert projection_log.score_forward(unfinished) is None  # results not in yet


def test_payload_carries_the_scorecard_shape():
    # the site reads meta/forward defensively; an empty log must serialise as null
    assert json.dumps({"forward": None}) == '{"forward": null}'

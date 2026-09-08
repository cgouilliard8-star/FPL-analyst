"""The live collector, exercised on a saved snapshot -- never the network."""

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl.data import fpl_api

FIXTURE = Path(__file__).parent / "fixtures" / "snapshot_small.json"


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return json.loads(FIXTURE.read_text())


def test_next_gameweek_is_the_one_marked_next(snapshot):
    gw = fpl_api.next_gameweek(snapshot)
    assert gw["is_next"] and not gw["finished"]
    assert gw["deadline_time"].endswith("Z")


def test_history_rows_match_archive_shape(snapshot):
    frame = fpl_api.snapshot_to_gameweeks(snapshot)
    for column in (
        "code", "element", "name", "position", "team", "GW", "fixture", "opponent_team",
        "was_home", "kickoff_time", "minutes", "total_points", "expected_goals", "value",
        "selected", "bps", "season", "defensive_contribution",
    ):  # fmt: skip
        assert column in frame.columns, column
    assert frame["season"].eq("2026-27").all()
    assert pd.api.types.is_datetime64_any_dtype(frame["kickoff_time"])
    assert pd.api.types.is_float_dtype(frame["expected_goals"])
    assert frame["GW"].max() <= fpl_api.next_gameweek(snapshot)["id"] - 1


def test_upcoming_rows_have_no_outcomes(snapshot):
    frame = fpl_api.snapshot_to_upcoming(snapshot)
    assert len(frame) > 0
    assert frame["GW"].eq(fpl_api.next_gameweek(snapshot)["id"]).all()
    for outcome in ("minutes", "total_points", "goals_scored"):
        assert outcome not in frame.columns


def test_upcoming_ownership_is_a_count_not_a_percentage(snapshot):
    """Feeding the percentage through as a count sank every live projection."""
    frame = fpl_api.snapshot_to_upcoming(snapshot)
    assert frame["selected"].median() > 1000
    assert frame["selected_by_percent"].max() <= 100


def test_upcoming_carries_availability_and_news(snapshot):
    frame = fpl_api.snapshot_to_upcoming(snapshot)
    flagged = frame[frame["availability"] < 1]
    assert len(flagged) > 0
    assert (flagged["news"].str.len() > 0).any()


@pytest.mark.parametrize(
    ("status", "chance", "expected"),
    [
        ("a", None, 1.0),
        ("d", None, 0.75),
        ("d", 25, 0.25),
        ("i", None, 0.0),
        ("i", 50, 0.5),
        ("s", None, 0.0),
        ("u", None, 0.0),
        ("n", None, 0.0),
    ],
)
def test_availability_multiplier(status, chance, expected):
    assert fpl_api.availability_multiplier(status, chance) == pytest.approx(expected)


def test_double_gameweek_yields_two_upcoming_rows(snapshot):
    """A club with two fixtures in the gameweek must produce two rows per player."""
    gw = fpl_api.next_gameweek(snapshot)["id"]
    fixtures = [f for f in snapshot["fixtures"] if f["event"] == gw]
    counts: dict[int, int] = {}
    for f in fixtures:
        counts[f["team_h"]] = counts.get(f["team_h"], 0) + 1
        counts[f["team_a"]] = counts.get(f["team_a"], 0) + 1
    frame = fpl_api.snapshot_to_upcoming(snapshot)
    per_player = frame.groupby("code").size()
    names = {t["id"]: t["name"] for t in snapshot["teams"]}
    for team_id, n in counts.items():
        codes = frame.loc[frame["team"] == names[team_id], "code"].unique()
        if len(codes):
            assert (per_player.loc[codes] == n).all()


def test_save_snapshot_never_overwrites(snapshot, tmp_path, monkeypatch):
    monkeypatch.setattr(fpl_api, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(fpl_api, "AVAILABILITY_LOG", tmp_path / "availability_log.csv")
    monkeypatch.setattr(fpl_api, "ensure_data_dirs", lambda: None)
    path = fpl_api.save_snapshot(snapshot)
    assert path.exists()
    with pytest.raises(FileExistsError):
        fpl_api.save_snapshot(snapshot)
    assert fpl_api.load_latest_snapshot()["captured_at"] == snapshot["captured_at"]
    # every snapshot appends one row per player to the availability log
    log = pd.read_csv(tmp_path / "availability_log.csv")
    assert len(log) == len(snapshot["elements"])
    assert {"captured_at", "gameweek", "code", "status", "chance", "news", "price"} <= set(log.columns)


def test_unknown_players_are_capped_at_fpls_view():
    import numpy as np
    import pandas as pd

    from fpl.models.combine import CONTRIBUTIONS
    from fpl.models.predict import UNKNOWN_CAP, _cap_unknowns

    rows = pd.DataFrame({"minutes_todate": [0.0, 2000.0, 0.0], "fpl_ep_next": [2.0, 2.0, 0.0]})
    parts = {term: [1.0, 1.0, 1.0] for term in CONTRIBUTIONS}
    breakdown = pd.DataFrame({**parts, "expected_points": [6.0, 6.0, 6.0], "p_60": [0.5] * 3})
    out = _cap_unknowns(breakdown, rows)
    assert out["expected_points"].iloc[0] == pytest.approx(UNKNOWN_CAP * 2.0)  # unknown: capped
    assert out["expected_points"].iloc[1] == 6.0  # a known player is left alone
    assert out["expected_points"].iloc[2] == 6.0  # no FPL figure: nothing to defer to
    assert np.isclose(out[list(CONTRIBUTIONS)].iloc[0].sum(), out["expected_points"].iloc[0])

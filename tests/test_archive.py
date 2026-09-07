"""The archive loader must fail loudly when upstream changes shape."""

import pandas as pd
import pytest

from fpl.data import archive


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, text: str):
        self._text = text

    def get(self, url, timeout=None):  # noqa: ARG002
        return _FakeResponse(self._text)


def _valid_csv() -> str:
    row = dict.fromkeys(archive.REQUIRED_COLUMNS, 0)
    row["name"] = "Test Player"
    row["position"] = "MID"
    row["team"] = "Arsenal"
    row["kickoff_time"] = "2024-08-16T19:00:00Z"
    row["was_home"] = True
    return pd.DataFrame([row]).to_csv(index=False)


def test_fetch_season_parses_valid_data():
    frame = archive.fetch_season("2024-25", session=_FakeSession(_valid_csv()))
    assert len(frame) == 1
    assert frame.loc[0, "season"] == "2024-25"
    assert pd.api.types.is_datetime64_any_dtype(frame["kickoff_time"])


def test_fetch_season_rejects_missing_columns():
    text = pd.DataFrame([{"name": "x", "GW": 1}]).to_csv(index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        archive.fetch_season("2024-25", session=_FakeSession(text))


def test_fetch_players_rejects_missing_columns():
    text = pd.DataFrame([{"id": 1}]).to_csv(index=False)
    with pytest.raises(ValueError, match="player registry is missing"):
        archive.fetch_players("2024-25", session=_FakeSession(text))


def test_required_columns_include_the_target():
    # If total_points ever drops out of the contract, there is nothing to learn.
    assert "total_points" in archive.REQUIRED_COLUMNS
    assert "minutes" in archive.REQUIRED_COLUMNS

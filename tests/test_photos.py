"""Portraits come from the newest season folder, and a new folder means a refetch."""

from __future__ import annotations

from fpl.data import photos


class _Resp:
    def __init__(self, ok):
        self.status_code = 200 if ok else 404
        self.content = b"png" if ok else b""


def _getter(available_prefix):
    def get(url, timeout=0, headers=None):
        return _Resp(available_prefix in url or "Photo-Missing" in url)  # placeholder too

    return get


def test_newest_available_folder_wins_and_a_change_refetches(tmp_path):
    snapshot = {
        "elements": [
            {"code": 1, "selected_by_percent": "40"},
            {"code": 2, "selected_by_percent": "1"},
        ]
    }
    counts = photos.fetch_photos(
        snapshot, directory=tmp_path, getter=_getter("premierleague25"), pause=0
    )
    assert counts == {"fetched": 3, "present": 0, "failed": 0}
    assert (
        (tmp_path / ".source")
        .read_text()
        .startswith("https://resources.premierleague.com/premierleague25")
    )
    # nothing new: everything is present
    counts = photos.fetch_photos(
        snapshot, directory=tmp_path, getter=_getter("premierleague25"), pause=0
    )
    assert counts["present"] == 3 and counts["fetched"] == 0
    # the next season's folder appears: the old portraits are replaced
    counts = photos.fetch_photos(
        snapshot, directory=tmp_path, getter=_getter("premierleague26"), pause=0
    )
    assert counts["fetched"] == 3 and (tmp_path / ".source").read_text().startswith(
        "https://resources.premierleague.com/premierleague26"
    )

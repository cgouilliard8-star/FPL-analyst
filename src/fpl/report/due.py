"""Decide whether a scheduled refresh is worth running right now.

The workflow wakes every few hours. Most of those wakes should do nothing: a full
refresh downloads every player's history, refits nine models and rebuilds the site,
which is pointless on a Tuesday afternoon between gameweeks. What matters is the
final day and a half before a deadline, when press conferences and injury news
land, and a daily heartbeat so the availability log keeps its record and a stalled
site is noticed. This is one cheap request to the bootstrap endpoint and a look at
the last published payload.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from fpl.config import PROJECT_ROOT

log = logging.getLogger(__name__)

BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
LIVE_PATH = PROJECT_ROOT / "site" / "data" / "live.json"
DEADLINE_WINDOW_HOURS = 36.0  # refresh at every wake this close to a deadline
HEARTBEAT_HOURS = 20.0  # ...and at least once a day regardless


@dataclass
class Verdict:
    due: bool
    reason: str
    hours_to_deadline: float | None
    hours_since_refresh: float | None

    def as_dict(self) -> dict:
        return self.__dict__


def next_deadline(session: requests.Session | None = None) -> datetime | None:
    getter = session.get if session else requests.get
    response = getter(BOOTSTRAP, timeout=30, headers={"User-Agent": "fpl-analyst/0.1"})
    response.raise_for_status()
    events = response.json().get("events", [])
    upcoming = [e for e in events if e.get("is_next")] or [
        e for e in events if not e.get("finished")
    ]
    if not upcoming:
        return None
    return datetime.fromisoformat(upcoming[0]["deadline_time"].replace("Z", "+00:00"))


def last_refresh(path: Path = LIVE_PATH) -> datetime | None:
    if not path.exists():
        return None
    try:
        stamp = json.loads(path.read_text())["meta"]["generated_at"]
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def decide(
    *,
    now: datetime | None = None,
    deadline: datetime | None = None,
    refreshed: datetime | None = None,
    force: bool = False,
    window_hours: float = DEADLINE_WINDOW_HOURS,
    heartbeat_hours: float = HEARTBEAT_HOURS,
) -> Verdict:
    now = now or datetime.now(timezone.utc)
    to_deadline = (deadline - now).total_seconds() / 3600.0 if deadline else None
    since = (now - refreshed).total_seconds() / 3600.0 if refreshed else None
    if force:
        return Verdict(True, "run requested by hand", to_deadline, since)
    if since is None:
        return Verdict(True, "no published data yet", to_deadline, since)
    if to_deadline is not None and 0.0 <= to_deadline <= window_hours:
        return Verdict(True, f"deadline in {to_deadline:.1f}h", to_deadline, since)
    if since >= heartbeat_hours:
        return Verdict(True, f"last refresh {since:.1f}h ago", to_deadline, since)
    return Verdict(
        False, f"deadline in {to_deadline:.1f}h, refreshed {since:.1f}h ago", to_deadline, since
    )


def main(force: bool = False) -> Verdict:
    try:
        deadline = next_deadline()
    except Exception as e:  # noqa: BLE001 - if FPL is unreachable, let the run decide
        log.warning("could not read the next deadline (%s); treating the refresh as due", e)
        deadline = None
        verdict = Verdict(True, "deadline unknown", None, None)
    else:
        verdict = decide(deadline=deadline, refreshed=last_refresh(), force=force)
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"due={'true' if verdict.due else 'false'}\n")
    return verdict


__all__ = ["Verdict", "decide", "last_refresh", "main", "next_deadline"]

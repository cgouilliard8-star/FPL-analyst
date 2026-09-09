"""When a scheduled refresh should actually build."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fpl.report.due import decide

NOW = datetime(2026, 9, 10, 6, 0, tzinfo=timezone.utc)


def test_near_a_deadline_every_wake_builds():
    v = decide(now=NOW, deadline=NOW + timedelta(hours=30), refreshed=NOW - timedelta(hours=6))
    assert v.due and "deadline" in v.reason


def test_between_gameweeks_only_the_daily_heartbeat_builds():
    quiet = decide(now=NOW, deadline=NOW + timedelta(hours=80), refreshed=NOW - timedelta(hours=6))
    assert not quiet.due
    stale = decide(now=NOW, deadline=NOW + timedelta(hours=80), refreshed=NOW - timedelta(hours=25))
    assert stale.due and "last refresh" in stale.reason


def test_manual_runs_and_first_runs_always_build():
    assert decide(now=NOW, deadline=NOW + timedelta(hours=80), refreshed=NOW, force=True).due
    assert decide(now=NOW, deadline=NOW + timedelta(hours=80), refreshed=None).due


def test_a_deadline_that_has_passed_does_not_count_as_near():
    v = decide(now=NOW, deadline=NOW - timedelta(hours=2), refreshed=NOW - timedelta(hours=3))
    assert not v.due

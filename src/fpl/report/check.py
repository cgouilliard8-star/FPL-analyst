"""Sanity checks on a built ``live.json`` before it is allowed to reach the site.

A refresh that "succeeds" with broken data is worse than one that fails: the page
would rate every team against nonsense until the next morning. So the deploy is gated
on these checks, and on failure the previously published file stays in place.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fpl.config import SQUAD_QUOTA

MIN_PLAYERS = 450
MIN_CLUBS = 20
OPTIMAL_RANGE = (35.0, 130.0)  # next-gameweek points of the best £100m squad
MAX_AGE_HOURS = 36
SCHEMA = 2


def _walk(obj, path="root"):
    """Yield the path of any NaN/inf, which JSON would have turned into garbage."""
    if isinstance(obj, float) and not math.isfinite(obj):
        yield path
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:2000]):
            yield from _walk(v, f"{path}[{i}]")


def check_payload(payload: dict, *, now: datetime | None = None) -> list[str]:
    """Return a list of problems; an empty list means the payload is fit to publish."""
    now = now or datetime.now(timezone.utc)
    problems: list[str] = []
    meta = payload.get("meta", {})
    players = payload.get("players", [])

    if meta.get("schema", SCHEMA) != SCHEMA:
        problems.append(f"schema {meta.get('schema')} != {SCHEMA}")
    if len(players) < MIN_PLAYERS:
        problems.append(f"only {len(players)} players (need {MIN_PLAYERS})")
    clubs = {p.get("team") for p in players}
    if len(clubs) < MIN_CLUBS:
        problems.append(f"only {len(clubs)} clubs")
    if not 1 <= int(meta.get("gameweek", 0)) <= 38:
        problems.append(f"gameweek {meta.get('gameweek')} out of range")

    opt = float(meta.get("optimal_points") or 0)
    if not OPTIMAL_RANGE[0] <= opt <= OPTIMAL_RANGE[1]:
        problems.append(f"optimal_points {opt} outside {OPTIMAL_RANGE}")

    for key in ("ep", "ep1", "ep3", "ep5", "price", "availability"):
        bad = [p["web_name"] for p in players if not isinstance(p.get(key), (int, float))]
        if bad:
            problems.append(f"{len(bad)} players missing {key} (e.g. {bad[:3]})")
    if any(p.get("price", 0) <= 0 for p in players):
        problems.append("a player has a non-positive price")
    if all(p.get("ep", 0) == 0 for p in players):
        problems.append("every projection is zero")

    # Each position must be fillable under the quota, or no squad can be built.
    counts: dict[str, int] = {}
    for p in players:
        counts[p.get("position")] = counts.get(p.get("position"), 0) + 1
    for pos, quota in SQUAD_QUOTA.items():
        if counts.get(pos, 0) < quota * 3:
            problems.append(f"only {counts.get(pos, 0)} {pos}s")

    codes = [p.get("code") for p in players]
    if len(set(codes)) != len(codes):
        problems.append("duplicate player codes")

    optimal = payload.get("optimal", {})
    if len(optimal.get("starters", [])) != 11 or len(optimal.get("bench", [])) != 4:
        problems.append("optimal squad is not 11 + 4")

    captured = meta.get("captured_at")
    try:
        age = now - datetime.fromisoformat(str(captured).replace("Z", "+00:00"))
        if age > timedelta(hours=MAX_AGE_HOURS):
            problems.append(f"snapshot is {age.total_seconds() / 3600:.0f}h old")
    except (TypeError, ValueError):
        problems.append(f"unreadable captured_at {captured!r}")

    nans = list(_walk(payload))
    if nans:
        problems.append(f"non-finite numbers at {nans[:3]}")
    return problems


def check_file(path: Path) -> list[str]:
    if not path.exists():
        return [f"{path} does not exist"]
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]
    return check_payload(payload)

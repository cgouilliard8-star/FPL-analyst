"""The deploy gate must pass a healthy payload and name what is wrong with a bad one."""

from datetime import datetime, timedelta, timezone

from fpl.report.check import check_payload


def good_payload(now):
    players = []
    code = 0
    for club in range(20):
        for position, n in (("GK", 3), ("DEF", 8), ("MID", 8), ("FWD", 5)):
            for _ in range(n):
                code += 1
                players.append(
                    {
                        "code": code, "web_name": f"p{code}", "position": position,
                        "team": f"club{club}", "price": 4.5, "ep": 2.0, "ep1": 2.0,
                        "ep3": 5.0, "ep5": 7.0, "availability": 1.0,
                    }
                )  # fmt: skip
    return {
        "meta": {
            "schema": 2,
            "gameweek": 4,
            "optimal_points": 70.0,
            "captured_at": (now - timedelta(hours=2)).isoformat(),
        },  # fmt: skip
        "optimal": {"starters": list(range(1, 12)), "bench": [12, 13, 14, 15]},
        "players": players,
    }


def test_healthy_payload_passes():
    now = datetime.now(timezone.utc)
    assert check_payload(good_payload(now), now=now) == []


def test_problems_are_named():
    now = datetime.now(timezone.utc)
    bad = good_payload(now)
    bad["players"] = bad["players"][:100]
    bad["meta"]["optimal_points"] = 3.0
    bad["meta"]["captured_at"] = (now - timedelta(days=3)).isoformat()
    bad["players"][0]["ep5"] = float("nan")
    problems = check_payload(bad, now=now)
    text = " | ".join(problems)
    assert "players" in text and "optimal_points" in text and "old" in text
    assert "non-finite" in text


def test_missing_projection_field_is_caught():
    now = datetime.now(timezone.utc)
    bad = good_payload(now)
    del bad["players"][5]["ep3"]
    assert any("ep3" in p for p in check_payload(bad, now=now))

"""Match-level context for the page: results, form, head-to-head history, and the
sentences the fixtures section shows ("Spurs have lost the last four against Chelsea").

Everything here is derived from the per-player fixture rows already in the archive
and the live snapshot: two players from opposite sides of the same fixture id carry
the same scoreline, so a match table falls out of a group-by. Nothing is fetched.
"""

from __future__ import annotations

import pandas as pd

RESULT_COLUMNS = ["season", "GW", "kickoff_time", "home", "away", "home_goals", "away_goals"]


def match_results(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per finished match from per-player fixture rows.

    Requires ``season, GW, fixture, team, was_home, team_h_score, team_a_score,
    kickoff_time``; rows without a scoreline (unplayed) are dropped.
    """
    needed = {"season", "GW", "fixture", "team", "was_home", "team_h_score", "team_a_score"}
    if rows.empty or not needed <= set(rows.columns):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    played = rows.dropna(subset=["team_h_score", "team_a_score"])
    if played.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    home = (
        played[played["was_home"].astype(bool)]
        .groupby(["season", "fixture"], as_index=False)
        .agg(home=("team", "first"), kickoff_time=("kickoff_time", "min"), GW=("GW", "first"),
             home_goals=("team_h_score", "first"), away_goals=("team_a_score", "first"))
    )  # fmt: skip
    away = (
        played[~played["was_home"].astype(bool)]
        .groupby(["season", "fixture"], as_index=False)
        .agg(away=("team", "first"))
    )
    out = home.merge(away, on=["season", "fixture"], how="inner")
    out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], utc=True, errors="coerce")
    out = out.dropna(subset=["home", "away"])
    out[["home_goals", "away_goals"]] = out[["home_goals", "away_goals"]].astype(int)
    return out[RESULT_COLUMNS].sort_values("kickoff_time").reset_index(drop=True)


def _from_side(results: pd.DataFrame, team: str) -> pd.DataFrame:
    """Results as seen by ``team``: goals for/against, venue, outcome."""
    h = results[results["home"] == team].assign(
        opponent=lambda d: d["away"], venue="H",
        gf=lambda d: d["home_goals"], ga=lambda d: d["away_goals"],
    )  # fmt: skip
    a = results[results["away"] == team].assign(
        opponent=lambda d: d["home"], venue="A",
        gf=lambda d: d["away_goals"], ga=lambda d: d["home_goals"],
    )  # fmt: skip
    both = pd.concat([h, a]).sort_values("kickoff_time")
    both["result"] = [
        "W" if g > c else "D" if g == c else "L"
        for g, c in zip(both["gf"], both["ga"], strict=True)
    ]
    return both


def team_form(results: pd.DataFrame, team: str, n: int = 5) -> dict:
    last = _from_side(results, team).tail(n)
    if last.empty:
        return {"games": 0}
    return {
        "games": int(len(last)),
        "form": "".join(last["result"]),
        "points": int((last["result"] == "W").sum() * 3 + (last["result"] == "D").sum()),
        "scored": int(last["gf"].sum()),
        "conceded": int(last["ga"].sum()),
        "clean_sheets": int((last["ga"] == 0).sum()),
        "blanks": int((last["gf"] == 0).sum()),
    }


def head_to_head(results: pd.DataFrame, home: str, away: str, n: int = 6) -> dict:
    """The last ``n`` meetings, from the home side's point of view."""
    mask = ((results["home"] == home) & (results["away"] == away)) | (
        (results["home"] == away) & (results["away"] == home)
    )
    last = results[mask].sort_values("kickoff_time").tail(n)
    if last.empty:
        return {"meetings": 0}
    home_goals = [
        hg if h == home else ag
        for h, hg, ag in zip(last["home"], last["home_goals"], last["away_goals"], strict=True)
    ]
    away_goals = [
        ag if h == home else hg
        for h, hg, ag in zip(last["home"], last["home_goals"], last["away_goals"], strict=True)
    ]
    wins = sum(1 for x, y in zip(home_goals, away_goals, strict=True) if x > y)
    draws = sum(1 for x, y in zip(home_goals, away_goals, strict=True) if x == y)
    return {
        "meetings": int(len(last)),
        "home_wins": wins,
        "draws": draws,
        "away_wins": int(len(last) - wins - draws),
        "avg_goals": round((sum(home_goals) + sum(away_goals)) / len(last), 1),
        "home_clean_sheets": sum(1 for y in away_goals if y == 0),
        "away_clean_sheets": sum(1 for x in home_goals if x == 0),
        "both_scored": sum(
            1 for x, y in zip(home_goals, away_goals, strict=True) if x > 0 and y > 0
        ),
        "last": [
            {
                "season": s,
                "gw": int(gw),
                "venue": "H" if h == home else "A",
                "score": f"{x}-{y}",
            }
            for s, gw, h, x, y in zip(
                last["season"], last["GW"], last["home"], home_goals, away_goals, strict=True
            )
        ],  # fmt: skip
    }


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def fixture_lines(home: str, away: str, h2h: dict, home_form: dict, away_form: dict) -> list[str]:
    """Short, factual sentences for one fixture."""
    lines: list[str] = []
    if h2h.get("meetings"):
        m = h2h["meetings"]
        lines.append(
            f"Last {m} meetings: {home} {h2h['home_wins']}, draws {h2h['draws']}, "
            f"{away} {h2h['away_wins']}; {h2h['avg_goals']} goals a game on average."
        )
        if h2h["home_clean_sheets"] == 0 and h2h["away_clean_sheets"] == 0:
            lines.append(
                f"Neither side has kept a clean sheet in those {m} — both teams scored in {h2h['both_scored']}."
            )
        else:
            lines.append(
                f"Clean sheets in those {m}: {home} {h2h['home_clean_sheets']}, {away} {h2h['away_clean_sheets']}; "
                f"both scored in {h2h['both_scored']}."
            )
        scores = ", ".join(
            f"{x['score']} ({x['season'][2:]} {x['venue']})" for x in h2h["last"][-4:]
        )
        lines.append(f"Recent scorelines from {home}'s side: {scores}.")
    for team, form in ((home, home_form), (away, away_form)):
        if form.get("games"):
            g = form["games"]
            lines.append(
                f"{team} form {form['form']} — scored {form['scored']} and conceded {form['conceded']} in the last {g}, "
                f"{_plural(form['clean_sheets'], 'clean sheet')}, {_plural(form['blanks'], 'blank')}."
            )
    return lines


def fixture_insights(
    fixtures: list[dict], results: pd.DataFrame, *, meetings: int = 6, form_games: int = 5
) -> list[dict]:
    """For each ``{gw, home, away, kickoff}`` fixture, its history and the lines."""
    out = []
    for f in fixtures:
        h2h = head_to_head(results, f["home"], f["away"], meetings)
        hf = team_form(results, f["home"], form_games)
        af = team_form(results, f["away"], form_games)
        out.append(
            {
                **f,
                "h2h": h2h,
                "home_form": hf,
                "away_form": af,
                "lines": fixture_lines(f["home"], f["away"], h2h, hf, af),
            }
        )
    return out

"""Keep every gameweek's projections as they stood at the deadline, then score them
against what actually happened.

The walk-forward backtest is honest but handicapped: the archive has no injury flags,
so a replay of a past season has to treat everyone as fit. The live model does not --
it reads FPL's flags and news every morning. The only way to measure *that* model is
to write its projections down before each deadline and check them afterwards, which is
what this does.

One CSV per gameweek in ``data/external/projections/``, rewritten by every refresh
while that gameweek is still the next one, so the file that survives is the last state
before the deadline. Committed by the refresh, so the record accumulates on its own
and cannot be revised after the fact.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fpl.config import CURRENT_SEASON, PROJECT_ROOT
from fpl.optimise.squad import pick_squad

log = logging.getLogger(__name__)

PROJECTION_DIR = PROJECT_ROOT / "data" / "external" / "projections"
COLUMNS = (
    "code", "web_name", "position", "team", "price", "ep1", "ep5", "availability", "p_60",
    # the blend's two ingredients, kept apart so the scorecard can judge each
    "ep1_model", "fpl_ep", "p_haul",
)  # fmt: skip
SOURCES = {"ep1": "blend", "ep1_model": "model", "fpl_ep": "fpl"}


def projection_path(season: str, gameweek: int) -> Path:
    return PROJECTION_DIR / f"{season}_gw{gameweek:02d}.csv"


def save_projections(
    players: pd.DataFrame, *, season: str = CURRENT_SEASON, captured_at: str = ""
) -> Path:
    """Write this gameweek's projections, replacing any earlier copy for it."""
    gameweek = int(players["gameweek"].iloc[0])
    PROJECTION_DIR.mkdir(parents=True, exist_ok=True)
    frame = players[[c for c in COLUMNS if c in players.columns]].copy()
    frame.insert(0, "gameweek", gameweek)
    frame.insert(0, "season", season)
    frame.insert(0, "captured_at", captured_at or "")
    for column in ("ep1", "ep5", "availability", "p_60", "price", "ep1_model", "fpl_ep", "p_haul"):
        if column in frame.columns:
            frame[column] = frame[column].astype(float).round(3)
    path = projection_path(season, gameweek)
    frame.to_csv(path, index=False)
    log.info("projection log: %d players -> %s", len(frame), path.name)
    return path


def load_projections(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Every gameweek's saved projections for the season, concatenated."""
    if not PROJECTION_DIR.exists():
        return pd.DataFrame(columns=["season", "gameweek", *COLUMNS])
    parts = [pd.read_csv(p) for p in sorted(PROJECTION_DIR.glob(f"{season}_gw*.csv"))]
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["season", "gameweek", *COLUMNS])
    )


def _actual_points(snapshot: dict, season: str) -> pd.DataFrame:
    from fpl.data.fpl_api import snapshot_to_gameweeks

    rows = snapshot_to_gameweeks(snapshot, season)
    if rows.empty:
        return pd.DataFrame(columns=["code", "gameweek", "actual", "minutes"])
    grouped = rows.groupby(["code", "GW"], as_index=False)[["total_points", "minutes"]].sum()
    return grouped.rename(columns={"GW": "gameweek", "total_points": "actual"})


def _best_squad_points(rows: pd.DataFrame, column: str = "ep1") -> float | None:
    """What a fresh £100m squad picked on these projections actually scored."""
    pool = rows.rename(columns={column: "projected"}).assign(
        price_tenths=lambda d: (d["price"] * 10).round().astype(int),
        actual=lambda d: d["actual"],
    )
    needed = ["code", "position", "team", "price_tenths", "projected", "actual"]
    if pool[needed].isna().any().any() or len(pool) < 200:
        return None
    try:
        squad = pick_squad(pool[needed])
    except Exception:  # noqa: BLE001 - a scorecard must never break the refresh
        return None
    starters = squad[squad["is_starter"]]
    captain = squad[squad["is_captain"]]
    return float(starters["actual"].sum() + captain["actual"].sum())


def score_forward(snapshot: dict, season: str = CURRENT_SEASON) -> dict | None:
    """Score every saved gameweek whose results are in, one row per gameweek.

    ``squad_points`` is what a fresh optimal £100m squad picked on that gameweek's
    saved projections went on to score -- the number a manager would recognise, next
    to the average manager's for the same week.
    """
    saved = load_projections(season)
    if saved.empty:
        return None
    actual = _actual_points(snapshot, season)
    if actual.empty:
        return None
    averages = {int(e["id"]): e for e in snapshot["events"] if e.get("finished")}
    merged = saved.merge(actual, on=["code", "gameweek"], how="inner")

    weeks = []
    for gameweek, rows in merged.groupby("gameweek"):
        gameweek = int(gameweek)
        if gameweek not in averages or len(rows) < 100:
            continue
        played = rows[rows["minutes"] > 0]
        top10 = rows.nlargest(10, "ep1")
        week = {
            "gameweek": gameweek,
            "players": int(len(rows)),
            "spearman": round(float(rows["ep1"].corr(rows["actual"], method="spearman")), 3),
            "spearman_played": (
                round(float(played["ep1"].corr(played["actual"], method="spearman")), 3)
                if len(played) > 30
                else None
            ),
            "top10_points": int(top10["actual"].sum()),
            "top10_hits": int((top10["actual"] >= 5).sum()),
            "squad_points": _best_squad_points(rows),
            "average": averages[gameweek].get("average_entry_score"),
            "highest": averages[gameweek].get("highest_score"),
        }
        # The blend's ingredients scored on their own, where the log has them.
        for column, label in SOURCES.items():
            if column == "ep1" or column not in rows.columns or rows[column].fillna(0).eq(0).all():
                continue
            week[f"spearman_{label}"] = round(
                float(rows[column].corr(rows["actual"], method="spearman")), 3
            )
            week[f"squad_points_{label}"] = _best_squad_points(rows, column)
        weeks.append(week)
    if not weeks:
        return None
    scored = [w for w in weeks if w["squad_points"] is not None and w["average"]]
    summary = {
        "gameweeks": weeks,
        "n": len(weeks),
        "mean_spearman": round(sum(w["spearman"] for w in weeks) / len(weeks), 3),
        "mean_top10_hits": round(sum(w["top10_hits"] for w in weeks) / len(weeks), 1),
    }
    if scored:
        summary["mean_squad_points"] = round(
            sum(w["squad_points"] for w in scored) / len(scored), 1
        )
        summary["mean_average"] = round(sum(w["average"] for w in scored) / len(scored), 1)
        summary["edge"] = round(summary["mean_squad_points"] - summary["mean_average"], 1)
    for label in ("model", "fpl"):
        rows = [w for w in weeks if w.get(f"squad_points_{label}") is not None]
        if rows:
            summary[f"mean_squad_points_{label}"] = round(
                sum(w[f"squad_points_{label}"] for w in rows) / len(rows), 1
            )
            summary[f"mean_spearman_{label}"] = round(
                sum(w[f"spearman_{label}"] for w in rows) / len(rows), 3
            )
    log.info("forward scorecard: %d gameweeks scored", len(weeks))
    return summary

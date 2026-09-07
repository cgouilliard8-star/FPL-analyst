"""Produce the dashboard payload for one gameweek.

Writes a single JSON file per gameweek that the static site reads. No build step, no
server: the pipeline emits data, the page renders it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fpl.config import PROJECT_ROOT
from fpl.features.build import feature_columns, load_features
from fpl.models.combine import fit_component_model
from fpl.optimise.squad import pick_squad, squad_points

log = logging.getLogger(__name__)

SITE_DATA = PROJECT_ROOT / "site" / "data"
TOP_N = 40


def _pool(frame: pd.DataFrame, projections: pd.DataFrame) -> pd.DataFrame:
    pool = projections.rename(columns={"expected_points": "projected"}).copy()
    pool["price_tenths"] = frame["value"].to_numpy()
    pool["team"] = frame["team"].to_numpy()
    pool["position"] = frame["position"].to_numpy()
    pool["code"] = frame["code"].to_numpy()
    pool["actual"] = frame["total_points"].to_numpy()
    return pool


def build_gameweek(
    gameweek: int,
    *,
    season: str = "2024-25",
    frame: pd.DataFrame | None = None,
    explain: bool = True,
) -> dict:
    """Fit on everything before the gameweek, project it, pick a squad, explain it."""
    frame = load_features() if frame is None else frame
    features = feature_columns(frame)

    target = frame[(frame["season"] == season) & (frame["GW"] == gameweek)]
    if target.empty:
        raise ValueError(f"no rows for {season} GW{gameweek}")

    cutoff = target["kickoff_time"].min()
    train = frame[frame["kickoff_time"] < cutoff]
    log.info("GW%s: training on %d rows", gameweek, len(train))

    model = fit_component_model(train, features)
    breakdown = model.explain(target)

    # The feature table and the decomposition both carry a `bonus` column: one is the
    # bonus points actually awarded, the other the model's expected contribution.
    # Concatenating blindly leaves two columns of the same name, and every later
    # lookup silently returns a two-element Series instead of a value.
    target_side = target.reset_index(drop=True)
    breakdown_side = breakdown.reset_index(drop=True)
    clashing = sorted(set(target_side.columns) & set(breakdown_side.columns))
    enriched = pd.concat([target_side.drop(columns=clashing), breakdown_side], axis=1)
    if enriched.columns.duplicated().any():
        duplicated = enriched.columns[enriched.columns.duplicated()].tolist()
        raise ValueError(f"duplicate columns after join: {duplicated}")

    pool = _pool(target_side, breakdown_side)

    squad = pick_squad(pool)
    starters = squad[squad["is_starter"]]
    captain_code = int(squad.loc[squad["is_captain"], "code"].iloc[0])

    by_code = enriched.set_index("code")
    if explain:
        from fpl.explain.generate import explain_frame

        wanted = list(starters["code"])
        explained = explain_frame(by_code.loc[wanted])
        rationales = dict(zip(explained.index, explained["rationale"], strict=True))
        sources = dict(zip(explained.index, explained["rationale_source"], strict=True))
    else:
        rationales, sources = {}, {}

    def player_record(code: int) -> dict:
        row = by_code.loc[code]
        return {
            "code": int(code),
            "name": row["full_name"],
            "position": row["position"],
            "team": row["team"],
            "opponent": row["opponent"],
            "home": bool(row["is_home"]),
            "price": round(float(row["price"]), 1),
            "projected": round(float(row["expected_points"]), 2),
            "actual": float(row["total_points"]),
            "breakdown": {
                "appearance": round(float(row["appearance"]), 2),
                "attacking": round(float(row["attacking"]), 2),
                "clean_sheet": round(float(row["clean_sheet"]), 2),
                "bonus": round(float(row["bonus"]), 2),
                "goalkeeping": round(float(row["goalkeeping"]), 2),
            },
            "start_probability": round(float(row["p_60"]), 2),
            "is_captain": int(code) == captain_code,
            "rationale": rationales.get(code),
            "rationale_source": sources.get(code),
        }

    top = enriched.nlargest(TOP_N, "expected_points")

    return {
        "season": season,
        "gameweek": int(gameweek),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "component",
        "squad": {
            "starters": [player_record(c) for c in starters["code"]],
            "bench": [player_record(c) for c in squad.loc[~squad["is_starter"], "code"]],
            "captain": captain_code,
            "spend": round(int(squad["price_tenths"].sum()) / 10, 1),
            "projected": round(
                float(
                    starters["projected"].sum() + squad.loc[squad["is_captain"], "projected"].sum()
                ),
                2,
            ),
            "actual": round(squad_points(squad), 1),
        },
        "top_projections": [player_record(c) for c in top["code"]],
    }


def write_gameweek(payload: dict, directory: Path = SITE_DATA) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"gw{payload['gameweek']}.json"
    path.write_text(json.dumps(payload, indent=2))

    index_path = directory / "index.json"
    if index_path.exists():
        existing = json.loads(index_path.read_text())
        if existing.get("season") not in (None, payload["season"]):
            raise ValueError(
                f"{directory} already holds {existing['season']}; refusing to mix in "
                f"{payload['season']}. Use a separate directory per season."
            )

    index = sorted(int(p.stem[2:]) for p in directory.glob("gw*.json") if p.stem[2:].isdigit())
    index_path.write_text(
        json.dumps(
            {"season": payload["season"], "gameweeks": index, "latest": max(index)}, indent=2
        )
    )
    log.info("wrote %s", path)
    return path

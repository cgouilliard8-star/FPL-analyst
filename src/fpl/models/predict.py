"""Project the next gameweek from a live snapshot.

The archive supplies the seasons that are over; the snapshot supplies this season's
finished fixtures and the fixtures about to be played. All three are stacked into one
table, features are built exactly as in the backtest, the component model is fitted
on every row whose outcome is known, and the rows whose outcome is not yet known are
what it predicts.

The availability flag is applied *after* the model, as a multiplier. The model
learns what a fit player does; FPL's flag says whether he will be on the pitch, and
it knows things (press conferences, scans) that no rolling window can.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fpl.config import CURRENT_SEASON, TRAIN_SEASONS
from fpl.data.archive import load_players
from fpl.data.fpl_api import (
    next_gameweek,
    snapshot_to_gameweeks,
    snapshot_to_upcoming,
)
from fpl.data.silver import load_silver
from fpl.entity.resolve import canonical_team, load_team_aliases
from fpl.features.build import build_features, feature_columns
from fpl.models.combine import CONTRIBUTIONS, fit_component_model

log = logging.getLogger(__name__)


def _snapshot_registry(snapshot: dict, season: str) -> pd.DataFrame:
    """The live season's players in the archive registry's shape."""
    return pd.DataFrame(
        [
            {
                "season": season,
                "id": e["id"],
                "code": e["code"],
                "first_name": e["first_name"],
                "second_name": e["second_name"],
                "element_type": e["element_type"],
                "team": e["team"],
            }
            for e in snapshot["elements"]
        ]
    )


def _canonicalise(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = load_team_aliases()
    frame = frame.copy()
    frame["team"] = frame["team"].map(lambda n: canonical_team(n, aliases))
    unknown = frame["team"].isna()
    if unknown.any():
        raise ValueError(
            f"unrecognised clubs in live data: {sorted(frame.loc[unknown, 'team'].unique())}"
        )
    frame["full_name"] = frame["name"]
    return frame[frame["position"].isin(["GK", "DEF", "MID", "FWD"])]


def assemble(snapshot: dict, season: str = CURRENT_SEASON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack archive seasons, live finished fixtures and upcoming fixtures.

    Returns the combined per-fixture table and the combined player registry.
    """
    archive = load_silver()
    history = _canonicalise(snapshot_to_gameweeks(snapshot, season))
    upcoming = _canonicalise(snapshot_to_upcoming(snapshot, season))
    history["is_upcoming"] = 0
    upcoming["is_upcoming"] = 1
    archive["is_upcoming"] = 0

    combined = pd.concat([archive, history, upcoming], ignore_index=True, sort=False)
    registry = pd.concat(
        [load_players(TRAIN_SEASONS), _snapshot_registry(snapshot, season)], ignore_index=True
    )
    log.info(
        "assembled %d archive + %d live + %d upcoming fixture rows",
        len(archive),
        len(history),
        len(upcoming),
    )
    return combined, registry


def project_gameweek(snapshot: dict, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """One row per player for the next gameweek, with projection and decomposition."""
    gameweek = next_gameweek(snapshot)
    combined, registry = assemble(snapshot, season)

    seasons = (*TRAIN_SEASONS, season)
    features = build_features(seasons, write=False, silver=combined, players=registry)
    columns = feature_columns(features)

    is_target = (features["season"] == season) & (features["GW"] == gameweek["id"])
    is_upcoming = features["is_upcoming"].fillna(0).astype(int) == 1
    train = features[
        ~is_upcoming & (features["kickoff_time"] < features.loc[is_target, "kickoff_time"].min())
    ]
    target = features[is_target & is_upcoming].copy()
    if target.empty:
        raise ValueError(f"no upcoming rows for {season} GW{gameweek['id']}")

    log.info("fitting on %d rows, projecting %d players", len(train), len(target))
    model = fit_component_model(train, columns)
    breakdown = model.explain(target)

    out = target[
        [
            "code",
            "element",
            "web_name",
            "full_name",
            "position",
            "team",
            "opponent",
            "is_home",
            "fixtures_this_gw",
            "price",
            "selected",
            "selected_by_percent",
            "status",
            "chance_of_playing",
            "news",
            "availability",
            "fpl_ep_next",
            "minutes_share_r5",
        ]  # fmt: skip
    ].reset_index(drop=True)
    breakdown = breakdown.reset_index(drop=True)

    # The model's view of a fit player, then FPL's view of whether he plays.
    out["raw_expected_points"] = breakdown["expected_points"]
    availability = out["availability"].fillna(1.0).to_numpy()
    for term in CONTRIBUTIONS:
        out[term] = breakdown[term].to_numpy() * availability
    out["expected_points"] = out[list(CONTRIBUTIONS)].sum(axis=1)
    out["p_60"] = np.clip(breakdown["p_60"].to_numpy() * availability, 0, 1)
    out["gameweek"] = gameweek["id"]
    out["deadline"] = gameweek["deadline_time"]

    flagged = int((out["availability"] < 1).sum())
    log.info("GW%s: %d projections, %d availability-adjusted", gameweek["id"], len(out), flagged)
    return out.sort_values("expected_points", ascending=False).reset_index(drop=True)

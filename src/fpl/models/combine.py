"""Turn component predictions into expected points, using FPL's actual scoring rules.

This is deliberately arithmetic rather than another learned layer. Every term maps to
a line in the rulebook, so the output carries its own explanation: a projection of 6.2
decomposes into appearance, attacking, clean sheet and bonus contributions that sum to
6.2 exactly. The explanation layer narrates these numbers; it never invents them.
"""

from __future__ import annotations

import pandas as pd

from fpl.config import (
    APPEARANCE_POINTS,
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    CONCEDED_PER_MINUS_ONE,
    DEFENSIVE_CONTRIBUTION_POINTS,
    GOAL_POINTS,
    SAVES_PER_POINT,
    SIXTY_MINUTE_POINTS,
)
from fpl.evaluate.walkforward import Predictor
from fpl.models.components import ComponentEnsemble

# Human-readable label for each contribution, used by the explanation layer.
CONTRIBUTIONS = ("appearance", "attacking", "clean_sheet", "bonus", "goalkeeping", "defensive")


def _position_map(positions: pd.Series, table: dict[str, int]) -> pd.Series:
    """Look up a per-position points value, refusing to guess.

    Filling unknown positions with zero is what let assistant-manager rows through
    the pipeline scoring a predicted ~0 against an actual ~6. An unrecognised position
    is a data problem and must be loud.
    """
    mapped = positions.map(table)
    if mapped.isna().any():
        unknown = sorted(set(positions[mapped.isna()].dropna().unique()))
        raise ValueError(f"no scoring rule for position(s) {unknown}")
    return mapped.astype(float)


def decompose(components: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """Expand component predictions into per-source point contributions.

    Returns a frame whose ``expected_points`` column is the exact sum of the
    contribution columns beside it.
    """
    out = pd.DataFrame(index=components.index)
    positions = positions.reindex(components.index)

    # Appearance: 1 point per fixture played, 1 more per fixture reaching 60 minutes.
    out["appearance"] = (
        components["e_appearances"] * APPEARANCE_POINTS + components["e_full"] * SIXTY_MINUTE_POINTS
    )

    # Attacking: goals are worth more the further back you start.
    goal_value = _position_map(positions, GOAL_POINTS)
    out["attacking"] = components["e_goals"] * goal_value + components["e_assists"] * ASSIST_POINTS

    # Clean sheets: a count, so a double gameweek can earn two.
    cs_value = _position_map(positions, CLEAN_SHEET_POINTS)
    out["clean_sheet"] = components["e_cs"] * cs_value

    out["bonus"] = components["e_bonus"]

    # Goalkeeping: saves earn, concessions cost. Keepers only for saves; keepers and
    # defenders for the concession penalty.
    is_keeper = (positions == "GK").astype(float)
    is_back = positions.isin(["GK", "DEF"]).astype(float)
    out["goalkeeping"] = (
        is_keeper * components["e_saves"] / SAVES_PER_POINT
        - is_back * components["e_conceded"] / CONCEDED_PER_MINUS_ONE
    )

    # Defensive contribution (2025-26 rule): 2 points per fixture the threshold is hit.
    out["defensive"] = components["e_dc"] * DEFENSIVE_CONTRIBUTION_POINTS

    out["expected_points"] = out[list(CONTRIBUTIONS)].sum(axis=1)
    return out


class ComponentPredictor(Predictor):
    """Wraps the ensemble so it plugs into the same walk-forward harness."""

    def __init__(self, ensemble: ComponentEnsemble):
        self.ensemble = ensemble

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return self.explain(frame)["expected_points"]

    def explain(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predictions with the full decomposition attached."""
        components = self.ensemble.predict_components(frame)
        breakdown = decompose(components, frame["position"])
        return pd.concat([components, breakdown], axis=1)


def fit_component_model(train: pd.DataFrame, features: list[str]) -> ComponentPredictor:
    return ComponentPredictor(ComponentEnsemble.fit(train, features))

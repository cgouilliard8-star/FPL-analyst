"""Run models over identical folds, cache their predictions, and score them together.

Predictions are cached rather than recomputed because everything downstream wants
them: the optimiser picks squads from projections, and the dashboard renders them.
Refitting for each consumer would be wasteful and would risk two consumers silently
seeing different numbers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fpl.config import GOLD
from fpl.evaluate import walkforward
from fpl.evaluate.metrics import evaluate, scorecard
from fpl.features.build import feature_columns, load_features
from fpl.models import baseline

log = logging.getLogger(__name__)

DEFAULT_TEST_SEASON = "2024-25"
PREDICTIONS_DIR = GOLD / "predictions"

LABELS = {
    "constant": "constant (train mean)",
    "rolling3": "rolling 3-GW mean",
    "fpl_xp": "FPL own xP",
    "lightgbm": "LightGBM (monolithic)",
    "component": "component model",
}


def _fitters() -> dict[str, object]:
    from fpl.models.combine import fit_component_model

    return {
        "constant": baseline.fit_constant,
        "rolling3": baseline.fit_rolling3,
        "fpl_xp": baseline.fit_fpl_xp,
        "lightgbm": baseline.fit_lightgbm,
        "component": fit_component_model,
    }


def prediction_path(model: str, season: str) -> Path:
    return PREDICTIONS_DIR / f"{model}_{season}.parquet"


def horizon_path(model: str, season: str) -> Path:
    """Projections for the gameweeks *after* each fold's, made at that fold."""
    return PREDICTIONS_DIR / f"{model}_{season}_horizon.parquet"


def run_model(
    model: str,
    *,
    test_season: str = DEFAULT_TEST_SEASON,
    refit_every: int = 1,
    first_gameweek: int = 6,
    last_gameweek: int | None = None,
    frame: pd.DataFrame | None = None,
    horizon: int = 5,
) -> pd.DataFrame:
    """Walk-forward one model and cache its predictions.

    A gameweek range may be given so a long backtest can be run in chunks. Each run
    merges into the existing cache, replacing any gameweeks it just recomputed, so
    repeated partial runs converge on a complete season.
    """
    fitters = _fitters()
    if model not in fitters:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(fitters)}")

    frame = load_features() if frame is None else frame
    features = feature_columns(frame)

    predictions = walkforward.run(
        frame,
        features,
        fitters[model],
        test_season=test_season,
        refit_every=refit_every,
        first_gameweek=first_gameweek,
        last_gameweek=last_gameweek,
        name=LABELS[model],
        horizon=horizon if model == "component" else 1,
    )

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ahead = predictions.attrs.get("horizon")
    if ahead is not None:
        hpath = horizon_path(model, test_season)
        if hpath.exists():
            old = pd.read_parquet(hpath)
            ahead = pd.concat([old[~old["gameweek"].isin(ahead["gameweek"].unique())], ahead])
        ahead.sort_values(["gameweek", "k", "code"]).to_parquet(hpath, index=False)
    path = prediction_path(model, test_season)
    if path.exists():
        existing = pd.read_parquet(path)
        kept = existing[~existing["gameweek"].isin(predictions["gameweek"].unique())]
        predictions = pd.concat([kept, predictions], ignore_index=True)
        predictions = predictions.sort_values(["gameweek", "code"]).reset_index(drop=True)
    predictions.attrs = {}  # the horizon frame is saved on its own, above
    predictions.to_parquet(path, index=False)
    log.info(
        "cached %s (%d gameweeks, %d rows)",
        path.name,
        predictions["gameweek"].nunique(),
        len(predictions),
    )
    return predictions


def load_predictions(model: str, test_season: str = DEFAULT_TEST_SEASON) -> pd.DataFrame:
    path = prediction_path(model, test_season)
    if not path.exists():
        raise FileNotFoundError(f"no cached predictions for {model!r}; run `fpl backtest {model}`")
    return pd.read_parquet(path)


def build_scorecard(test_season: str = DEFAULT_TEST_SEASON) -> pd.DataFrame:
    """Score every model that has cached predictions for this season."""
    results = []
    for model, label in LABELS.items():
        path = prediction_path(model, test_season)
        if not path.exists():
            log.warning("skipping %s: not yet run", model)
            continue
        results.append(evaluate(pd.read_parquet(path), label))
    if not results:
        raise FileNotFoundError("no cached predictions; run `fpl backtest` first")
    return scorecard(results)

"""Walk-forward evaluation.

Train on every gameweek up to *t*, predict *t+1*, roll forward. Never a random split:
a random split lets the model learn from March to predict September, which inflates
every metric and describes a system that could not exist.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

Fitter = Callable[[pd.DataFrame, list[str]], "Predictor"]


class Predictor:
    """Anything with ``predict``. Models and naive baselines share the interface."""

    def predict(self, frame: pd.DataFrame) -> pd.Series:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class Fold:
    season: str
    gameweek: int
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def label(self) -> str:
        return f"{self.season} GW{self.gameweek}"


def folds(
    frame: pd.DataFrame,
    *,
    test_season: str,
    first_gameweek: int = 6,
    last_gameweek: int | None = None,
) -> Iterator[Fold]:
    """Yield one fold per gameweek of the test season.

    Training data is everything that kicked off before the test gameweek's first
    match, which naturally includes earlier seasons. ``first_gameweek`` skips the
    opening weeks, where rolling features have almost no history to work with, and
    ``last_gameweek`` bounds the other end so a long backtest can be run in chunks.
    """
    test_rows = frame[frame["season"] == test_season]
    if test_rows.empty:
        raise ValueError(f"no rows for test season {test_season}")

    for gameweek in sorted(test_rows["GW"].unique()):
        if gameweek < first_gameweek:
            continue
        if last_gameweek is not None and gameweek > last_gameweek:
            continue
        test = test_rows[test_rows["GW"] == gameweek]
        cutoff = test["kickoff_time"].min()
        train = frame[frame["kickoff_time"] < cutoff]
        if train.empty:
            continue
        yield Fold(test_season, int(gameweek), train, test)


def run(
    frame: pd.DataFrame,
    features: list[str],
    fit: Fitter,
    *,
    test_season: str,
    first_gameweek: int = 6,
    last_gameweek: int | None = None,
    refit_every: int = 1,
    target: str = "total_points",
    name: str = "model",
) -> pd.DataFrame:
    """Execute the walk-forward loop and collect predictions.

    ``refit_every`` trades fidelity for time: 1 refits before every gameweek, which is
    what a real deployment would do.
    """
    predictions: list[pd.DataFrame] = []
    model: Predictor | None = None

    fold_iter = folds(
        frame,
        test_season=test_season,
        first_gameweek=first_gameweek,
        last_gameweek=last_gameweek,
    )
    for index, fold in enumerate(fold_iter):
        if model is None or index % refit_every == 0:
            started = time.perf_counter()
            model = fit(fold.train, features)
            log.info(
                "%s | refit %s on %d rows in %.1fs",
                name,
                fold.label,
                len(fold.train),
                time.perf_counter() - started,
            )

        predictions.append(
            pd.DataFrame(
                {
                    "code": fold.test["code"].to_numpy(),
                    "season": fold.season,
                    "gameweek": fold.gameweek,
                    "position": fold.test["position"].to_numpy(),
                    "actual": fold.test[target].to_numpy(),
                    "predicted": model.predict(fold.test).to_numpy(),
                }
            )
        )

    if not predictions:
        raise ValueError("walk-forward produced no folds")

    result = pd.concat(predictions, ignore_index=True)
    log.info(
        "%s: %d predictions over %d gameweeks",
        name,
        len(result),
        result["gameweek"].nunique(),
    )
    return result

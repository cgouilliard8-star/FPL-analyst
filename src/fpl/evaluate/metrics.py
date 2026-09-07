"""Scoring. Overall MAE flatters everyone, so nothing here reports it alone.

60.6% of player-gameweeks score zero and another 25.5% score one or two. A model
that predicts "everybody gets 1.2" achieves a respectable-looking MAE and is worth
nothing. The segments below are where models actually differ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# OpenFPL's segmentation (arXiv 2508.09992), adopted so results are comparable.
SEGMENTS: dict[str, tuple[float, float]] = {
    "zeros": (-np.inf, 0.0),
    "blanks": (1.0, 2.0),
    "tickers": (3.0, 4.0),
    "haulers": (5.0, np.inf),
}


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def segmented_rmse(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """RMSE within each outcome band, keyed on what actually happened."""
    out: dict[str, float] = {}
    for name, (low, high) in SEGMENTS.items():
        mask = actual.between(low, high)
        out[name] = (
            rmse(actual[mask].to_numpy(), predicted[mask].to_numpy()) if mask.any() else np.nan
        )
        out[f"{name}_n"] = int(mask.sum())
    return out


def precision_at_k(frame: pd.DataFrame, k: int = 10, *, by: str = "gameweek") -> float:
    """Of the k highest-projected players each gameweek, what share actually hauled?

    This is closer to the real question than any error metric: you field eleven
    players, so what matters is whether the right names float to the top.
    """
    haul = SEGMENTS["haulers"][0]
    hits = []
    for _, group in frame.groupby(by):
        if len(group) < k:
            continue
        top = group.nlargest(k, "predicted")
        hits.append((top["actual"] >= haul).mean())
    return float(np.mean(hits)) if hits else np.nan


def rank_correlation(frame: pd.DataFrame, *, by: str = "gameweek") -> float:
    """Mean within-gameweek Spearman correlation between projection and outcome."""
    values = []
    for _, group in frame.groupby(by):
        if group["predicted"].nunique() < 2 or len(group) < 10:
            continue
        rho = spearmanr(group["predicted"], group["actual"]).statistic
        if not np.isnan(rho):
            values.append(rho)
    return float(np.mean(values)) if values else np.nan


def evaluate(frame: pd.DataFrame, name: str = "model") -> dict[str, float]:
    """Full scorecard for a frame of (gameweek, actual, predicted)."""
    actual, predicted = frame["actual"], frame["predicted"]
    scores: dict[str, float] = {
        "model": name,
        "n": len(frame),
        "rmse": rmse(actual.to_numpy(), predicted.to_numpy()),
        "mae": mae(actual.to_numpy(), predicted.to_numpy()),
        "spearman": rank_correlation(frame),
        "precision@10": precision_at_k(frame, 10),
        "precision@20": precision_at_k(frame, 20),
    }
    scores.update(segmented_rmse(actual, predicted))
    return scores


def scorecard(results: list[dict[str, float]]) -> pd.DataFrame:
    """Tidy comparison table, best rank correlation first."""
    frame = pd.DataFrame(results)
    ordered = [
        "model",
        "n",
        "rmse",
        "mae",
        "spearman",
        "precision@10",
        "precision@20",
        "zeros",
        "blanks",
        "tickers",
        "haulers",
    ]
    columns = [c for c in ordered if c in frame.columns]
    return frame[columns].sort_values("spearman", ascending=False).reset_index(drop=True)

"""Per-position calibration factors from cached walk-forward predictions.

Usage: python scripts/calibrate.py [predictions.parquet]
Prints the dict to paste into ``fpl.config.CALIBRATION``.
"""

from __future__ import annotations

import sys

import pandas as pd


def main(path: str = "data/gold/predictions/component_2024-25.parquet") -> None:
    preds = pd.read_parquet(path)
    by = preds.groupby("position")[["predicted", "actual"]].mean()
    factors = (by["actual"] / by["predicted"]).round(3)
    print(
        f"overall: predicted {preds['predicted'].mean():.3f}, actual {preds['actual'].mean():.3f}"
    )
    print("CALIBRATION =", {k: float(v) for k, v in factors.items()})


if __name__ == "__main__":
    main(*sys.argv[1:])

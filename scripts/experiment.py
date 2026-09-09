"""Coarse walk-forward harness for comparing model or feature changes quickly.

Refits at gameweeks 6, 14, 22 and 30 of the test season and scores every gameweek
from 6 on, so a candidate is judged on identical folds in a few minutes on two cores.
Usage: python scripts/experiment.py LABEL [--season 2024-25] [--rebuild] [--refit 8]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from fpl.evaluate import walkforward
from fpl.evaluate.metrics import evaluate
from fpl.features.build import build_features, feature_columns, load_features
from fpl.models.combine import fit_component_model

RESULTS = Path("data/gold/experiments.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--season", default="2024-25")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--refit", type=int, default=8)
    ap.add_argument("--first", type=int, default=6)
    ap.add_argument("--minutes", choices=["rate", "poisson"], default=None)
    ap.add_argument("--halflife", type=float, default=None, help="recency half-life, days")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    from fpl.models import components

    if args.minutes:
        components.MINUTES_MODEL = args.minutes
    if args.halflife is not None:
        components.RECENCY_HALFLIFE_DAYS = args.halflife
    if args.seeds:
        components.SEEDS = args.seeds

    frame = build_features() if args.rebuild else load_features()
    features = feature_columns(frame)
    t0 = time.time()
    preds = walkforward.run(
        frame, features, fit_component_model, test_season=args.season,
        first_gameweek=args.first, refit_every=args.refit, name=args.label,
    )  # fmt: skip
    preds.to_parquet(f"data/gold/predictions/exp_{args.label}_{args.season}.parquet", index=False)
    score = evaluate(preds, args.label)
    score["season"] = args.season
    score["n_features"] = len(features)
    score["seconds"] = round(time.time() - t0)
    row = pd.DataFrame([score])
    if RESULTS.exists():
        row = pd.concat([pd.read_csv(RESULTS), row], ignore_index=True)
    row.to_csv(RESULTS, index=False)
    print(json.dumps(score, indent=1, default=str), flush=True)
    print(row.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

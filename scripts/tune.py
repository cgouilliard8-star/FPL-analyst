"""Small hyper-parameter search for the component model, judged by the walk-forward.

Each candidate is fitted at gameweeks 6, 14, 22 and 30 of the test season (refit
every eight) and scored on the gameweeks in between, on identical folds, so the
comparison is fair and the whole grid runs in well under an hour on two cores.
Usage: python scripts/tune.py [test_season]
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from fpl.evaluate import walkforward
from fpl.evaluate.metrics import evaluate
from fpl.features.build import feature_columns, load_features
from fpl.models import components
from fpl.models.combine import fit_component_model

GRID = [
    {},  # current defaults
    {"n_estimators": 600, "learning_rate": 0.03},
    {"num_leaves": 15, "min_child_samples": 80},
    {"num_leaves": 63, "min_child_samples": 20},
    {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 60, "colsample_bytree": 0.5},
    {"reg_lambda": 5.0, "colsample_bytree": 0.5},
]


def main(test_season: str = "2024-25") -> None:
    frame = load_features()
    features = feature_columns(frame)
    base = dict(components._COMMON)
    results = []
    for params in GRID:
        components._COMMON.clear()
        components._COMMON.update(base)
        components._COMMON.update(params)
        t0 = time.time()
        preds = walkforward.run(
            frame, features, fit_component_model, test_season=test_season,
            first_gameweek=6, refit_every=8, name=json.dumps(params),
        )  # fmt: skip
        score = evaluate(preds, json.dumps(params))
        score["seconds"] = round(time.time() - t0)
        results.append(score)
        print(pd.DataFrame(results).to_string(index=False), flush=True)
    components._COMMON.clear()
    components._COMMON.update(base)
    pd.DataFrame(results).to_csv("data/gold/tuning.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])

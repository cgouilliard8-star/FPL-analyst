"""Does a haul-probability model pick better captains than expected points alone?

Coarse walk-forward (refit at gameweeks 6, 14, 22, 30 of the test season); for every
gameweek the captain is chosen from a plausible squad -- the fifteen highest projected
players -- by ``expected points + w * P(haul)`` for a grid of weights, and what the
chosen captain actually scored is averaged over the season. The weight that scores
most goes into ``config.CAPTAIN_HAUL_WEIGHT``.
Usage: python scripts/captaincy.py [test_season]
"""

from __future__ import annotations

import sys

import pandas as pd

from fpl.evaluate import walkforward
from fpl.features.build import feature_columns
from fpl.models.combine import fit_component_model

WEIGHTS = (0.0, 2.0, 4.0, 6.0, 8.0, 12.0)


def main(test_season: str = "2024-25") -> None:
    from fpl.features.build import load_features

    frame = load_features()
    features = feature_columns(frame)
    rows = []
    for fold_index, fold in enumerate(
        walkforward.folds(frame, test_season=test_season, first_gameweek=6)
    ):
        if fold_index % 8 == 0:
            model = fit_component_model(fold.train, features)
            print(f"refit at {fold.label}", flush=True)
        out = model.explain(fold.test)
        rows.append(
            pd.DataFrame(
                {
                    "gameweek": fold.gameweek,
                    "code": fold.test["code"].to_numpy(),
                    "ep": out["expected_points"].to_numpy(),
                    "p_haul": out["p_haul"].to_numpy(),
                    "actual": fold.test["total_points"].to_numpy(),
                }
            )
        )
    preds = pd.concat(rows, ignore_index=True)
    preds.to_parquet(f"data/gold/predictions/captaincy_{test_season}.parquet", index=False)

    results = []
    for w in WEIGHTS:
        picked = []
        for _, g in preds.groupby("gameweek"):
            squad = g.nlargest(15, "ep")
            best = squad.assign(score=squad["ep"] + w * squad["p_haul"]).nlargest(1, "score")
            picked.append(float(best["actual"].iloc[0]))
        results.append({"weight": w, "captain_points": sum(picked) / len(picked), "n": len(picked)})
    oracle = preds.groupby("gameweek").apply(lambda g: g.nlargest(15, "ep")["actual"].max()).mean()
    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    print(f"perfect hindsight within the same fifteen: {oracle:.2f}")
    print(
        "p_haul calibration:",
        preds.assign(b=pd.cut(preds.p_haul, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1]))
        .groupby("b", observed=True)
        .agg(
            pred=("p_haul", "mean"),
            actual=("actual", lambda a: (a >= 8).mean()),
            n=("actual", "size"),
        )
        .round(3)
        .to_string(),
    )


if __name__ == "__main__":
    main(*sys.argv[1:])

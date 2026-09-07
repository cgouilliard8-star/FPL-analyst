"""Command line entry point: ``fpl <command>``."""

from __future__ import annotations

import argparse
import logging
import sys

from fpl.config import TRAIN_SEASONS


def _bootstrap(args: argparse.Namespace) -> int:
    from fpl.data.archive import load_seasons

    seasons = tuple(args.seasons) if args.seasons else TRAIN_SEASONS
    frame = load_seasons(seasons, refresh=args.refresh)
    print(f"{len(frame):,} player-gameweek rows across {len(seasons)} seasons")
    print(
        f"gameweeks {frame['GW'].min()}-{frame['GW'].max()}, "
        f"{frame['name'].nunique():,} distinct players"
    )
    return 0


def _silver(args: argparse.Namespace) -> int:
    from fpl.data.silver import build_silver

    seasons = tuple(args.seasons) if args.seasons else TRAIN_SEASONS
    frame = build_silver(seasons)
    print(
        f"{len(frame):,} rows | {frame['code'].nunique():,} players | "
        f"{frame['team'].nunique()} clubs | seasons {frame['season'].min()}-{frame['season'].max()}"
    )
    return 0


def _features(args: argparse.Namespace) -> int:
    from fpl.features.build import build_features, feature_columns

    seasons = tuple(args.seasons) if args.seasons else TRAIN_SEASONS
    frame = build_features(seasons)
    print(
        f"{len(frame):,} player-gameweeks | {len(feature_columns(frame))} features | "
        f"{frame['code'].nunique():,} players"
    )
    return 0


def _backtest(args: argparse.Namespace) -> int:
    from fpl.evaluate.compare import LABELS, run_model

    models = args.models or list(LABELS)
    for model in models:
        run_model(
            model,
            test_season=args.season,
            refit_every=args.refit_every,
            first_gameweek=args.from_gw,
            last_gameweek=args.to_gw,
        )
    return _scorecard(args)


def _scorecard(args: argparse.Namespace) -> int:
    import pandas as pd

    from fpl.evaluate.compare import build_scorecard

    table = build_scorecard(args.season)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def _optimise(args: argparse.Namespace) -> int:
    from fpl.evaluate.compare import load_predictions
    from fpl.features.build import load_features
    from fpl.optimise.squad import backtest

    results = backtest(load_predictions(args.model), load_features())
    print(
        f"{args.model}: {len(results)} gameweeks | "
        f"{results['points'].sum():.0f} points total | "
        f"{results['points'].mean():.1f} per gameweek"
    )
    return 0


def _publish(args: argparse.Namespace) -> int:
    from fpl.report.publish import build_gameweek, write_gameweek

    for gameweek in args.gameweeks:
        payload = build_gameweek(gameweek, season=args.season, explain=not args.no_explain)
        path = write_gameweek(payload)
        squad = payload["squad"]
        print(
            f"GW{gameweek}: projected {squad['projected']:.1f}, "
            f"actual {squad['actual']}, spend GBP{squad['spend']}m -> {path.name}"
        )
    return 0


def _snapshot(args: argparse.Namespace) -> int:
    from fpl.data.fpl_api import fetch_snapshot, next_gameweek, save_snapshot

    snapshot = fetch_snapshot()
    path = save_snapshot(snapshot)
    gameweek = next_gameweek(snapshot)
    flagged = sum(1 for e in snapshot["elements"] if e["status"] != "a")
    print(
        f"GW{gameweek['id']} (deadline {gameweek['deadline_time']}): "
        f"{len(snapshot['elements'])} players, {flagged} flagged -> {path.name}"
    )
    return 0


def _simulate(args: argparse.Namespace) -> int:
    import json

    from fpl.data.fpl_api import load_latest_snapshot
    from fpl.evaluate.season_sim import simulate_season
    from fpl.report.live import SEASON_SIM_PATH

    gameweeks = tuple(range(args.from_gw, args.to_gw + 1))
    result = simulate_season(load_latest_snapshot(), gameweeks=gameweeks)
    SEASON_SIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEASON_SIM_PATH.write_text(json.dumps(result, indent=1))
    for g in result["gameweeks"]:
        moves = "; ".join(
            f"{', '.join(t['out_names'])} -> {', '.join(t['in_names'])}" for t in g["transfers"]
        )
        print(
            f"GW{g['gameweek']}: {g['points']} pts (average {g['average']}, "
            f"highest {g['highest']}) {moves or 'rolled'}"
        )
    print(f"total {result['total']} vs average {result['average_total']} -> {SEASON_SIM_PATH}")
    return 0


def _live(args: argparse.Namespace) -> int:
    from fpl.report.live import build_live, write_live

    payload = build_live(explain=not args.no_explain)
    path = write_live(payload)
    meta, optimal = payload["meta"], payload["optimal"]
    print(
        f"GW{meta['gameweek']}: {meta['players']} projections, {meta['flagged']} flagged, "
        f"best squad {optimal['projected']:.1f} pts -> {path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    # Shared flags live on a parent parser so they work either side of the
    # subcommand: both "fpl -v bootstrap" and "fpl bootstrap -v" are accepted.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="log progress")

    parser = argparse.ArgumentParser(prog="fpl", description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser(
        "bootstrap",
        parents=[common],
        help="download historical seasons into bronze",
    )
    boot.add_argument("--seasons", nargs="*", help=f"default: {' '.join(TRAIN_SEASONS)}")
    boot.add_argument("--refresh", action="store_true", help="ignore the local cache")
    boot.set_defaults(func=_bootstrap)

    silver = sub.add_parser(
        "silver",
        parents=[common],
        help="resolve identity and build the canonical table",
    )
    silver.add_argument("--seasons", nargs="*", help=f"default: {' '.join(TRAIN_SEASONS)}")
    silver.set_defaults(func=_silver)

    feats = sub.add_parser("features", parents=[common], help="build the leak-free feature table")
    feats.add_argument("--seasons", nargs="*", help=f"default: {' '.join(TRAIN_SEASONS)}")
    feats.set_defaults(func=_features)

    back = sub.add_parser(
        "backtest", parents=[common], help="walk-forward comparison of every model"
    )
    back.add_argument("--season", default="2024-25", help="season to test on")
    back.add_argument("--refit-every", type=int, default=1, help="gameweeks between refits")
    back.add_argument("models", nargs="*", help="models to run (default: all)")
    back.set_defaults(func=_backtest)

    card = sub.add_parser(
        "scorecard", parents=[common], help="score whatever predictions are cached"
    )
    card.add_argument("--season", default="2024-25")
    card.set_defaults(func=_scorecard)

    opt = sub.add_parser(
        "optimise", parents=[common], help="backtest the squad optimiser on projections"
    )
    opt.add_argument("--model", default="component")
    opt.set_defaults(func=_optimise)

    pub = sub.add_parser(
        "publish", parents=[common], help="write dashboard JSON for one or more gameweeks"
    )
    pub.add_argument("gameweeks", nargs="+", type=int)
    pub.add_argument("--season", default="2024-25")
    pub.add_argument("--no-explain", action="store_true", help="skip rationale generation")
    pub.set_defaults(func=_publish)

    snap = sub.add_parser(
        "snapshot", parents=[common], help="pull the live FPL API into an immutable snapshot"
    )
    snap.set_defaults(func=_snapshot)

    live = sub.add_parser(
        "live", parents=[common], help="project the next gameweek and write site/data/live.json"
    )
    live.add_argument("--no-explain", action="store_true", help="skip rationale generation")
    live.set_defaults(func=_live)

    sim = sub.add_parser(
        "simulate",
        parents=[common],
        help="replay the season from GW1 with one free transfer a week, scored on real points",
    )
    sim.add_argument("--from-gw", type=int, default=1)
    sim.add_argument("--to-gw", type=int, default=3)
    sim.set_defaults(func=_simulate)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

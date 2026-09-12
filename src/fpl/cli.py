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
    result = simulate_season(load_latest_snapshot(), gameweeks=gameweeks, planner=args.planner)
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


def _schedule(args: argparse.Namespace) -> int:
    from fpl.config import CURRENT_SEASON, TRAIN_SEASONS
    from fpl.data.schedule import fetch_schedule, schedule_path

    seasons = (*TRAIN_SEASONS, CURRENT_SEASON) if args.all else (CURRENT_SEASON,)
    for season in seasons:
        frame = fetch_schedule(season, domestic=not args.no_domestic)
        by = frame.groupby("competition").size().to_dict() if not frame.empty else {}
        print(f"{season}: {len(frame)} matches {by} -> {schedule_path(season).name}")
    return 0


def _replay(args: argparse.Namespace) -> int:
    import json

    from fpl.evaluate.season import replay
    from fpl.report.live import replay_path

    out = replay(args.season)
    try:  # the walk-forward scorecard for the same season, for the page's accuracy table
        from fpl.evaluate.compare import build_scorecard

        card = build_scorecard(args.season)
        keep = [
            c
            for c in ("model", "n", "spearman", "precision@10", "rmse", "haulers")
            if c in card.columns
        ]
        out["scorecard"] = json.loads(card[keep].to_json(orient="records"))
    except FileNotFoundError:
        pass
    path = replay_path(args.season)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    for m, r in out.items():
        if m == "scorecard":
            continue
        extra = f", {r['hits']} points of hits, chips {r['chips']}" if "hits" in r else ""
        print(
            f"{m}: {r['total']} points over {len(r['gameweeks'])} gameweeks"
            f" ({r['per_gameweek']} a week{extra})"
        )
    return 0


def _odds(args: argparse.Namespace) -> int:
    from fpl.config import CURRENT_SEASON, TRAIN_SEASONS
    from fpl.data.odds import fetch_live_odds, fetch_odds, load_odds

    seasons = (*TRAIN_SEASONS, CURRENT_SEASON) if args.all else (CURRENT_SEASON,)
    stored = fetch_odds(seasons)
    live = fetch_live_odds()
    table = load_odds((*TRAIN_SEASONS, CURRENT_SEASON))
    print(
        f"fetched {stored}; live prices for {live} matches; {len(table) // 2} matches with odds on disk"
    )
    return 0


def _core(args: argparse.Namespace) -> int:
    from fpl.config import CURRENT_SEASON, TRAIN_SEASONS
    from fpl.data.core_insights import fetch_core_insights

    seasons = (*TRAIN_SEASONS, CURRENT_SEASON) if args.all else (CURRENT_SEASON,)
    written = fetch_core_insights(seasons, source=args.source)
    print("core insights: " + ", ".join(f"{k}: {v} rows" for k, v in written.items()))
    return 0


def _photos(args: argparse.Namespace) -> int:
    from fpl.data.fpl_api import load_latest_snapshot
    from fpl.data.photos import fetch_photos

    counts = fetch_photos(load_latest_snapshot())
    print(
        f"photos: {counts['fetched']} fetched, {counts['present']} present, {counts['failed']} failed"
    )
    return 0


def _due(args: argparse.Namespace) -> int:
    from fpl.report.due import main as decide_due

    verdict = decide_due(force=args.force)
    print(f"{'due' if verdict.due else 'not due'}: {verdict.reason}")
    return 0


def _check(args: argparse.Namespace) -> int:
    from fpl.report.check import check_file
    from fpl.report.live import LIVE_PATH

    problems = check_file(LIVE_PATH)
    if problems:
        for p in problems:
            print(f"FAIL {p}")
        return 1
    print(f"ok {LIVE_PATH}")
    return 0


def _projection_frame(payload: dict):
    """The columns the projection log keeps, read back off the built payload so the
    model is not run twice."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "code": p["code"], "web_name": p["web_name"], "position": p["position"],
                "team": p["team"], "price": p["price"], "ep1": p["ep1"], "ep5": p["ep5"],
                "availability": p["availability"], "p_60": p["p60"], "p_haul": p.get("haul"),
                "ep1_model": p.get("ep_model", p["ep1"]), "fpl_ep": p.get("fpl_ep", 0.0),
                "gameweek": payload["meta"]["gameweek"],
            }
            for p in payload["players"]
        ]
    )  # fmt: skip


def _live(args: argparse.Namespace) -> int:
    from fpl.data.fpl_api import load_latest_snapshot
    from fpl.report.live import build_live, write_live
    from fpl.report.projection_log import save_projections

    snapshot = load_latest_snapshot()
    payload = build_live(snapshot, explain=not args.no_explain)
    # Write this gameweek's projections down before the deadline, so the live model
    # can be scored honestly later (see fpl.report.projection_log).
    save_projections(_projection_frame(payload), captured_at=snapshot["captured_at"])
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
    back.add_argument("--from-gw", type=int, default=6, help="first gameweek to score")
    back.add_argument("--to-gw", type=int, default=None, help="last gameweek to score")
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

    sch = sub.add_parser(
        "schedule",
        parents=[common],
        help="fetch cup and European fixtures for PL clubs (fixture congestion)",
    )
    sch.add_argument("--all", action="store_true", help="every training season, not just this one")
    sch.add_argument("--no-domestic", action="store_true", help="skip the FA Cup / League Cup")
    sch.set_defaults(func=_schedule)

    rp = sub.add_parser(
        "replay",
        parents=[common],
        help="replay an archived season with the full rules (transfers, hits, chips): model vs average vs crowd",
    )
    rp.add_argument("--season", default="2024-25")
    rp.set_defaults(func=_replay)

    od = sub.add_parser(
        "odds",
        parents=[common],
        help="fetch bookmaker odds (football-data.co.uk) for results and upcoming fixtures",
    )
    od.add_argument(
        "--all", action="store_true", help="every training season, not just the current one"
    )
    od.set_defaults(func=_odds)

    core = sub.add_parser(
        "core",
        parents=[common],
        help="fetch per-match player actions from FPL-Core-Insights (GitHub)",
    )
    core.add_argument(
        "--all", action="store_true", help="every training season, not just the current one"
    )
    core.add_argument(
        "--source", default=None, help="a local checkout of the dataset instead of cloning"
    )
    core.set_defaults(func=_core)

    ph = sub.add_parser(
        "photos", parents=[common], help="copy every current player's FPL headshot into site/photos"
    )
    ph.set_defaults(func=_photos)

    due = sub.add_parser(
        "due", parents=[common],
        help="is a refresh worth running now? (near a deadline, or a day since the last)",
    )  # fmt: skip
    due.add_argument("--force", action="store_true", help="always due (manual runs)")
    due.set_defaults(func=_due)

    chk = sub.add_parser(
        "check", parents=[common], help="validate site/data/live.json before it is deployed"
    )
    chk.set_defaults(func=_check)

    sim = sub.add_parser(
        "simulate",
        parents=[common],
        help="replay the season from GW1 with one free transfer a week, scored on real points",
    )
    sim.add_argument("--from-gw", type=int, default=1)
    sim.add_argument("--to-gw", type=int, default=3)
    sim.add_argument(
        "--planner", choices=["milp", "greedy"], default="milp",
        help="transfer planner: the multi-week solver (default) or the one-week suggester",
    )  # fmt: skip
    sim.set_defaults(func=_simulate)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

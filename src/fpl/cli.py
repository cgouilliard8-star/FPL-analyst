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

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

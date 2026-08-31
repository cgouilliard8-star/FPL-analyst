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
    print(f"gameweeks {frame['GW'].min()}-{frame['GW'].max()}, "
          f"{frame['name'].nunique():,} distinct players")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fpl", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("bootstrap", help="download historical seasons into bronze")
    boot.add_argument("--seasons", nargs="*", help=f"default: {' '.join(TRAIN_SEASONS)}")
    boot.add_argument("--refresh", action="store_true", help="ignore the local cache")
    boot.set_defaults(func=_bootstrap)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

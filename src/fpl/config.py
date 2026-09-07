"""Central configuration: paths, seasons, and FPL scoring constants."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FPL_DATA_DIR", PROJECT_ROOT / "data"))

BRONZE = DATA_DIR / "bronze"  # immutable, deadline-stamped snapshots
SILVER = DATA_DIR / "silver"  # canonical, entity-resolved
GOLD = DATA_DIR / "gold"  # leak-free feature store


def ensure_data_dirs() -> None:
    """Create the data layers. Called by loaders, not at import time, so importing
    the package never writes to disk."""
    for layer in (BRONZE, SILVER, GOLD):
        layer.mkdir(parents=True, exist_ok=True)


# Seasons used for training.
#
# 2021-22 is deliberately excluded: it carries no expected_goals, expected_assists,
# expected_goals_conceded or starts columns at all (measured: 0% non-null). Including
# it would mean a quarter of the training set has none of the features the attacking
# model is built on. The archive goes back to 2016-17 with the same limitation.
TRAIN_SEASONS: tuple[str, ...] = ("2022-23", "2023-24", "2024-25")
ARCHIVE_SEASONS: tuple[str, ...] = ("2021-22", *TRAIN_SEASONS)
CURRENT_SEASON = "2026-27"

FPL_API_BASE = "https://fantasy.premierleague.com/api"
ARCHIVE_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

POSITIONS = ("GK", "DEF", "MID", "FWD")

# --- FPL scoring rules (2025-26 onward) -------------------------------------
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
APPEARANCE_POINTS = 1  # for playing at all
SIXTY_MINUTE_POINTS = 1  # additional, for 60+ minutes
DEFENSIVE_CONTRIBUTION_POINTS = 2
# Defensive-contribution thresholds (CBIT for defenders, CBIRT for others).
# NOTE: this rule arrived in 2025-26, so the column does not exist in any season we
# train on. The component is implemented and wired into the combiner, but contributes
# zero until 2025-26+ data is present. See models/combine.py.
DC_THRESHOLD = {"GK": None, "DEF": 10, "MID": 12, "FWD": 12}
SAVES_PER_POINT = 3
CONCEDED_PER_MINUS_ONE = 2  # GK/DEF only
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
OWN_GOAL_POINTS = -2
PENALTY_MISS_POINTS = -2
PENALTY_SAVE_POINTS = 5

# --- squad constraints ------------------------------------------------------
SQUAD_SIZE = 15
SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
XI_SIZE = 11
MAX_PER_CLUB = 3
BUDGET_TENTHS = 1000  # £100.0m, stored in tenths as the API does
TRANSFER_HIT = 4  # points cost of an extra transfer

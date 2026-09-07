"""Canonical player and team identity across seasons and across sources.

Why this module exists
----------------------
FPL's ``element`` id (the ``element`` column in ``merged_gw.csv``) is **season-local
and reused**. Measured across the 2023-24 and 2024-25 registries: 803 ids appear in
both seasons, and in all 803 cases they refer to a *different footballer*.

Joining gameweek rows across seasons on ``element`` therefore produces 100% wrong
player identity, silently. Every rolling form feature would be computed over a
mixture of unrelated players and nothing would visibly break.

The cross-season identifier is ``code``, which lives in ``players_raw.csv``. This
module builds the ``(season, element) -> code`` bridge and exposes name matching for
the sources that have no FPL id at all.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from fpl.config import POSITIONS

log = logging.getLogger(__name__)

MAPPINGS_DIR = Path(__file__).parent / "mappings"
TEAM_ALIASES_PATH = MAPPINGS_DIR / "teams.csv"
PLAYER_OVERRIDES_PATH = MAPPINGS_DIR / "players.csv"

# element_type 5 is a *manager*, introduced in 2024-25 alongside the Assistant
# Manager chip. Managers score by completely different rules (match result, team
# clean sheet, goal difference) and are not footballers. They are labelled here
# rather than dropped, so that filtering them out is a visible decision.
ELEMENT_TYPE_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", 5: "MGR"}
MANAGER_POSITION = "MGR"

# Below this rapidfuzz score a match is not trusted and must be resolved by hand.
MATCH_THRESHOLD = 88


def normalise_name(name: str) -> str:
    """Fold a name to a comparable form.

    Strips diacritics, punctuation and case, and collapses whitespace, so that
    "Gabriel dos Santos Magalhães" and "Gabriel Dos Santos Magalhaes" agree.
    """
    if not isinstance(name, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", ascii_only)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def build_player_index(players: pd.DataFrame) -> pd.DataFrame:
    """Build the ``(season, element) -> code`` bridge from player registries.

    Args:
        players: concatenated ``players_raw`` frames, as returned by
            ``fpl.data.archive.load_players``.

    Returns:
        One row per (season, element) with the stable ``code``, full name and position.
    """
    index = players.rename(columns={"id": "element"})[
        ["season", "element", "code", "first_name", "second_name", "element_type"]
    ].copy()

    index["position"] = index["element_type"].map(ELEMENT_TYPE_TO_POSITION)
    if index["position"].isna().any():
        unknown = sorted(index.loc[index["position"].isna(), "element_type"].unique())
        raise ValueError(f"unknown element_type values {unknown}; FPL added a position")

    index["full_name"] = (
        index["first_name"].fillna("") + " " + index["second_name"].fillna("")
    ).str.strip()
    index["norm_name"] = index["full_name"].map(normalise_name)

    duplicated = index.duplicated(subset=["season", "element"]).sum()
    if duplicated:
        raise ValueError(f"{duplicated} duplicate (season, element) pairs in the registry")

    return index.drop(columns=["element_type"])


def drop_managers(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove manager rows, logging how many went.

    The player model predicts footballer points. Managers need their own model and
    are out of scope for v1 — but they are removed loudly, because a silent filter
    is how a row count quietly stops matching the thing you think you are modelling.
    """
    if "position" not in frame.columns:
        raise ValueError("call attach_player_code before drop_managers")
    is_manager = frame["position"] == MANAGER_POSITION
    if is_manager.any():
        log.info(
            "dropping %d manager rows (out of scope for the player model)",
            int(is_manager.sum()),
        )
    kept = frame.loc[~is_manager].copy()

    # Anything left must be one of the four playing positions. A stray label here
    # means a scoring lookup will silently return zero for those rows.
    unexpected = set(kept["position"].dropna().unique()) - set(POSITIONS)
    if unexpected:
        raise ValueError(f"unexpected positions after dropping managers: {sorted(unexpected)}")
    return kept


def attach_player_code(gameweeks: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Attach the cross-season ``code`` to gameweek rows.

    Raises:
        ValueError: any gameweek row fails to resolve. A partial join here would
            corrupt every downstream rolling feature, so it is fatal rather than
            something to fix up later.
    """
    registry = index[["season", "element", "code", "position", "full_name"]].rename(
        columns={"position": "registry_position"}
    )
    merged = gameweeks.merge(registry, on=["season", "element"], how="left")

    unresolved = merged["code"].isna()
    if unresolved.any():
        sample = (
            merged.loc[unresolved, ["season", "element", "name"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{unresolved.sum()} gameweek rows have no registry entry. Sample: {sample}"
        )

    merged["code"] = merged["code"].astype("int64")

    # The gameweek files carry their own `position` column, and it does not agree with
    # the registry: it labels assistant managers "AM" where the registry says "MGR".
    # Letting the gameweek column win silently defeated drop_managers and left 322
    # manager fixture-rows in the 2024-25 training data, predicted at ~0 against an
    # actual average of ~6. The registry is authoritative, so it overwrites here.
    merged["position"] = merged["registry_position"]
    return merged.drop(columns=["registry_position"])


def _load_overrides(path: Path) -> dict[str, int]:
    """Read the hand-curated name -> code overrides, if any exist yet."""
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return dict(zip(frame["source_name"].map(normalise_name), frame["code"], strict=False))


def match_external_names(
    names: list[str],
    index: pd.DataFrame,
    *,
    threshold: int = MATCH_THRESHOLD,
    overrides_path: Path = PLAYER_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Match names from an external source (Understat, odds feeds) to FPL codes.

    Fuzzy matching proposes; the overrides file decides. Anything scoring below
    ``threshold`` is returned unmatched rather than guessed at, so that ambiguous
    cases surface as a short review list instead of quietly wrong joins.

    Returns:
        Frame with columns ``source_name, code, matched_name, score, method``.
        ``code`` is ``<NA>`` where no confident match was found.
    """
    from rapidfuzz import process
    from rapidfuzz.fuzz import WRatio

    columns = ["source_name", "code", "matched_name", "score", "method"]
    if not names:
        return pd.DataFrame(columns=columns)

    overrides = _load_overrides(overrides_path)
    candidates = index.drop_duplicates(subset="code")[["code", "norm_name", "full_name"]]
    lookup = candidates["norm_name"].tolist()

    rows = []
    for name in names:
        norm = normalise_name(name)

        if norm in overrides:
            rows.append(
                {
                    "source_name": name,
                    "code": overrides[norm],
                    "matched_name": None,
                    "score": 100.0,
                    "method": "override",
                }
            )
            continue

        best = process.extractOne(norm, lookup, scorer=WRatio)
        if best is None or best[1] < threshold:
            rows.append(
                {
                    "source_name": name,
                    "code": pd.NA,
                    "matched_name": None,
                    "score": float(best[1]) if best else 0.0,
                    "method": "unmatched",
                }
            )
            continue

        hit = candidates.iloc[best[2]]
        rows.append(
            {
                "source_name": name,
                "code": int(hit["code"]),
                "matched_name": hit["full_name"],
                "score": float(best[1]),
                "method": "fuzzy",
            }
        )

    result = pd.DataFrame(rows, columns=columns)
    unmatched = int((result["method"] == "unmatched").sum())
    if unmatched:
        log.warning(
            "%d of %d names unmatched at threshold %d; add them to %s",
            unmatched,
            len(names),
            threshold,
            overrides_path.name,
        )
    return result


def load_team_aliases(path: Path = TEAM_ALIASES_PATH) -> dict[str, str]:
    """Map every known spelling of a club to its FPL short name."""
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return dict(zip(frame["alias"].map(normalise_name), frame["fpl_name"], strict=False))


def canonical_team(name: str, aliases: dict[str, str] | None = None) -> str | None:
    """Resolve one club spelling. Returns ``None`` when unknown, never a guess."""
    aliases = load_team_aliases() if aliases is None else aliases
    return aliases.get(normalise_name(name))

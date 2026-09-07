"""Identity is the foundation. If these break, every downstream feature is wrong."""

import pandas as pd
import pytest

from fpl.entity import resolve


def _registry() -> pd.DataFrame:
    """Two seasons in which element id 7 is reused for a different footballer."""
    return pd.DataFrame(
        [
            {"season": "2023-24", "id": 7, "code": 1001, "first_name": "Gabriel",
             "second_name": "dos Santos Magalhães", "element_type": 2, "team": 1},
            {"season": "2023-24", "id": 9, "code": 1002, "first_name": "Son",
             "second_name": "Heung-min", "element_type": 3, "team": 2},
            {"season": "2024-25", "id": 7, "code": 1002, "first_name": "Heung-Min",
             "second_name": "Son", "element_type": 3, "team": 2},
            {"season": "2024-25", "id": 12, "code": 1003, "first_name": "Mikel",
             "second_name": "Arteta", "element_type": 5, "team": 1},
        ]
    )  # fmt: skip


# --- name normalisation -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Gabriel dos Santos Magalhães", "gabriel dos santos magalhaes"),
        ("Gabriel Dos Santos Magalhaes", "gabriel dos santos magalhaes"),
        ("  Kevin   De  Bruyne  ", "kevin de bruyne"),
        ("N'Golo Kanté", "n golo kante"),
        ("Nott'm Forest", "nott m forest"),
    ],
)
def test_normalise_name_folds_accents_and_punctuation(raw, expected):
    assert resolve.normalise_name(raw) == expected


def test_normalise_name_handles_non_strings():
    assert resolve.normalise_name(None) == ""
    assert resolve.normalise_name(float("nan")) == ""


# --- the element trap -------------------------------------------------------


def test_element_ids_are_reused_across_seasons():
    """The premise of the whole module: element 7 is two different people."""
    index = resolve.build_player_index(_registry())
    element_7 = index[index["element"] == 7]
    assert len(element_7) == 2
    assert element_7["code"].nunique() == 2, "element must not be treated as an identity"


def test_code_is_stable_across_seasons_despite_name_reordering():
    """Son appears as 'Son Heung-min' then 'Heung-Min Son'. Same code, both times."""
    index = resolve.build_player_index(_registry())
    son = index[index["code"] == 1002]
    assert son["season"].nunique() == 2
    assert son["full_name"].nunique() == 2, "fixture should exercise the rename"


def test_build_player_index_maps_positions():
    index = resolve.build_player_index(_registry())
    assert set(index["position"]) == {"DEF", "MID", "MGR"}


def test_build_player_index_rejects_unknown_position():
    registry = _registry()
    registry.loc[0, "element_type"] = 99
    with pytest.raises(ValueError, match="unknown element_type"):
        resolve.build_player_index(registry)


def test_build_player_index_rejects_duplicate_keys():
    registry = pd.concat([_registry(), _registry().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        resolve.build_player_index(registry)


# --- attaching codes to gameweeks -------------------------------------------


def _gameweeks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": "2023-24", "element": 7, "name": "Gabriel", "position": "DEF"},
            {"season": "2024-25", "element": 7, "name": "Son", "position": "MID"},
        ]
    )


def test_attach_player_code_resolves_reused_ids_correctly():
    resolved = resolve.attach_player_code(_gameweeks(), resolve.build_player_index(_registry()))
    by_season = dict(zip(resolved["season"], resolved["code"], strict=True))
    assert by_season["2023-24"] == 1001
    assert by_season["2024-25"] == 1002


def test_registry_position_overrides_the_gameweek_files_position():
    """The gameweek files label managers 'AM'; the registry says 'MGR'. The registry
    wins, otherwise drop_managers silently matches nothing."""
    gameweeks = pd.DataFrame(
        [{"season": "2024-25", "element": 12, "name": "Arteta", "position": "AM"}]
    )
    resolved = resolve.attach_player_code(gameweeks, resolve.build_player_index(_registry()))
    assert resolved.loc[0, "position"] == "MGR"
    assert "registry_position" not in resolved.columns


def test_attach_player_code_is_fatal_on_unresolved_rows():
    """A partial join would corrupt every rolling feature. It must not be survivable."""
    orphan = pd.DataFrame([{"season": "2023-24", "element": 999, "name": "Nobody"}])
    with pytest.raises(ValueError, match="no registry entry"):
        resolve.attach_player_code(orphan, resolve.build_player_index(_registry()))


def test_drop_managers_removes_only_managers():
    index = resolve.build_player_index(_registry())
    frame = pd.DataFrame(
        [
            {"season": "2024-25", "element": 12, "name": "Arteta", "position": "AM"},
            {"season": "2024-25", "element": 7, "name": "Son", "position": "MID"},
        ]
    )
    kept = resolve.drop_managers(resolve.attach_player_code(frame, index))
    assert list(kept["position"]) == ["MID"]


def test_drop_managers_rejects_stray_positions():
    """Anything that is not a playing position after managers are gone is a data bug."""
    frame = pd.DataFrame([{"position": "AM"}, {"position": "MID"}])
    with pytest.raises(ValueError, match="unexpected positions"):
        resolve.drop_managers(frame)


def test_drop_managers_requires_resolution_first():
    with pytest.raises(ValueError, match="attach_player_code"):
        resolve.drop_managers(pd.DataFrame([{"name": "x"}]))


# --- external name matching -------------------------------------------------


def test_match_external_names_finds_close_matches():
    index = resolve.build_player_index(_registry())
    result = resolve.match_external_names(["Gabriel Dos Santos Magalhaes"], index)
    assert result.loc[0, "code"] == 1001
    assert result.loc[0, "method"] == "fuzzy"


def test_match_external_names_refuses_to_guess():
    """Unknown names come back unmatched, never attached to the nearest stranger."""
    index = resolve.build_player_index(_registry())
    result = resolve.match_external_names(["Cristiano Ronaldo"], index)
    assert pd.isna(result.loc[0, "code"])
    assert result.loc[0, "method"] == "unmatched"


def test_match_external_names_handles_empty_input():
    result = resolve.match_external_names([], resolve.build_player_index(_registry()))
    assert result.empty
    assert list(result.columns) == ["source_name", "code", "matched_name", "score", "method"]


# --- teams ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("Manchester United", "Man Utd"),
        ("Man United", "Man Utd"),
        ("MAN UTD", "Man Utd"),
        ("Nottingham Forest", "Nott'm Forest"),
        ("Tottenham", "Spurs"),
        ("Wolverhampton Wanderers", "Wolves"),
    ],
)
def test_canonical_team_resolves_known_spellings(spelling, expected):
    assert resolve.canonical_team(spelling) == expected


def test_canonical_team_returns_none_for_unknown():
    assert resolve.canonical_team("Real Madrid") is None


def test_every_fpl_spelling_maps_to_itself():
    """Each club's own FPL name must round-trip, or joins to FPL data will fail."""
    aliases = resolve.load_team_aliases()
    for name in set(aliases.values()):
        assert resolve.canonical_team(name, aliases) == name, f"{name} does not round-trip"

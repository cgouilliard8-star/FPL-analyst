"""The scoring constants are the model's contract with the game. Guard them."""

from fpl import config


def test_squad_quota_sums_to_squad_size():
    assert sum(config.SQUAD_QUOTA.values()) == config.SQUAD_SIZE


def test_xi_bounds_are_satisfiable():
    assert sum(config.XI_MIN.values()) <= config.XI_SIZE <= sum(config.XI_MAX.values())


def test_every_position_has_scoring_rules():
    for position in config.POSITIONS:
        assert position in config.GOAL_POINTS
        assert position in config.CLEAN_SHEET_POINTS
        assert position in config.DC_THRESHOLD


def test_clean_sheet_points_match_fpl_rules():
    assert config.CLEAN_SHEET_POINTS == {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}


def test_squad_cannot_be_bought_from_one_club():
    # 15 players, max 3 per club, so at least 5 clubs must be represented.
    assert config.SQUAD_SIZE / config.MAX_PER_CLUB >= 5

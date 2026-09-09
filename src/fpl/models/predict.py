"""Project the next gameweek -- and the run of gameweeks after it -- from a snapshot.

The archive supplies the seasons that are over; the snapshot supplies this season's
finished fixtures and the fixtures still to come. Everything is stacked into one
table, features are built exactly as in the backtest, the component model is fitted
on every row whose outcome is known, and the rows whose outcome is not yet known are
what it predicts.

Horizon
-------
Only the next gameweek can be projected from a clean feature row: a rolling window
for the gameweek after it would have to include a match that has not been played.
So the player's form features are frozen at today and, for each later gameweek,
only the fixture context changes -- who he plays, where, how many times, and how
strong that opponent's attack and defence have been. That is the part of a
multi-week projection that is actually knowable in advance, and it is what makes a
defender with three soft fixtures worth more than one with three hard ones.

Availability
------------
FPL's flag is applied after the model, as a multiplier. For the next gameweek it is
used as given. For later gameweeks the news text is read for a return date
("Expected back 18 Sep") so a player is zeroed only for the fixtures before it; a
loan or permanent departure is zeroed throughout; an unspecified doubt clears.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fpl.config import (
    CALIBRATION,
    CURRENT_SEASON,
    FPL_BLEND,
    HORIZON_WEIGHTS,
    MARKET_WEIGHT,
    MAX_HORIZON,
    TRAIN_SEASONS,
)
from fpl.data.archive import load_players
from fpl.data.fpl_api import next_gameweek, snapshot_to_gameweeks, snapshot_to_upcoming
from fpl.data.odds import ODDS_FEATURES, load_odds
from fpl.data.schedule import load_schedule
from fpl.data.silver import load_silver
from fpl.entity.resolve import canonical_team, load_team_aliases
from fpl.features.aggregate import attach_opponent, build_team_id_map
from fpl.features.build import attach_congestion, build_features, feature_columns
from fpl.models.combine import CONTRIBUTIONS, fit_component_model
from fpl.models.team_strength import FEATURES as TS_FEATURES
from fpl.models.team_strength import Ratings, latest_ratings

log = logging.getLogger(__name__)

# Nearer gameweeks matter more: a transfer can be undone next week, and the fixture
# context is better known. Index 0 is the next gameweek.
CONGESTION_COLUMNS = (
    "other_games_7d", "euro_midweek", "other_game_next_4d", "days_since_any_match",
    "opp_other_games_7d", "opp_euro_midweek", "days_rest_all",
)  # fmt: skip


TEAM_METRICS = (
    "team_goals_r5", "team_xg_r5", "team_conceded_r5", "team_xgc_r5",
    "team_goals_r10", "team_xg_r10", "team_conceded_r10", "team_xgc_r10",
    "team_goals_r38", "team_xg_r38", "team_conceded_r38", "team_xgc_r38",
    "team_goals_ew", "team_xg_ew", "team_conceded_ew", "team_xgc_ew",
)  # fmt: skip

TS_SUMMED = ("ts_xg_for", "ts_xg_against", "ts_cs", "ts_win")

_RETURN = re.compile(r"(?:expected back|suspended until|until)\s+(\d{1,2})\s+([A-Za-z]{3})", re.I)
_GONE = re.compile(r"joined .* (?:on loan|permanently)|left the club|has left|released", re.I)


def _snapshot_registry(snapshot: dict, season: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "id": e["id"],
                "code": e["code"],
                "first_name": e["first_name"],
                "second_name": e["second_name"],
                "element_type": e["element_type"],
                "team": e["team"],
                # set-piece duties, today's state: exactly what the feature wants
                "penalties_order": e.get("penalties_order"),
                "corners_and_indirect_freekicks_order": e.get(
                    "corners_and_indirect_freekicks_order"
                ),
                "direct_freekicks_order": e.get("direct_freekicks_order"),
            }
            for e in snapshot["elements"]
        ]
    )


def _canonicalise(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = load_team_aliases()
    frame = frame.copy()
    if frame.empty:
        return frame
    frame["team"] = frame["team"].map(lambda n: canonical_team(n, aliases))
    unknown = frame["team"].isna()
    if unknown.any():
        raise ValueError(
            f"unrecognised clubs in live data: {sorted(frame.loc[unknown, 'team'].unique())}"
        )
    frame["full_name"] = frame["name"]
    return frame[frame["position"].isin(["GK", "DEF", "MID", "FWD"])]


def return_date(news: str, reference: datetime) -> datetime | None:
    """Parse "Expected back 18 Sep" style news into a date, or None."""
    match = _RETURN.search(news or "")
    if not match:
        return None
    day, month = int(match.group(1)), match.group(2).title()
    try:
        parsed = datetime.strptime(f"{day} {month} {reference.year}", "%d %b %Y")
    except ValueError:
        return None
    parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed < reference - pd.Timedelta(days=60):  # a January return read in September
        parsed = parsed.replace(year=reference.year + 1)
    return parsed


def availability_for(row: pd.Series, kickoff: pd.Timestamp, reference: datetime) -> float:
    """Availability for a specific future fixture, given the flag and its news."""
    status, chance, news = row["status"], row["chance_of_playing"], row["news"] or ""
    first = float(row["availability"])
    if kickoff <= reference + pd.Timedelta(days=8):
        return first  # the next gameweek: take FPL's word for it
    if _GONE.search(news):
        return 0.0
    back = return_date(news, reference)
    if back is not None:
        return 1.0 if kickoff >= back else 0.0
    if status == "d":
        return 1.0  # an unspecified doubt clears
    if status in ("i", "s", "u", "n") and (chance is None or pd.isna(chance) or chance == 0):
        return 0.0  # out with no return date: assume out
    return first


def assemble(
    snapshot: dict, season: str = CURRENT_SEASON, *, as_of_gameweek: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Stack archive, live history and the first projected gameweek's fixtures.

    Returns the combined per-fixture table, the combined registry, and the first
    gameweek being projected.
    """
    first = as_of_gameweek or next_gameweek(snapshot)["id"]
    archive = load_silver()
    history = _canonicalise(snapshot_to_gameweeks(snapshot, season, before_gameweek=first))
    upcoming = _canonicalise(
        snapshot_to_upcoming(snapshot, season, gameweeks=[first], as_of_gameweek=as_of_gameweek)
    )
    if not history.empty:
        history["is_upcoming"] = 0
    upcoming["is_upcoming"] = 1
    archive["is_upcoming"] = 0

    parts = [archive, history, upcoming] if not history.empty else [archive, upcoming]
    combined = pd.concat(parts, ignore_index=True, sort=False)
    registry = pd.concat(
        [load_players(TRAIN_SEASONS), _snapshot_registry(snapshot, season)], ignore_index=True
    )
    log.info(
        "GW%s: assembled %d archive + %d live + %d upcoming fixture rows",
        first, len(archive), len(history), len(upcoming),
    )  # fmt: skip
    return combined, registry, first


def _team_strength_now(
    features: pd.DataFrame, season: str, gameweek: int, ratings: Ratings | None = None
) -> pd.DataFrame:
    """Each club's attack/defence numbers as of the first projected gameweek.

    The rolling windows are kept for the fixture context (they are what the model
    was trained on). The *ranks* -- what the page shows as fixture toughness -- come
    from the opponent-adjusted ratings: expected goals for and against an average
    opponent at a neutral venue, with every match weighted down as it ages
    (half-life 100 days). Ranking on a two-match form window instead put a club
    with two lucky clean sheets third in the league for defence; the ratings do not
    forget a season of evidence that quickly, and they credit a clean sheet against
    the champions more than one against a promoted side.
    """
    rows = features[(features["season"] == season) & (features["GW"] == gameweek)]
    own = [f"own_{m}" for m in TEAM_METRICS]
    table = rows.groupby("team")[own].first()
    table.columns = list(TEAM_METRICS)
    # A club with no recent history (promoted, first gameweek) is treated as average
    # rather than dropped: it still has to be ranked and faced.
    table = table.fillna(table.mean()).fillna(0.0)
    if ratings is not None and ratings.att:
        neutral = ratings.intercept + ratings.home / 2.0
        att = pd.Series(
            [math.exp(neutral + ratings.att.get(t, 0.0)) for t in table.index], index=table.index
        )
        dfn = pd.Series(
            [math.exp(neutral - ratings.dfn.get(t, 0.0)) for t in table.index], index=table.index
        )
    else:  # no ratings (tests, tiny fixtures): fall back to recent form
        att = 0.7 * table["team_xg_ew"] + 0.3 * table["team_xg_r38"]
        dfn = 0.7 * table["team_xgc_ew"] + 0.3 * table["team_xgc_r38"]
    table["att_score"] = att
    table["def_score"] = dfn
    table["att_rank"] = att.rank(ascending=False, method="min").astype(int)
    table["def_rank"] = dfn.rank(ascending=True, method="min").astype(int)
    return table


def _future_rows(
    base: pd.DataFrame,
    snapshot: dict,
    first: int,
    horizon: int,
    strength: pd.DataFrame,
    league_xgc: float,
    odds: pd.DataFrame | None = None,
    ratings: Ratings | None = None,
) -> pd.DataFrame:
    """Clone each player's frozen feature row for each later gameweek, swapping in
    that gameweek's fixture context (opponent strength, congestion, and the market's
    prices for that match where they are already quoted)."""
    market: dict[tuple[str, str, str], dict] = {}
    if odds is not None and not odds.empty:
        for r in odds.itertuples(index=False):
            market[(r.home, r.away, r.team)] = {f: getattr(r, f) for f in ODDS_FEATURES}
    aliases = load_team_aliases()
    names = {t["id"]: canonical_team(t["name"], aliases) for t in snapshot["teams"]}
    fixtures = [
        f for f in snapshot["fixtures"] if f["event"] and first < f["event"] < first + horizon
    ]
    by_team: dict[str, dict[int, list[dict]]] = {}
    for f in fixtures:
        for tid, opp, home in ((f["team_h"], f["team_a"], True), (f["team_a"], f["team_h"], False)):
            by_team.setdefault(names[tid], {}).setdefault(f["event"], []).append(
                {"opponent": names[opp], "home": home, "kickoff": pd.Timestamp(f["kickoff_time"])}
            )

    rows = []
    for _, row in base.iterrows():
        last_kick = row["kickoff_time"]
        for gw in range(first + 1, first + horizon):
            games = by_team.get(row["team"], {}).get(gw, [])
            if not games:
                continue  # blank gameweek: nothing to project
            g = games[0]
            clone = row.copy()
            clone["GW"] = gw
            clone["opponent"] = g["opponent"]
            clone["is_home"] = int(g["home"])
            clone["was_home"] = g["home"]
            clone["fixtures_this_gw"] = len(games)
            clone["kickoff_time"] = g["kickoff"]
            clone["days_rest"] = min(60.0, (g["kickoff"] - last_kick).total_seconds() / 86400.0)
            last_kick = g["kickoff"]
            opp = strength.loc[g["opponent"]] if g["opponent"] in strength.index else None
            for m in TEAM_METRICS:
                clone[f"opp_{m}"] = float(opp[m]) if opp is not None else np.nan
            clone["fixture_ease"] = (
                (clone["opp_team_xgc_r10"] / league_xgc) if league_xgc else np.nan
            )
            home_name = row["team"] if g["home"] else g["opponent"]
            away_name = g["opponent"] if g["home"] else row["team"]
            quoted = market.get((home_name, away_name, row["team"]))
            for f in ODDS_FEATURES:
                clone[f] = quoted[f] if quoted else np.nan
            if ratings is not None:
                # As in the training table: the expected-goal terms add up across a
                # double gameweek, the ratings themselves are the club's.
                per_game = [ratings.features(row["team"], x["opponent"], x["home"]) for x in games]
                for f in TS_FEATURES:
                    values = [g[f] for g in per_game]
                    clone[f] = sum(values) if f in TS_SUMMED else values[0]
            clone["all_opponents"] = " + ".join(
                f"{x['opponent']} ({'H' if x['home'] else 'A'})" for x in games
            )
            rows.append(clone)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=base.columns)


UNKNOWN_MINUTES = 90  # fewer league minutes on record than this: no evidence of our own
UNKNOWN_CAP = 1.25  # ...so defer to FPL's own number, allowing it this much upside


_MARKET_PAIRS = (
    ("ts_xg_for", "odds_xg"),
    ("ts_xg_against", "odds_xgc"),
    ("ts_cs", "odds_cs"),
    ("ts_win", "odds_win"),
)


def _fold_market(rows: pd.DataFrame, weight: float = MARKET_WEIGHT) -> pd.DataFrame:
    """Average the market's view of a coming fixture into the club-rating features.

    Both are estimates of the same thing -- how many goals each side should score
    -- and the bookmakers' one also knows the team news. The model was trained on
    the ratings, so the blend stays on the same scale; where no price exists the
    ratings stand alone.
    """
    if weight <= 0 or "odds_xg" not in rows.columns:
        return rows
    out = rows.copy()
    priced = pd.to_numeric(out["odds_xg"], errors="coerce").notna()
    if not priced.any():
        return out
    for rating, market in _MARKET_PAIRS:
        if rating in out.columns and market in out.columns:
            r = pd.to_numeric(out[rating], errors="coerce")
            m = pd.to_numeric(out[market], errors="coerce")
            out[rating] = np.where(priced & m.notna() & r.notna(), (1 - weight) * r + weight * m, r)
    log.info("market prices folded into the club ratings for %d fixture rows", int(priced.sum()))
    return out


def _calibrate(breakdown: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """Scale each position's projections by its measured calibration factor."""
    factor = positions.map(CALIBRATION).fillna(1.0).to_numpy()
    if np.allclose(factor, 1.0):
        return breakdown
    out = breakdown.copy()
    for column in (*CONTRIBUTIONS, "expected_points"):
        out[column] = out[column].to_numpy() * factor
    return out


def _cap_unknowns(breakdown: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Rein in projections for players the data has never really seen.

    A summer signing with twenty league minutes has price, ownership and club
    strength but no record, and the trees extrapolate from whoever else looked like
    that -- sometimes wildly. FPL's own expected points know the pre-season and the
    press conference, so for such players every component is scaled so the total
    does not exceed FPL's figure by more than ``UNKNOWN_CAP``. Past deadlines
    (backtests) have no FPL figure and are left alone.
    """
    minutes = rows["minutes_todate"].fillna(0.0).to_numpy()
    fpl = rows.get("fpl_ep_next")
    if fpl is None:
        return breakdown
    fpl = pd.to_numeric(fpl, errors="coerce").fillna(0.0).to_numpy()
    total = breakdown["expected_points"].to_numpy()
    unknown = (minutes < UNKNOWN_MINUTES) & (fpl > 0) & (total > UNKNOWN_CAP * fpl)
    if not unknown.any():
        return breakdown
    scale = np.where(unknown, UNKNOWN_CAP * fpl / np.where(total > 0, total, 1.0), 1.0)
    out = breakdown.copy()
    for column in (*CONTRIBUTIONS, "expected_points"):
        out[column] = out[column].to_numpy() * scale
    log.info(
        "capped %d projections for players with under %d minutes on record",
        int(unknown.sum()),
        UNKNOWN_MINUTES,
    )
    return out


def _blend_fpl(fixtures: pd.DataFrame, every: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Fold FPL's own next-gameweek figure into the next gameweek's projection.

    Applied only where FPL has a figure and the player is unflagged: FPL's number
    already discounts a doubt in its own way, and stacking two discounts would
    punish a flagged player twice. The decomposition is scaled with the total so
    the "why" numbers still add up.
    """
    out = fixtures.copy()
    fpl = pd.to_numeric(every.get("fpl_ep_next"), errors="coerce").fillna(0.0).to_numpy()
    model = out["expected_points"].to_numpy()
    eligible = (out["offset"].to_numpy() == 0) & (out["availability"].to_numpy() >= 1.0) & (fpl > 0)
    blended = np.where(eligible, (1.0 - weight) * model + weight * fpl, model)
    scale = np.where(model > 0, blended / np.where(model > 0, model, 1.0), 1.0)
    for term in CONTRIBUTIONS:
        out[term] = out[term].to_numpy() * scale
    out["expected_points"] = blended
    log.info("blended FPL's figure into %d next-gameweek projections", int(eligible.sum()))
    return out


def project_horizon(
    snapshot: dict,
    season: str = CURRENT_SEASON,
    *,
    horizon: int = MAX_HORIZON,
    as_of_gameweek: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Project every player for the next ``horizon`` gameweeks.

    Returns ``(players, fixtures, teams)``:

    * ``players`` -- one row per player: identity, price, availability, the next
      gameweek's projection and decomposition, and ``ep1``/``ep3``/``ep5`` horizon
      totals (weighted by ``HORIZON_WEIGHTS``).
    * ``fixtures`` -- one row per player per projected gameweek with opponent,
      venue, projection and decomposition.
    * ``teams`` -- each club's attack/defence numbers and ranks as of now.
    """
    horizon = max(1, min(horizon, MAX_HORIZON))
    combined, registry, first = assemble(snapshot, season, as_of_gameweek=as_of_gameweek)
    seasons = (*TRAIN_SEASONS, season)
    features = build_features(seasons, write=False, silver=combined, players=registry)
    columns = feature_columns(features)

    is_target = (features["season"] == season) & (features["GW"] == first)
    is_upcoming = features["is_upcoming"].fillna(0).astype(int) == 1
    cutoff = features.loc[is_target & is_upcoming, "kickoff_time"].min()
    train = features[~is_upcoming & (features["kickoff_time"] < cutoff)]
    base = features[is_target & is_upcoming].copy()
    if base.empty:
        raise ValueError(f"no upcoming rows for {season} GW{first}")
    base["all_opponents"] = base["opponent"] + " (" + np.where(base["is_home"] == 1, "H", "A") + ")"

    log.info(
        "GW%s: fitting on %d rows, projecting %d players x %d gameweeks",
        first, len(train), len(base), horizon,
    )  # fmt: skip
    model = fit_component_model(train, columns)

    # Club ratings as of the deadline: the fixture-toughness ranks, and the fixture
    # context for the gameweeks beyond the next one.
    in_scope = combined[combined["season"].isin(seasons)]
    with_opponent = attach_opponent(in_scope, build_team_id_map(registry, in_scope))
    ratings = latest_ratings(with_opponent, pd.Timestamp(cutoff))
    strength = _team_strength_now(features, season, first, ratings)
    league_xgc = float(features.loc[is_target, "opp_team_xgc_r10"].mean())
    future = _future_rows(
        base, snapshot, first, horizon, strength, league_xgc, load_odds((season,)), ratings
    )
    if not future.empty:
        # The clones carry the first gameweek's cup/European context; recompute it
        # for each later kick-off from the same schedule the features were built on.
        drop = [c for c in future.columns if c in CONGESTION_COLUMNS]
        future = attach_congestion(future.drop(columns=drop), load_schedule((season,)))
    every = (
        pd.concat([base, future], ignore_index=True)
        if not future.empty
        else base.reset_index(drop=True)
    )

    every = _fold_market(every.reset_index(drop=True))
    breakdown = model.explain(every).reset_index(drop=True)
    breakdown = _calibrate(breakdown, every["position"])
    breakdown = _cap_unknowns(breakdown, every)

    reference = (
        datetime.now(timezone.utc)
        if as_of_gameweek is None
        else pd.Timestamp(cutoff).to_pydatetime()
    )
    avail = np.array(
        [availability_for(r, r["kickoff_time"], reference) for _, r in every.iterrows()]
    )

    fixtures = every[
        [
            "code",
            "web_name",
            "position",
            "team",
            "GW",
            "opponent",
            "all_opponents",
            "is_home",
            "fixtures_this_gw",
            "kickoff_time",
            "other_games_7d",
            "euro_midweek",
            "other_game_next_4d",
        ]  # fmt: skip
    ].copy()
    fixtures["raw_expected_points"] = breakdown["expected_points"].to_numpy()
    for term in CONTRIBUTIONS:
        fixtures[term] = breakdown[term].to_numpy() * avail
    fixtures["expected_points"] = fixtures[list(CONTRIBUTIONS)].sum(axis=1)
    fixtures["model_expected_points"] = fixtures["expected_points"]
    fixtures["availability"] = avail
    fixtures["p_60"] = np.clip(breakdown["p_60"].to_numpy() * avail, 0, 1)
    # The armband: the chance of a haul is judged per gameweek, never over a run,
    # because the captain is chosen again every week.
    fixtures["p_haul"] = np.clip(breakdown["p_haul"].to_numpy() * avail, 0, 1)
    fixtures["opp_att_rank"] = fixtures["opponent"].map(strength["att_rank"]).astype("Int64")
    fixtures["opp_def_rank"] = fixtures["opponent"].map(strength["def_rank"]).astype("Int64")
    fixtures["opp_xg_r5"] = fixtures["opponent"].map(strength["team_xg_r5"])
    fixtures["opp_xgc_r5"] = fixtures["opponent"].map(strength["team_xgc_r5"])
    fixtures["offset"] = fixtures["GW"] - first
    fixtures["weight"] = fixtures["offset"].map(
        lambda k: HORIZON_WEIGHTS[int(k)] if 0 <= k < MAX_HORIZON else 0.0
    )
    if as_of_gameweek is None and FPL_BLEND > 0:
        fixtures = _blend_fpl(fixtures, every, FPL_BLEND)
    fixtures = fixtures.sort_values(["code", "GW"]).reset_index(drop=True)

    # per-player summary
    nxt = fixtures[fixtures["offset"] == 0].set_index("code")
    ident = base.set_index("code")
    players = pd.DataFrame(index=ident.index)
    for col in (
        "element", "web_name", "full_name", "position", "team", "price", "selected",
        "selected_by_percent", "status", "chance_of_playing", "news", "availability",
        "fpl_ep_next", "minutes_share_r5", "minutes_todate", "penalties_order",
        "corners_order", "freekicks_order", "transfers_in_event", "transfers_out_event",
        "cost_change_start", "value_season",
    ):  # fmt: skip
        players[col] = ident[col] if col in ident.columns else None
    players["opponent"] = nxt["all_opponents"]
    players["is_home"] = nxt["is_home"]
    players["fixtures_this_gw"] = nxt["fixtures_this_gw"]
    players["raw_expected_points"] = nxt["raw_expected_points"]
    players["model_expected_points"] = nxt["model_expected_points"]
    for term in CONTRIBUTIONS:
        players[term] = nxt[term]
    players["expected_points"] = nxt["expected_points"]
    players["p_60"] = nxt["p_60"]
    players["p_haul"] = nxt["p_haul"]
    weighted = fixtures.assign(w=fixtures["expected_points"] * fixtures["weight"])
    for h in (1, 3, 5):
        if h <= horizon:
            players[f"ep{h}"] = (
                weighted[weighted["offset"] < h].groupby("code")["w"].sum()
                .reindex(players.index).fillna(0.0)
            )  # fmt: skip
    players["own_att_rank"] = players["team"].map(strength["att_rank"]).astype("Int64")
    players["own_def_rank"] = players["team"].map(strength["def_rank"]).astype("Int64")
    players["gameweek"] = first
    players["deadline"] = (
        next_gameweek(snapshot)["deadline_time"] if as_of_gameweek is None else None
    )
    players = (
        players.reset_index().sort_values("expected_points", ascending=False).reset_index(drop=True)
    )

    teams = strength.reset_index().rename(columns={"index": "team"})
    log.info("GW%s: %d players projected over %d gameweeks", first, len(players), horizon)
    return players, fixtures, teams


def project_gameweek(snapshot: dict, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """The next gameweek only, in the flat shape the earlier callers expect."""
    players, _, _ = project_horizon(snapshot, season, horizon=1)
    return players

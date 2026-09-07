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
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fpl.config import CURRENT_SEASON, TRAIN_SEASONS
from fpl.data.archive import load_players
from fpl.data.fpl_api import next_gameweek, snapshot_to_gameweeks, snapshot_to_upcoming
from fpl.data.silver import load_silver
from fpl.entity.resolve import canonical_team, load_team_aliases
from fpl.features.build import build_features, feature_columns
from fpl.models.combine import CONTRIBUTIONS, fit_component_model

log = logging.getLogger(__name__)

# Nearer gameweeks matter more: a transfer can be undone next week, and the fixture
# context is better known. Index 0 is the next gameweek.
HORIZON_WEIGHTS: tuple[float, ...] = (1.0, 0.85, 0.7, 0.55, 0.4)
MAX_HORIZON = len(HORIZON_WEIGHTS)

TEAM_METRICS = (
    "team_goals_r5", "team_xg_r5", "team_conceded_r5", "team_xgc_r5",
    "team_goals_r10", "team_xg_r10", "team_conceded_r10", "team_xgc_r10",
)  # fmt: skip

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


def _team_strength_now(features: pd.DataFrame, season: str, gameweek: int) -> pd.DataFrame:
    """Each club's lagged attack/defence numbers as of the first projected gameweek."""
    rows = features[(features["season"] == season) & (features["GW"] == gameweek)]
    own = [f"own_{m}" for m in TEAM_METRICS]
    table = rows.groupby("team")[own].first()
    table.columns = list(TEAM_METRICS)
    # A club with no recent history (promoted, first gameweek) is treated as average
    # rather than dropped: it still has to be ranked and faced.
    table = table.fillna(table.mean()).fillna(0.0)
    table["att_rank"] = table["team_xg_r5"].rank(ascending=False, method="min").astype(int)
    table["def_rank"] = table["team_xgc_r5"].rank(ascending=True, method="min").astype(int)
    return table


def _future_rows(
    base: pd.DataFrame,
    snapshot: dict,
    first: int,
    horizon: int,
    strength: pd.DataFrame,
    league_xgc: float,
) -> pd.DataFrame:
    """Clone each player's frozen feature row for each later gameweek, swapping in
    that gameweek's fixture context."""
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
            clone["all_opponents"] = " + ".join(
                f"{x['opponent']} ({'H' if x['home'] else 'A'})" for x in games
            )
            rows.append(clone)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=base.columns)


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

    strength = _team_strength_now(features, season, first)
    league_xgc = float(features.loc[is_target, "opp_team_xgc_r10"].mean())
    future = _future_rows(base, snapshot, first, horizon, strength, league_xgc)
    every = (
        pd.concat([base, future], ignore_index=True)
        if not future.empty
        else base.reset_index(drop=True)
    )

    breakdown = model.explain(every).reset_index(drop=True)
    every = every.reset_index(drop=True)

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
        ]  # fmt: skip
    ].copy()
    fixtures["raw_expected_points"] = breakdown["expected_points"].to_numpy()
    for term in CONTRIBUTIONS:
        fixtures[term] = breakdown[term].to_numpy() * avail
    fixtures["expected_points"] = fixtures[list(CONTRIBUTIONS)].sum(axis=1)
    fixtures["availability"] = avail
    fixtures["p_60"] = np.clip(breakdown["p_60"].to_numpy() * avail, 0, 1)
    fixtures["opp_att_rank"] = fixtures["opponent"].map(strength["att_rank"]).astype("Int64")
    fixtures["opp_def_rank"] = fixtures["opponent"].map(strength["def_rank"]).astype("Int64")
    fixtures["opp_xg_r5"] = fixtures["opponent"].map(strength["team_xg_r5"])
    fixtures["opp_xgc_r5"] = fixtures["opponent"].map(strength["team_xgc_r5"])
    fixtures["offset"] = fixtures["GW"] - first
    fixtures["weight"] = fixtures["offset"].map(
        lambda k: HORIZON_WEIGHTS[int(k)] if 0 <= k < MAX_HORIZON else 0.0
    )
    fixtures = fixtures.sort_values(["code", "GW"]).reset_index(drop=True)

    # per-player summary
    nxt = fixtures[fixtures["offset"] == 0].set_index("code")
    ident = base.set_index("code")
    players = pd.DataFrame(index=ident.index)
    for col in (
        "element", "web_name", "full_name", "position", "team", "price", "selected",
        "selected_by_percent", "status", "chance_of_playing", "news", "availability",
        "fpl_ep_next", "minutes_share_r5",
    ):  # fmt: skip
        players[col] = ident[col]
    players["opponent"] = nxt["all_opponents"]
    players["is_home"] = nxt["is_home"]
    players["fixtures_this_gw"] = nxt["fixtures_this_gw"]
    players["raw_expected_points"] = nxt["raw_expected_points"]
    for term in CONTRIBUTIONS:
        players[term] = nxt[term]
    players["expected_points"] = nxt["expected_points"]
    players["p_60"] = nxt["p_60"]
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

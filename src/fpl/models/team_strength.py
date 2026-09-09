"""Club attack and defence ratings, refitted before every gameweek.

The rolling windows in ``features.windows`` say how a club *has* scored and conceded.
This says how it *should*, against a given opponent at a given venue: a Poisson
model in the Dixon-Coles tradition, where the log of a side's expected goals is an
attack rating minus the opponent's defence rating plus a home advantage. Every
match a club has played contributes, weighted down as it ages, so the rating is
today's quality rather than a window's average -- and it is opponent-adjusted,
which the rolling numbers are not (scoring three against a promoted side and three
against the champions look the same to a rolling mean).

The fitted target is a blend of goals and expected goals: xG carries the
repeatable part of a performance and goals the part xG misses (finishing, set
pieces), and the blend backtests better than either alone in the football-
analytics literature. Nothing here is external: results and xG come from the same
silver table as everything else, so a source outage cannot take it away.

Every rating is fitted only on matches that kicked off before the gameweek it is
used for. The leakage test rewrites the future and asserts these do not move.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HALFLIFE_DAYS = 100.0  # a match half a season old counts half
GOAL_WEIGHT = 0.35  # target = GOAL_WEIGHT * goals + (1 - GOAL_WEIGHT) * xG
RIDGE = 0.5  # shrinkage toward league average; keeps a promoted club near the mean
MAX_GOALS = 10  # grid size for the win/draw probabilities

FEATURES = (
    "ts_xg_for",
    "ts_xg_against",
    "ts_cs",
    "ts_win",
    "ts_att",
    "ts_def",
    "ts_opp_att",
    "ts_opp_def",
)


def team_matches(silver: pd.DataFrame) -> pd.DataFrame:
    """One row per club per played fixture: goals and xG for and against."""
    played = silver[silver["minutes"].notna()]
    if "team_h_score" not in played.columns:
        return pd.DataFrame(
            columns=["season", "fixture", "team", "opponent", "kickoff_time", "home",
                     "goals_for", "goals_against", "xg_for", "xg_against"]
        )  # fmt: skip
    played = played[played["team_h_score"].notna() & played["team_a_score"].notna()]
    rows = played.groupby(["season", "fixture", "team"], sort=False).agg(
        opponent=("opponent", "first"),
        kickoff_time=("kickoff_time", "min"),
        home=("was_home", "first"),
        team_h_score=("team_h_score", "first"),
        team_a_score=("team_a_score", "first"),
        xg_for=("expected_goals", "sum"),
        xg_against=("expected_goals_conceded", "max"),
    ).reset_index()  # fmt: skip
    home = rows["home"].astype(bool)
    rows["goals_for"] = np.where(home, rows["team_h_score"], rows["team_a_score"]).astype(float)
    rows["goals_against"] = np.where(home, rows["team_a_score"], rows["team_h_score"]).astype(float)
    rows["home"] = home.astype(int)
    # Seasons before xG was published fall back to goals alone.
    rows["xg_for"] = rows["xg_for"].where(rows["xg_for"].notna(), rows["goals_for"])
    rows["xg_against"] = rows["xg_against"].where(rows["xg_against"].notna(), rows["goals_against"])
    return rows.drop(columns=["team_h_score", "team_a_score"])


class Ratings:
    """Attack/defence ratings and home advantage as of one moment."""

    def __init__(self, intercept: float, home: float, att: dict[str, float], dfn: dict[str, float]):
        self.intercept, self.home, self.att, self.dfn = intercept, home, att, dfn

    def expected_goals(self, team: str, opponent: str, home: bool) -> tuple[float, float]:
        lam_for = math.exp(
            self.intercept
            + (self.home if home else 0.0)
            + self.att.get(team, 0.0)
            - self.dfn.get(opponent, 0.0)
        )
        lam_against = math.exp(
            self.intercept
            + (0.0 if home else self.home)
            + self.att.get(opponent, 0.0)
            - self.dfn.get(team, 0.0)
        )
        return lam_for, lam_against

    def features(self, team: str, opponent: str, home: bool) -> dict[str, float]:
        lam_for, lam_against = self.expected_goals(team, opponent, home)
        return {
            "ts_xg_for": lam_for,
            "ts_xg_against": lam_against,
            "ts_cs": math.exp(-lam_against),
            "ts_win": win_probability(lam_for, lam_against),
            "ts_att": self.att.get(team, 0.0),
            "ts_def": self.dfn.get(team, 0.0),
            "ts_opp_att": self.att.get(opponent, 0.0),
            "ts_opp_def": self.dfn.get(opponent, 0.0),
        }

    def table(self) -> pd.DataFrame:
        teams = sorted(set(self.att) | set(self.dfn))
        return pd.DataFrame(
            {"team": teams, "ts_att": [self.att.get(t, 0.0) for t in teams],
             "ts_def": [self.dfn.get(t, 0.0) for t in teams]}
        )  # fmt: skip


def win_probability(lam_for: float, lam_against: float, cap: int = MAX_GOALS) -> float:
    pf = [math.exp(-lam_for) * lam_for**i / math.factorial(i) for i in range(cap)]
    pa = [math.exp(-lam_against) * lam_against**j / math.factorial(j) for j in range(cap)]
    return float(sum(pf[i] * pa[j] for i in range(cap) for j in range(i)))


def fit_ratings(matches: pd.DataFrame, as_of: pd.Timestamp) -> Ratings:
    """Fit on every match that kicked off before ``as_of``, weighted by recency."""
    past = matches[matches["kickoff_time"] < as_of]
    teams = sorted(set(past["team"]) | set(past["opponent"]))
    if past.empty or len(teams) < 2:
        return Ratings(math.log(1.35), 0.2, {}, {})
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    age_days = (as_of - past["kickoff_time"]).dt.total_seconds().to_numpy() / 86400.0
    weight = np.power(0.5, age_days / HALFLIFE_DAYS)
    y = GOAL_WEIGHT * past["goals_for"].to_numpy() + (1 - GOAL_WEIGHT) * past["xg_for"].to_numpy()

    X = np.zeros((len(past), 2 * n + 1))
    X[np.arange(len(past)), past["team"].map(index).to_numpy()] = 1.0
    X[np.arange(len(past)), n + past["opponent"].map(index).to_numpy()] = -1.0
    X[:, 2 * n] = past["home"].to_numpy()

    coef, intercept = _poisson_newton(X, np.clip(y, 0.0, None), weight)
    att = {t: float(coef[index[t]]) for t in teams}
    dfn = {t: float(coef[n + index[t]]) for t in teams}
    # Centre the ratings so they read as "above/below average"; the intercept absorbs it.
    att_mean, def_mean = float(np.mean(list(att.values()))), float(np.mean(list(dfn.values())))
    att = {t: v - att_mean for t, v in att.items()}
    dfn = {t: v - def_mean for t, v in dfn.items()}
    return Ratings(intercept + att_mean - def_mean, float(coef[2 * n]), att, dfn)


def _poisson_newton(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, *, ridge: float = RIDGE, iters: int = 25
) -> tuple[np.ndarray, float]:
    """Weighted Poisson regression with a log link by Newton's method.

    Forty-odd parameters and a few thousand rows: each step is one small linear
    solve, so the whole fit takes milliseconds -- it is refitted before every
    gameweek of every season, so this matters. The ridge term (not applied to the
    intercept) keeps a club with two matches on record close to the average.
    """
    n, p = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(p + 1)
    beta[-1] = math.log(max(float(np.average(y, weights=weight)), 0.05))
    penalty = np.full(p + 1, ridge)
    penalty[-1] = 0.0

    def objective(b: np.ndarray) -> float:
        eta = Xb @ b
        return float(np.sum(weight * (np.exp(eta) - y * eta)) + 0.5 * np.sum(penalty * b * b))

    current = objective(beta)
    for _ in range(iters):
        lam = np.exp(Xb @ beta)
        grad = Xb.T @ (weight * (lam - y)) + penalty * beta
        hess = (Xb * (weight * lam)[:, None]).T @ Xb + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        size = 1.0
        while size > 1e-4:
            candidate = beta - size * step
            value = objective(candidate)
            if value <= current:
                break
            size /= 2
        if abs(current - value) < 1e-9:
            beta, current = candidate, value
            break
        beta, current = candidate, value
    return beta[:-1], float(beta[-1])


def attach_team_strength(silver: pd.DataFrame) -> pd.DataFrame:
    """Per-fixture strength features on the silver rows, fitted before each gameweek.

    A gameweek's ratings use only matches that kicked off before its first fixture,
    so the rows of the gameweek being predicted never contribute to their own
    features -- the same cutoff the walk-forward uses.
    """
    out = silver.copy()
    for column in FEATURES:
        out[column] = np.nan
    if not {"team", "opponent", "was_home", "kickoff_time", "season", "GW"} <= set(out.columns):
        return out
    matches = team_matches(silver)
    if matches.empty:
        return out
    cutoffs = out.groupby(["season", "GW"])["kickoff_time"].min().sort_values().reset_index()
    fixtures = out[["season", "GW", "team", "opponent", "was_home"]].drop_duplicates()
    records = []
    for r in cutoffs.itertuples(index=False):
        ratings = fit_ratings(matches, r.kickoff_time)
        block = fixtures[(fixtures["season"] == r.season) & (fixtures["GW"] == r.GW)]
        for f in block.itertuples(index=False):
            feats = ratings.features(f.team, f.opponent, bool(f.was_home))
            records.append({"season": r.season, "GW": r.GW, "team": f.team,
                            "opponent": f.opponent, "was_home": f.was_home, **feats})  # fmt: skip
    table = pd.DataFrame(records)
    out = out.drop(columns=list(FEATURES)).merge(
        table, on=["season", "GW", "team", "opponent", "was_home"], how="left"
    )
    log.info("team strength: %d gameweek refits, %d fixture rows rated", len(cutoffs), len(table))
    return out


def latest_ratings(silver: pd.DataFrame, as_of: pd.Timestamp) -> Ratings:
    """The ratings a projection made at ``as_of`` should use."""
    return fit_ratings(team_matches(silver), as_of)


__all__ = [
    "FEATURES",
    "Ratings",
    "attach_team_strength",
    "fit_ratings",
    "latest_ratings",
    "team_matches",
]

# FPL Analyst

Rate your Fantasy Premier League squad against this week's projections, see who's
flagged by injury or transfer news, get the five transfers that lift your team most,
a three-gameweek plan, and the week each chip is worth playing. Import your squad
with your FPL team ID. Live for 2026-27, refreshed daily and every six hours in the
last day and a half before a deadline.

```
fpl snapshot     # pull the live FPL API into an immutable, timestamped snapshot
fpl schedule     # cup and European fixtures for PL clubs (fixture congestion)
fpl simulate     # replay GW1-3 from scratch with real transfer rules, scored on real points
fpl live         # project the next five gameweeks and write site/data/live.json
fpl check        # validate live.json before it is deployed
```

The page (`site/`) is static: it loads that one JSON file and does the rating in the
browser, so it runs on GitHub Pages with nothing behind it.

## What the live page does

**My Team.** Search and pick your fifteen (quotas and the three-per-club rule are
enforced as you go), enter your team value, bank and free transfers, and the page
scores the squad and rates it against the best £100m squad the model can build over
the same horizon. Flagged players are called out with FPL's own news text.

**How a squad is scored.** For each gameweek in the horizon the best legal eleven is
picked on *that* week's projections (your formation is honoured for the next gameweek
only — you can change it later), the top projection is doubled as captain, and
expected bench cover is added: the chance that at least one, two or three starters
miss out, times what the first, second or third sub would bring on. So a cheap bench
that would actually play is worth something, an expensive one that never plays is
not, and a player who is benched this week but starts next week against a kinder
opponent is credited for that. The weeks are weighted 100/85/70/55/40%.

**Best lineup.** Under the rating, the recommended eleven for the next gameweek:
formation, captain and vice, bench in order, each fixture coloured by difficulty for
that position, and the formation and captain the model would use in the weeks after.

**Five moves, ranked.** For every player you own, every same-position replacement you
can afford is tried, and the resulting fifteen is re-solved for its best eleven. The
gain reported is the change in the *team's* projected points, not the difference
between two players — a signing who doesn't make your eleven gains you nothing, and
one who changes your captain gains more than his own number. The top five by gain,
never repeating a signing, are shown with the rating you'd have after each. If the
best upgrade costs more than you have, a second, funding transfer is found so the
pair fits under the cap (£100m, or your own team value plus bank if you enter it).

**Free transfers and hits.** Enter how many free transfers you hold. Any transfer
beyond them is charged FPL's 4-point hit, moves are ranked by gain *net* of the hit,
and each says plainly whether it is still worth it.

**Budget.** The cap is £100m, or your own team value plus bank when you enter them —
a team value above what your fifteen cost is headroom the moves may spend; one below
it means the squad is over budget, in which case only moves that bring it back under
are offered, ranked by how little they cost you, even if they cost points. The Best
squad tab is solved at £100m, £105m … £130m and shows the one nearest your budget.

**Teams & fixtures.** Every club's attack and defence on current form — expected goals
for and against per match, exponentially weighted so the last two or three matches
carry most of the weight, blended 70/30 with the last 38 matches — ranked and banded
1 (easy) to 5 (hard) from a defender's point of view (how much they score) and an
attacker's (how little they concede). Fixture chips on the pitch, in the lineup, in
the moves and in the per-gameweek columns of the player table use the same bands.
Each club's full schedule, league and European, is listed.

**Fixtures & insights.** Under the moves, this gameweek's fixtures with both sides'
bands and, on tap, the head-to-head record (last six meetings, clean sheets, both
scored, recent scorelines) and each side's form line.

**All players.** The whole data set in one sideways-scrolling table: projections per
horizon and per gameweek (coloured by fixture difficulty), FPL's own xP, price and
price change, ownership and this week's transfers in and out, start probability,
minutes share, set-piece duties (penalties, corners, free kicks), club attack and
defence ranks, and the full season stat sheet.

**The next five gameweeks, weighted.** Every player is projected for each of the next
five gameweeks against that gameweek's actual opponent: a defender's clean-sheet odds
depend on how much the opponent scores (rolling xG) and how much his own club concedes;
an attacker's returns on how much the opponent concedes (rolling xG conceded) and how
much his own club scores. The gameweeks are weighted 100/85/70/55/40% so the next
deadline dominates but a kind run still counts, and you can rate over the next one,
three or five.

**Why this move.** Each suggestion expands into its case: the fixture-by-fixture run
for the player out and the player in, with the opponent's attack or defence rank
alongside each fixture, both clubs' own strength, which scoring components the points
come from, availability news, the money, and the hit arithmetic.

**Backtest.** The Backtest tab replays the season from the GW1 deadline: a £100m squad
built with only what was knowable then, one free transfer a week from GW2 (bankable,
never a hit), lineup and captain by projection, scored on real points with automatic
substitutions, against the average manager and the week's top score.

**News and injuries.** Every projection is scaled by FPL's availability flag:
`chance_of_playing_next_round` where a percentage is given, zero for injured,
suspended or departed players with no percentage, 75% for an unspecified doubt. The
flag is applied after the model rather than learned by it, because it carries
information — scans, press conferences, loan moves — that no rolling window can see.
It moves every day, which is why the refresh is daily.

## Pipeline

```
fpl bootstrap    # historical seasons -> data/bronze
fpl silver       # one identity per player across seasons -> data/silver
fpl features     # leak-free feature table -> data/gold
fpl backtest     # walk-forward comparison of every model
fpl optimise     # what the projections score as an actual squad
fpl publish 38   # backtest dashboard JSON for a past gameweek
```

Training covers 2022-23 to 2025-26 from the archive plus the current season's
finished fixtures from the snapshot; the upcoming fixtures are what gets predicted.
Double gameweeks produce two rows per player and are scored twice, as the rules do.

## Results

Walk-forward over 2024-25, gameweeks 6–38, refitting before every gameweek on all
prior data (2022-23 onward). 23,708 player-gameweek predictions per model, identical
folds for all of them.

| Model | Spearman | Precision@10 | RMSE | Haulers RMSE | Squad pts / GW |
| --- | ---: | ---: | ---: | ---: | ---: |
| FPL's own `xP` (as archived — see caveat) | *0.756* | *0.642* | *1.844* | *4.975* | *91.3* |
| Component model, round two (this project) | **0.717** | 0.518 | **1.909** | **5.475** | — |
| Component model, round one (tuned) | 0.715 | **0.521** | 1.914 | 5.485 | 69.1 |
| Component model, earlier defaults | 0.716 | 0.521 | 1.918 | 5.568 | 69.1 |
| Component model, trained on six seasons (2020-26) | 0.716 | 0.524 | 1.914 | 5.533 | — |
| LightGBM, one regressor | 0.709 | 0.479 | 1.934 | 5.524 | 61.7 |
| Rolling 3-gameweek mean | 0.703 | 0.373 | 2.182 | 5.790 | 52.6 |
| Constant (training mean) | — | 0.073 | 2.354 | 7.172 | — |
| *Perfect foresight (ceiling)* | | | | | *152.3* |

Spearman is the mean within-gameweek rank correlation between projection and outcome.
Precision@10 is the share of each gameweek's ten highest projections that scored five
or more. "Squad pts / GW" picks a fresh optimal fifteen every gameweek under the full
rules — £100m, positional quotas, three per club, legal formation, captain doubled —
and reports what that eleven actually scored. Haulers RMSE follows
[OpenFPL](https://arxiv.org/abs/2508.09992)'s segmentation so the figure is comparable
(OpenFPL reports 5.142; FPL Review, a paid service, 5.172).

**The honest reading.** The component model beats every baseline it was designed to
beat, by a clear margin in the metric that matters most for squad selection
(precision@10: 0.521 against 0.479), and by 7.4 points a gameweek when the projections
are turned into a squad. The club-strength, form and congestion features added later
moved precision@10 from 0.506 to 0.521 and left the other metrics flat: the model
already knew most of what those features say. Two more experiments, both measured
on the same 33 folds, say the same thing from the other side: training on six seasons
instead of four (adding 2020-21 and 2021-22, which have no expected-goals data)
changes nothing that matters, and a hyper-parameter search (`scripts/tune.py`:
shallower trees, half the features per tree, slower learning) improves the error on
big scores a little and nothing else. The model is saturated on the information it
has. What moves it now is information it does not have — bookmaker prices and the
availability history, both wired in above and both waiting on data.

**Round two.** The minutes model, the club ratings, the set-piece duties and the
minutes-pattern features (all described below) were added together and judged the
same way, first on coarse folds (refit at gameweeks 6, 14, 22 and 30) to choose
between variants, then on the full 33 refits above. They move every error metric
the right way by a small, consistent amount -- RMSE 1.914 to 1.909, the big-score
RMSE 5.485 to 5.475, rank correlation 0.715 to 0.717 -- and leave precision@10 where
it was (0.518 against 0.521 is one pick in three hundred and thirty). Fitting the
appearance components as rates rather than counts was worth more than the new
features were. Two further ideas were measured on the same folds and rejected:
weighting recent seasons more heavily (a 365-day half-life) cost precision@10 seven
picks and made big scores worse, and averaging three differently seeded fits changed
nothing and took five times as long. The per-position calibration measured on this
run (`CALIBRATION` in `config.py`: midfielders 0.95, forwards 0.96, keepers 0.97,
defenders 1.02) is applied to the live projections so a midfielder's five and a
defender's five mean the same thing. The archive cannot measure the live-only
changes -- FPL's figure blended in, the deadline-aware refresh -- which is what the
live scorecard is for.

**Captaincy.** The haul classifier is well calibrated, and it does not pick better
captains. On the same coarse folds, choosing the captain each gameweek from the
fifteen highest-projected players by expected points alone averaged 6.73 points
(2024-25) and 6.52 (2023-24); adding the weighted haul chance moved that to 7.12 and
6.18 at a weight of 8, 6.94 and 6.58 at 12, 6.79 and 5.45 at 4 -- up one season, down
the other, with no weight that helps both. Thirty-three captain picks a season is a
small sample and the differences are one or two hauls either way. So the weight ships
at zero: the captain is the highest expected points, the haul chance is shown beside
each candidate for the reader, and the weight is re-measured when more seasons of
walk-forward predictions exist. Perfect hindsight within the same fifteen would
average about 16 a week, which says how much of captaincy is luck.

**A whole season, with transfer rules.** `fpl replay` plays 2024-25 from gameweek 6
to 38 under FPL's rules — a £100m squad, one free transfer a week bankable to five,
never a hit, lineup and captain re-picked weekly, automatic substitutions — using the
cached walk-forward projections, each made before its gameweek. Two managers play the
same season on the same code: the *model* manager ranks by projection, the *crowd*
manager ranks by ownership at each deadline (the template team, which is what the
average manager owns). Neither can see injury flags. Result: model **1,840** points,
crowd **1,759** — an 81-point edge, 55.8 against 53.3 a week, over 33 gameweeks
(round one of the model scored 1,823). The Backtest tab shows the two week by week.

The GW1–3 season replay is a three-gameweek sample and behaves like one: the same
model with the earlier defaults scored 181 points (43 / 88 / 50) and with the tuned
parameters 166 (27 / 99 / 40) — a fifteen-point swing between two models the 33-fold
backtest cannot tell apart, driven by which fringe player happened to blank. Read it
as a demonstration of the pipeline, not as a measurement. What it *is* good for is
catching a broken decision: the 2026-27 replay once built a £95.5m squad with £4.5m
unspent, two £4.5m forwards on the bench who could never play, a 5-5-1 eleven and a
£4.1m defender as captain. Those were three separate bugs — a bench counted at
nothing, no prior for players with no record, and an armband chosen on the mean —
and each has its own section below. The replay after the fixes: 33 / 95 / 47 = 175
against a 182 average, with the money spent (£99.5m), a bench that plays, a 3-5-2
and Bruno Fernandes captain.

**Defensive contribution.** The 2025-26 rule (two points for ten tackles,
interceptions and clearances, or twelve for midfielders) is a scoring event like any
other, so it has its own sub-model — but until this round it had nothing to learn
from: the target was zero for three of four training seasons and there was no
history feature for it, so it predicted 0.00 where defenders actually earn 0.21 a
game, and every defender was under-projected by about a point. The player's own
defensive-contribution history (rolling and per-90) and a flag for the seasons the
rule applies to are now features. On the first rule season, walk-forward, the
component moves from 0.00 to 0.11 against an actual 0.21 — the fold refits see only
a few weeks of the new rule — and defenders' projections from 2.09 to 2.34 against
3.06. On 2024-25 the features are constant and the scores are unchanged (Spearman
0.716). With a full season of the rule now on record, the live model has what the
backtest folds did not.

**The `xP` caveat.** The archived `xP` row is not a fair baseline. It was scraped
after each gameweek, and FPL folds the gameweek's real points into the "form" its
figure is built from: hauls in the archive carry `xP` values of 20, 30, even 49.6,
which FPL never publishes before a deadline. The tell was that stacking it as a
feature lifted precision@10 to 0.90 — the leak was caught and the feature removed.
FPL's live `ep_next` is legitimate pre-deadline information and is used only to rein
in players with no league record. The real pre-match `xP` is certainly weaker than
the row above; how much weaker cannot be measured from this archive.

## Why the component model

Most FPL projects fit one regressor to `total_points`. This one predicts each way of
scoring separately and sums them using FPL's actual rules:

```
E[points] = E[appearances] x 1 + E[60-minute appearances] x 1
          + E[goals]  x {6 GK/DEF, 5 MID, 4 FWD}
          + E[assists] x 3
          + E[clean sheets] x {4 GK/DEF, 1 MID}
          + E[bonus]
          + E[saves]/3            (GK)
          - E[goals conceded]/2   (GK/DEF)
```

Every term is a Poisson-objective LightGBM model over the same leak-free features.
Modelling counts rather than probabilities is what makes double gameweeks score
correctly: a defender with two clean sheets in a week earns eight points, not four.
Switching from binary flags to counts moved precision@10 from 0.470 to 0.506 on its
own, because double gameweeks are where the big scores are.

The decomposition is also what makes the explanation layer possible. A projection of
6.2 arrives as 1.8 appearance + 2.9 attacking + 0.9 clean sheet + 0.6 bonus, and the
written rationale narrates those figures rather than inventing its own.

What would lift it further, in order of expected value:

1. **Bookmaker odds.** Wired in (see below) but not yet measured: football-data.co.uk
   was returning 503s the day the pipeline was built, so the results files and the
   first backtest with odds land with the next refresh that can reach it.
2. **Availability flags.** Applied to every live projection, and logged from every
   snapshot from now on, so the replay becomes a fair test as the season goes.
3. **Shot-level xG** from Understat, to separate a striker taking six weak shots from
   one taking a single big chance.
4. **More seasons.** 2020-21 and 2021-22 carry no expected-goals columns at all, so
   adding them means a third of the training set lacks the attacking model's core
   inputs. Whether they help anyway is an empirical question the backtest answers
   (`FPL_TRAIN_SEASONS="2020-21,2021-22,2022-23,2023-24,2024-25,2025-26"`); see Results.

## Two design decisions worth reading

**Point-in-time correctness is enforced, not assumed.** `tests/test_no_leakage.py`
does not inspect the feature code. It rewrites match outcomes and asserts that the
features for earlier gameweeks are bit-identical. Two corruptions run: one rewrites
every gameweek *after* a cutoff (catching features that look forward), and one
rewrites the cutoff gameweek *itself* (catching features that absorb team-mates'
results from the very match being predicted). The second was added after the first
had been passing for a while — the shrinkage prior was accumulating row by row and
leaking sideways, and no amount of reading the code had spotted it. The test found it
in seconds. It also caught a whole-season mean used as a normaliser. Neither would
have shown up in the backtest scores; both would have made them lies.

**Identity is resolved before anything else.** FPL's `element` id is season-local and
reused: of the 866 ids in the archive, 804 refer to more than one footballer. Joining
seasons on it silently corrupts every rolling feature. `fpl.entity.resolve` builds the
`(season, element) -> code` bridge from the player registry and fails hard on any
unresolved row — a 99%-successful join is the worst outcome, because it looks fine.

**The squad optimiser buys a bench.** The mixed-integer program that builds a squad
from scratch (`optimise/squad.py`) maximises the starting eleven plus a captain, with
the four bench players counted at `BENCH_WEIGHT` (0.15) of their projection --
roughly the chance a substitute comes on. Counting the bench at nothing, as the first
version did, is what produced a £95.5m squad with £4.5m unspent and two £4.5m
forwards who could never cover an absence: the solver had no reason to spend on
anyone who was not starting. Counting it in full would buy an expensive bench that
never plays. The weight is small on purpose; it is there to make the solver prefer a
£5.0m sub who plays to a £4.5m one who does not.

## Validation

Walk-forward only. Train on every gameweek up to *t*, predict *t+1*, roll forward. A
random train/test split lets the model learn from March to predict September and
inflates every metric.

Four baselines are scored on the same folds, in ascending order of how embarrassing
it is to lose to them: the training mean, a rolling three-gameweek average, FPL's own
published projection, and a monolithic LightGBM regressor. Metrics are segmented by
outcome — zeros, blanks, tickers, haulers — because 60% of player-gameweeks score
nothing and an overall MAE flatters any model that predicts "everybody gets one".

## Explanation layer

The language model never produces a number. It receives the decomposition as JSON
and writes two or three sentences; a guardrail then rejects any output containing a
numeral that was not in the input and falls back to a deterministic template. The
interesting engineering claim is the guardrail, not the prompt — anyone can call an
API; constraining it so it cannot fabricate a statistic is the part worth defending.
Without an `ANTHROPIC_API_KEY` the template runs, so the repo works for anyone who
clones it.

## Refresh schedule

Publishing it and switching the refresh on is a one-off, documented step by step in
[PUBLISHING.md](PUBLISHING.md).

The page's **Refresh data** button re-fetches the published file (bypassing any cached
copy) and says whether anything newer landed; there is nothing to run by hand.

`.github/workflows/weekly-refresh.yml` wakes every six hours. Its first step, `fpl
due`, is one request to the FPL API and a look at the last published file: the run
goes on only within 36 hours of the next deadline (when press conferences and injury
news land), or if nothing has been published for 20 hours (a daily heartbeat, so the
availability log keeps its record and a stalled site is noticed), or when started by
hand. A run that goes on:

1. runs the test suite -- a refresh from broken code is not a refresh;
2. restores the historical archive from cache (it only changes when a season ends);
3. snapshots the live API (never overwriting an earlier snapshot; retried with
   backoff because the API rate-limits around deadlines);
4. fetches the cup and European fixture lists (`fpl schedule --all`);
5. replays the season from GW1 for the Backtest tab;
6. refits the model on everything up to the next gameweek and writes
   `site/data/live.json`;
7. validates the file (`fpl check`: player count, clubs, sane optimum, no NaNs,
   snapshot age); if the build or the check fails, the previously published file is
   restored and deployed instead, and the run is flagged;
8. commits the refreshed data back to the repository so the last good copy advances,
   then deploys the site.

The page shows when the data was captured and how long remains to the deadline, and
says so plainly when the file is more than 36 hours old or the deadline has passed.

The historical archive this trains on stopped weekly updates after 2024-25, so the
snapshots are also the in-season history: each one carries every finished fixture of
the current season from `element-summary/`.

## Serving many users

The site is static: one HTML file and one JSON file on GitHub Pages, served by a CDN
with no server of ours to overload. Every rating and transfer search runs in the
visitor's browser (a full five-move search over the five-week horizon takes well
under a second on a laptop; the picker renders its 650 rows in chunks so a tap is
never blocked). The JSON is about 2 MB raw, roughly 300 KB compressed, and is
revalidated against the CDN's ETag on each visit. Each
browser keeps the last good copy it saw, so a failed fetch shows that copy with a
banner rather than a blank page, and a page error shows a notice instead of dying
silently. Squads are stored per device in `localStorage`; nothing is sent anywhere.

## Club ratings

The rolling windows say how a club *has* scored and conceded; `fpl/models/team_strength.py`
says how it *should*, against a given opponent at a given venue. It is a Poisson model
in the Dixon-Coles tradition: the log of a side's expected goals is its attack rating
minus the opponent's defence rating plus a home advantage, fitted on every match the
clubs have played, each weighted down as it ages (half-life 100 days), on a target that
blends goals with expected goals (35/65 -- xG carries the repeatable part, goals the
finishing). The fit is a forty-parameter weighted Poisson regression solved by Newton's
method in milliseconds, so it is refitted before every gameweek of every season, on
matches that kicked off before that gameweek; the leakage test covers it. Per fixture
it yields expected goals for and against, a clean-sheet probability and a win
probability, summed across a double gameweek like the odds; for the four later
gameweeks of the horizon the ratings as of the deadline are applied to each opponent.
It is opponent-adjusted, which a rolling mean is not, and it needs no external source.
The attack and defence ranks the page shows -- and the fixture-toughness colours built
on them -- come from these ratings too (expected goals for and against an average
opponent at a neutral venue). They used to follow a two-match form window, and a
data bug flattered it further: the live snapshot booked a transferred player's
earlier fixtures to his *new* club, as phantom matches with nothing conceded, which
once put a defence that had shipped five in three games third in the league. Each
row is now booked to the club the player was actually playing for that day, and the
ratings do not forget a season of evidence the way a two-match window does.

## Minutes and set pieces

Whether a player plays, and for how long, is the biggest single source of error in
any projection. Two changes address it. The appearance and 60-minute components are
now fitted as per-fixture *rates* -- a label in [0, 1] with a cross-entropy objective,
multiplied back by the number of fixtures -- rather than Poisson counts: a probability
of playing is what they are, and a classifier calibrates it better. And the feature
table carries how a player has been used, not just how much: consecutive starts,
gameweeks since the last start, the share of recent appearances that were cameos off
the bench, the spread of his minutes, and where his minutes rank among his club's
players in the same position (with how many of them are regular starters). Set-piece
duties -- penalties, corners, direct free kicks -- come from the player registry:
today's for the live season, the end-of-season state for archive seasons (a taker who
inherited the duty in March is flagged from August; duties change rarely enough that
the signal is worth that blemish, and it is documented in the code).

## FPL's own figure

For the next gameweek only, FPL's `ep_next` is blended into the projection at 25%
(`FPL_BLEND` in `config.py`), for unflagged players with a figure. FPL's number knows
the press conference and the training ground; the model knows the fixtures and the
underlying rates; two decent, different forecasts averaged usually beat either. The
archive has no honest pre-deadline copy of `ep_next` (see the `xP` caveat), so the
weight cannot be backtested; instead the projection log keeps the model's figure,
FPL's and the blend side by side, and the live scorecard reports each one's rank
correlation and squad points as gameweeks are played. If the model alone keeps
beating the blend, the weight comes down; if FPL alone does, it goes up.

## Match actions

FPL's feed says what a player was *awarded*; it does not say how many shots he had,
how often he touched the ball in the box, how many chances he created or how many
tackles, interceptions, blocks and clearances he made. Those actions are what turn
into goals, assists and defensive-contribution points, and they are far steadier
than the points themselves. The open FPL-Core-Insights dataset publishes them per
player per match from 2024-25 on, keyed by the official FPL ids, refreshed twice a
day; `fpl core` pulls it with a sparse git clone and boils it down to one committed
CSV per season (`data/external/core_insights_<season>.csv`), and the feature table
carries each action as a per-gameweek count rolled over three and ten gameweeks and
as a shrunk per-90 rate, plus whether he started. Seasons before the dataset are NaN
-- "unknown" to the trees, not zero -- and the leakage suite corrupts this source
exactly as it corrupts the archive. Measured on the 2025-26 coarse walk-forward
against the same model without them: rank correlation 0.731 either way, RMSE 1.906
against 1.907, precision@20 0.398 against 0.383, precision@10 0.442 against 0.448 --
a wash on the season the folds could see, which is what one expects when only one
prior season carries the columns. They stay in for the season now being played,
where every gameweek adds to the history they need, and are re-measured as it goes.

## Newcomers

A player with fewer than ninety league minutes on record has a price, an ownership
and a club, but no evidence of his own; the trees still have to say something about
him, and what they say is an extrapolation from the handful who looked like him --
which is how a £4.1m defender with ten minutes to his name was once the model's
captain in a backtest. Two guards. Live, `ep_next` caps such a player at 1.25x FPL's
own figure. Everywhere -- backtests included, where FPL's figure does not exist --
the *quality* of his appearances is shrunk toward the market's view: the median
points-per-appearance of established players in his position within £0.5m of his
price, in the same gameweek, with the model's own figure earning its full weight only
at ninety minutes. The shrinkage is applied per appearance rather than per game on
purpose: the model's judgement of whether he plays (price, ownership, the registry)
is the best there is for a newcomer, and a per-game prior would hand a non-playing
£4.5m keeper a starter's projection.

## Captaincy

The armband is the highest-leverage decision of the week and it is decided for
*one* gameweek at a time -- a player's run of fixtures never chooses it. A tenth
sub-model, a binary classifier on the same features, predicts each player's chance of
a haul (eight points or more) in the coming gameweek; it is well calibrated (players
given a 24% chance haul 23% of the time). The captain each week is the starter with
the highest **expected points + `CAPTAIN_HAUL_WEIGHT` x P(haul)**, so a steady 6.0
could lose the armband to a 5.8 with a fatter tail. The weight is set on the
walk-forward: for every gameweek the captain is picked from the fifteen highest
projected players and what he actually scored is averaged over the season
(`scripts/captaincy.py`). Measured on two seasons it does not help reliably, so it
ships at zero -- expected points choose, the haul chance informs; see Results.

One guardrail: the armband goes to a midfielder or forward unless a keeper or
defender is at least `CAPTAIN_DEFENDER_MARGIN` (1.0) points clear of the best
attacker. Expected points are a mean; a captain is a bet on a ceiling, and a
defender's ceiling is a clean sheet where an attacker's is a hat-trick. The
projection alone cannot see that, and left to it the squad builder once handed the
armband to a cheap defender on a 0.1-point edge. The same rule runs in the squad
optimiser (its captain variable is fixed at zero for keepers and defenders), in the
season simulation, and in the page's Arrange and Auto-pick. The Captain card on the
My team tab shows the top five options for the gameweek with each one's haul chance.

## Plan and chips

**The multi-week solver.** Transfers are planned over the next five gameweeks as one
mixed-integer program (`optimise/plan.py`), the formulation the open-source FPL
solvers (sertalpbilal/FPL-Optimization-Tools and its descendants) made standard,
written against this project's projections and rules: a squad, a lineup and a
captain for every week; transfers linking one week's squad to the next; free
transfers banking up to five, every transfer beyond them costing four points; the
budget fixed at squad value plus bank; the objective the same team-points measure the
rating uses (eleven, captain doubled, bench at 15%), weighted 100/85/70/55/40, less
hits, less a hair per transfer so it does not churn. It runs on a pruned pool (the
fifteen, the best eight per position over the run, the cheapest two who actually
play) and CBC solves it in a couple of seconds. The season replay uses it for every
week's transfer (`fpl simulate --planner greedy` gives the older one-week suggester).
The page cannot run a solver, so its plan is a beam search that scores every path the
same way -- each week the best few moves and holding are carried forward and the best
three-week path wins -- which is how banking a transfer for a double next week can
beat a small move now, and why the page's plan and the solver's usually agree.

**On the page.** Under the moves, the plan for the next three gameweeks with each
week's transfer, its gain, any hit and the free transfers left, against what standing
still would return. Only this week's move is a decision; later weeks say where the
squad is heading and get re-planned every refresh. Chips: for this squad, the week in
the run where each chip would earn most and by how much -- Bench Boost (all fifteen
score), Triple Captain (the captain counts three times), Free Hit (the best one-week
squad money can buy, against yours), Wildcard (the solved best squad over the run,
against yours) -- with the note that blank and double gameweeks are known only a few
weeks ahead. Chips already played, when a team is imported, are marked used.

## Substitutions

Drag a card onto another (or tap the ⇄ on a card, then the card to swap with) to
make a substitution. Dragging works with a finger as well as a mouse: the browser's
own drag-and-drop never fires on iPhone Safari, so the pitch uses SortableJS (from a
CDN, with the native events as a fallback) -- press, hold a moment, drag onto the
player to swap with; the cards you can drop on light up, the rest fade. Same position swaps straight across, pitch or bench; a different
position is allowed between pitch and bench when the eleven stays a legal formation,
and the formation changes with it, as in the FPL app -- keepers only swap with
keepers. The arrangement is kept between visits. When the eleven on the pitch are not
the best eleven from the fifteen in that shape, the rating card says by how much and
offers to arrange them.

## Player profiles

Tap any player -- a card on the pitch, a row in a table, a name in a transfer -- and
a sheet opens with his profile, the way the FPL app does it: photo, club, price and
ownership; the next-gameweek and five-gameweek projections beside FPL's own figure
and his chance of starting; the run of five fixtures as a ribbon, each opponent
coloured by how tough it is for his position, with the points projected against each;
where this week's points come from; and, under "Full Profile", the season so far
(points, form, minutes, goals, assists, xG, xA, clean sheets, bonus, defensive
contributions), his set-piece duties, his club's attack and defence ranks and the
market (transfers in and out, price change). The actions follow from where you are:
*Select Replacement* for a player you own (it opens the picker for his slot), *Add to
Squad* or *Swap In…* for one you don't, *Replace X* when you came from the picker,
and *Add to Comparison* everywhere.

## Comparison

Up to five players side by side. Add them with the + beside any player (the picker,
the Players table, a profile) or with *Compare* on a suggested transfer, which puts
the player leaving and the player arriving in the tray together. The comparison
sheet lines up everything the model knows about them -- projections, the five
fixtures with their toughness, the points decomposition, the season's numbers, set
pieces, club strength and the market -- and marks the best of the group on each line
(lower is better for goals conceded, ranks and set-piece order). A row at the top
says who is already in your squad; for the others, *Bring in for…* swaps a player of
the same position out and takes you back to the rating. The tray stays put as you
move between tabs, so a comparison can be built up from several places.

## The interface

The page follows Apple's Human Interface Guidelines, so it feels at home on an
iPhone and a Mac: the system typeface with the iOS type scale on phones and the macOS
scale on desktop, Apple's semantic colours in light and dark (with the elevated
surfaces dark mode uses for sheets), inset grouped cards, segmented controls for
horizon and formation, a translucent bottom tab bar on phones and a toolbar on
desktop, and bottom sheets with a grabber that swipe down to dismiss (a centred card
on wider screens). Every control meets the 44pt touch target on phones; keyboard
focus is visible; reduced motion and reduced transparency are honoured. The one
splash of saturated colour is the pitch, because it is the content; fixture
difficulty uses FPL's own five-step ramp, always with the opponent named beside it,
so nothing is said by colour alone.

## The pitch, the ticker, the stars

Each player on the pitch is a card in the lineup-graphic style: his cut-out photo on
a dark card with a soft glow in his position's colour, the projected points for the
gameweek as the big number, position, name, price, and a full-width fixture bar in
FPL's difficulty colour. Starters and substitutes are told apart by the card tone,
the captain by a yellow C. The Best Squad tab shows the solved fifteen on the same
pitch rather than as a list, with each starter's points breakdown beside it. The
Matches tab has a fixture ticker -- every club's next three, five or eight league
fixtures in a grid, capitals for home, lower case for away, coloured by how tough each
is for the view chosen (defenders or attackers), easiest run first. A star beside any
player keeps him at the top of every list and picker (an idea taken, with the ticker,
from the open-source open-fpl). Comparison lives in its own tab, second from the
left, and the deadline countdown in the header ticks.

## Import your team

Type your FPL team ID (the number in the address bar of your Points page) and the
page loads the squad as it stood after the last gameweek, the bank, the team value,
the chips already used and a reconstruction of your free transfers (FPL does not
publish the count; it is rebuilt from the transfer history: one a week, banked to
five, reset by a wildcard or free hit). FPL's API refuses browser requests from other
sites, so the fetch goes through a public relay -- three are tried in turn -- and
if none answers the page says so and you pick by hand. Nothing private is involved: a
team's picks are public on FPL for anyone with the ID.

## Bookmaker odds

`fpl odds --all` pulls football-data.co.uk's results files (one per season, a dozen
bookmakers' prices per match) and its `fixtures.csv` (the same prices for matches not
yet played), and stores them in `data/external/`. The site sits behind a bot filter
that has answered 503 to plain clients for days at a time, so the fetch now presents
itself as a browser, tries three spellings of the host in turn, and logs every
response code so the Actions log says why a file is missing rather than just that it
is. For the matches ahead there is a second source: with an `ODDS_API_KEY` secret
(The Odds API, free tier, one request per refresh) the coming gameweek's prices are
averaged across the bookmakers quoted and stored as `odds_live_<season>.csv`, merged
week by week so a history of pre-match prices accumulates on its own. Where a coming
fixture has a price, the market's implied goals for and against are averaged with the
club ratings' (`MARKET_WEIGHT`, 50/50) before the model sees them -- both estimate
the same thing, and the market also knows the team news -- so the live prices help
now, without waiting for a training history. From the 1X2 and over/under 2.5
prices each match yields, per side: the chance of winning, drawing and losing, and —
by solving a two-team Poisson model so that the win probability and the over-2.5
probability both match the market — an implied expected goals for and against, and a
clean-sheet probability. Those join every fixture row by (season, home, away), so a
double gameweek carries two markets, and the same lookup fills the four later
gameweeks of the horizon where the market has already quoted them. Closing prices
are pre-match information, so they are legitimate for training; a season the site
cannot serve is simply NaN and the model falls back to what it knows.

## Live scorecard

The walk-forward and the season replays are handicapped in the same way: the archive
has no injury flags, so every replayed manager has to treat everyone as fit. The live
model does not -- it reads FPL's flags and news every morning. To measure *that*
model, every refresh writes the coming gameweek's projections to
`data/external/projections/<season>_gw<NN>.csv` before the deadline and commits them;
once the gameweek is played they are scored against what happened. The Backtest tab
shows rank correlation, what the ten highest-projected players returned, and what a
fresh £100m squad picked on those projections actually scored next to the average
manager, gameweek by gameweek. Because the file is committed before kick-off, the
record cannot be revised afterwards.

## Availability history

FPL keeps no history of injury flags, which is why the season replay has to treat
everyone as fit at past deadlines. `fpl snapshot` now appends every player's flag,
chance, news and price to `data/external/availability_log.csv` (committed by the
refresh), so from this point on a backtest can rebuild what was known at each deadline.

## Fixture congestion

FPL only knows about league fixtures, but a Wednesday in Munich is why a full-back is
benched on Saturday. `fpl schedule` pulls every Champions League, Europa League and
Conference League fixture from fixturedownload.com (2022-23 to 2026-27, so the model
can learn the effect as well as see it), keeps the matches involving Premier League
clubs, and stores them in `data/external/` (committed, so a source that is down keeps
its last copy). The FA Cup and League Cup come from TheSportsDB, whose free feed is
too patchy to rely on yet, so domestic cups are a known gap. From them
each fixture row gets: matches in any other competition in the seven days before
kick-off, whether a European tie fell within four days before (rotation after), whether
a cup or European tie follows within four days (rotation before), and rest counted
across every competition. The same features are computed for the opponent, and for
every gameweek in the five-week horizon. The Why panel flags them on each fixture.

## Data sources

| Source | Provides | Access |
| --- | --- | --- |
| [Official FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) | Points, prices, minutes, xG/xA, set-piece order, availability | Public JSON |
| [vaastav archive](https://github.com/vaastav/Fantasy-Premier-League) | Historical seasons | GitHub |
| [FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) | Per-match player actions (shots, box touches, chances, tackles, interceptions, blocks, clearances, goals prevented) from 2024-25, keyed by FPL ids | GitHub, refreshed twice daily |
| [FPL image server](https://resources.premierleague.com) | Player headshots, copied once into `site/photos/` | Public PNG |
| [Understat](https://understat.com) | Shot-level xG | Scrape, gently |
| [football-data.co.uk](https://www.football-data.co.uk/englandm.php) | Historical closing odds | Static CSV |
| [ClubElo](http://clubelo.com/API) | Dated team strength | CSV API |
| [The Odds API](https://the-odds-api.com) | Live pre-deadline odds | REST, free tier |
| [fixturedownload.com](https://fixturedownload.com) | Champions / Europa / Conference League fixtures, current and past seasons | Public JSON |
| [TheSportsDB](https://www.thesportsdb.com/documentation) | FA Cup and League Cup fixtures | Public JSON, 15 season requests a month |

All free. FotMob is deliberately excluded: its terms forbid automated retrieval.
2021-22 is excluded from training because it carries no expected-goals columns at all.

## Layout

```
src/fpl/
  data/       archive loader, live API snapshots, silver builder
  entity/     cross-season identity, name matching, curated club aliases
  features/   gameweek aggregation, lagged windows, shrunk per-90 rates
  models/     baselines, component sub-models, rule-based combiner
  evaluate/   walk-forward harness, segmented metrics, cached comparisons
  optimise/   squad selection (MILP), team rating and transfer suggestions
  explain/    grounded prompt, guardrail, template fallback
  report/     live and backtest dashboard JSON
site/         static dashboard, no build step
tests/        the leakage suite, identity, rules, optimiser, guardrail
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,llm for the Anthropic client
pytest
fpl bootstrap && fpl silver && fpl features
fpl backtest                     # ~5 min on 4 cores; use --from-gw/--to-gw to chunk
```

## Licence

MIT

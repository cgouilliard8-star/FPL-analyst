# FPL Analyst

Rate your Fantasy Premier League squad against this week's projections, see who's
flagged by injury or transfer news, and get the five transfers that lift your team
most. Live for 2026-27, refreshed every morning.

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

**Rate my team.** Search and pick your fifteen (quotas and the three-per-club rule are
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
| Component model (this project, tuned) | **0.715** | **0.521** | 1.914 | 5.485 | 69.1 |
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

**A whole season, with transfer rules.** `fpl replay` plays 2024-25 from gameweek 6
to 38 under FPL's rules — a £100m squad, one free transfer a week bankable to five,
never a hit, lineup and captain re-picked weekly, automatic substitutions — using the
cached walk-forward projections, each made before its gameweek. Two managers play the
same season on the same code: the *model* manager ranks by projection, the *crowd*
manager ranks by ownership at each deadline (the template team, which is what the
average manager owns). Neither can see injury flags. Result: model **1,823** points,
crowd **1,759** — a 64-point edge, 55.2 against 53.3 a week, over 33 gameweeks. The
Backtest tab shows the two week by week.

The GW1–3 season replay is a three-gameweek sample and behaves like one: the same
model with the earlier defaults scored 181 points (43 / 88 / 50) and with the tuned
parameters 166 (27 / 99 / 40) — a fifteen-point swing between two models the 33-fold
backtest cannot tell apart, driven by which fringe player happened to blank. Read it
as a demonstration of the pipeline, not as a measurement.

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

`.github/workflows/weekly-refresh.yml` runs every day at 06:00 UTC and again on
Saturday at 10:00 UTC ahead of the usual deadline. Each run:

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

## Bookmaker odds

`fpl odds --all` pulls football-data.co.uk's results files (one per season, a dozen
bookmakers' prices per match) and its `fixtures.csv` (the same prices for matches not
yet played), and stores them in `data/external/`. From the 1X2 and over/under 2.5
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

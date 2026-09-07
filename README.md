# FPL Analyst

Weekly Fantasy Premier League analysis: projects expected points for every player,
picks the optimal squad under the real FPL constraints, and explains each pick from
the model's own arithmetic.

```
fpl bootstrap    # historical seasons -> data/bronze
fpl silver       # one identity per player across seasons -> data/silver
fpl features     # leak-free feature table -> data/gold
fpl backtest     # walk-forward comparison of every model
fpl optimise     # what the projections score as an actual squad
fpl publish 38   # dashboard JSON for a gameweek -> site/data
```

## Results

Walk-forward over 2024-25, gameweeks 6–38, refitting before every gameweek on all
prior data (2022-23 onward). 23,708 player-gameweek predictions per model, identical
folds for all of them.

| Model | Spearman | Precision@10 | RMSE | Haulers RMSE | Squad pts / GW |
| --- | ---: | ---: | ---: | ---: | ---: |
| FPL's own `xP` | **0.756** | **0.642** | **1.844** | **4.975** | **91.3** |
| Component model (this project) | 0.716 | 0.506 | 1.916 | 5.559 | 69.1 |
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
(precision@10: 0.506 against 0.479), and by 7.4 points a gameweek when the projections
are turned into a squad. It does not beat FPL's own published expected-points figure.
`xP` is produced with information this project does not have — confirmed team news,
press-conference availability, and whatever Opta feeds sit behind it — and the gap is
real. The next section says what would close it.

## Why the component model, and why it lost to `xP`

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

What would close the gap to `xP`, in order of expected value:

1. **Bookmaker odds.** Closing lines absorb team news and rotation that no rolling
   window can see. [football-data.co.uk](https://www.football-data.co.uk/englandm.php)
   publishes them free back to 1993 for training; The Odds API's free tier covers the
   live week. This is the single largest missing signal.
2. **Availability flags.** The live FPL API exposes `chance_of_playing_next_round` and
   `news`. They are not in the historical archive, so a collector has to snapshot them
   at each deadline from now on — see *Live pipeline* below.
3. **Shot-level xG** from Understat, to separate a striker taking six weak shots from
   one taking a single big chance.

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

## Live pipeline

The historical archive this trains on
([vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League))
stopped weekly updates after 2024-25. The in-season data source is therefore a
collector of our own against the official API, run by `.github/workflows/weekly-refresh.yml`
before each deadline. That collector is the next piece of work; the workflow file
shows where it slots in. Until it lands, `fpl publish` operates on the archive, which
is what the dashboard currently shows.

## Data sources

| Source | Provides | Access |
| --- | --- | --- |
| [Official FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) | Points, prices, minutes, xG/xA, set-piece order, availability | Public JSON |
| [vaastav archive](https://github.com/vaastav/Fantasy-Premier-League) | Historical seasons | GitHub |
| [Understat](https://understat.com) | Shot-level xG | Scrape, gently |
| [football-data.co.uk](https://www.football-data.co.uk/englandm.php) | Historical closing odds | Static CSV |
| [ClubElo](http://clubelo.com/API) | Dated team strength | CSV API |
| [The Odds API](https://the-odds-api.com) | Live pre-deadline odds | REST, free tier |

All free. FotMob is deliberately excluded: its terms forbid automated retrieval.
2021-22 is excluded from training because it carries no expected-goals columns at all.

## Layout

```
src/fpl/
  data/       archive loader, silver builder      (strict schema validation)
  entity/     cross-season identity, name matching, curated club aliases
  features/   gameweek aggregation, lagged windows, shrunk per-90 rates
  models/     baselines, component sub-models, rule-based combiner
  evaluate/   walk-forward harness, segmented metrics, cached comparisons
  optimise/   squad selection as a mixed-integer program (PuLP)
  explain/    grounded prompt, guardrail, template fallback
  report/     dashboard JSON
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

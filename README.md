# FPL Analyst

Weekly Fantasy Premier League analysis: projects expected points for every player,
picks the optimal squad under the real FPL constraints, and explains each pick from
the model's own arithmetic.

> **Status:** phase 00 — scaffold. See `docs/` for the architecture and build plan.

## Why this is not another points predictor

Most FPL models fit one regressor to `total_points`. This one predicts each scoring
component separately and sums them:

```
E[points] = P(appear) + P(minutes >= 60)
          + E[goals]  x {6 GK/DEF, 5 MID, 4 FWD}
          + E[assists] x 3
          + P(clean sheet) x {4 GK/DEF, 1 MID}
          + E[bonus]
          + P(defensive contribution) x 2
          + E[saves]/3            (GK)
          - E[goals conceded]/2   (GK/DEF)
          - E[cards]
```

That scores better than a monolithic model, because a defender's points come from
three nearly unrelated processes. It also produces a breakdown, which is what turns
a number into an explanation.

## Two design decisions worth reading

**Point-in-time snapshots.** The FPL API is mutable: prices drift daily,
`chance_of_playing` and `news` are rewritten as press conferences happen. Training on
data pulled today about a past gameweek means using facts that did not exist at that
deadline. Raw data is therefore stored as immutable, deadline-stamped snapshots, and
`tests/test_no_leakage.py` asserts in CI that no feature for gameweek *t* depends on
anything timestamped after deadline *t*.

**Walk-forward validation only.** Train on gameweeks <= t, predict t+1, roll forward.
A random train/test split lets the model learn from March to predict September.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
fpl bootstrap --verbose      # download historical seasons into data/bronze
```

## Data sources

| Source | Provides | Access |
| --- | --- | --- |
| [Official FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) | Points, prices, minutes, xG/xA, set-piece order, injuries | Public JSON |
| [vaastav archive](https://github.com/vaastav/Fantasy-Premier-League) | Historical seasons, 2016-17 onward | GitHub |
| [Understat](https://understat.com) | Shot-level xG, npxG, xGChain | Scrape |
| [football-data.co.uk](https://www.football-data.co.uk/englandm.php) | Historical closing bookmaker odds | Static CSV |
| [ClubElo](http://clubelo.com/API) | Dated team-strength ratings | CSV API |
| [The Odds API](https://the-odds-api.com) | Live pre-deadline odds | REST, free tier |

All free. FotMob is deliberately excluded: its terms forbid automated retrieval.

## Licence

MIT

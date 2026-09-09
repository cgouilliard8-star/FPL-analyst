# Publishing the site and turning the daily refresh on

Everything below is done once. After it, the site rebuilds itself every day (and
every six hours in the last day and a half before a deadline) and the data the model
needs accumulates on its own.

## 1. Put the repository on GitHub

The repository is a normal git repo with no remote yet. Nothing in it is secret
(`.env` is ignored; the API key lives in GitHub's own secret store, added in step 3).
GitHub Pages is free for **public** repositories; a private repository needs a paid
plan, so public is the assumption here.

The quickest route is GitHub's own command-line tool, which handles the login in a
browser window and creates the repository and the remote in one command:

```bash
brew install gh                 # once, if you don't have it
gh auth login                   # choose GitHub.com -> HTTPS -> login with a browser

cd ~/Documents/fpl-analyst
gh repo create fpl-analyst --public --source=. --remote=origin --push
```

Without `gh`: create an empty repository at <https://github.com/new> (no README, no
.gitignore), then

```bash
cd ~/Documents/fpl-analyst
git remote add origin https://github.com/<your-username>/fpl-analyst.git
git push -u origin main
```

## 2. Let the workflow publish and commit

In the repository on github.com:

* **Settings -> Pages -> Build and deployment -> Source: GitHub Actions.**
  (Not "Deploy from a branch" -- the workflow uploads the site itself.)
* **Settings -> Actions -> General -> Workflow permissions: Read and write.**
  The refresh commits the rebuilt `live.json`, the availability log and the
  projection log back to the repository, so the record advances by itself.

## 3. Add the API key (optional)

**Settings -> Secrets and variables -> Actions -> New repository secret**, named
`ANTHROPIC_API_KEY`. It is only used to write the one-line rationales under the
recommended eleven; without it the refresh still succeeds and a template sentence is
used instead.

## 4. Run it once by hand

**Actions -> refresh -> Run workflow.** The first run takes about fifteen minutes
because it downloads and builds the historical seasons; later runs reuse a cache and
take three or four. When it finishes, the site is at

```
https://<your-username>.github.io/fpl-analyst/
```

## What happens from then on

The workflow wakes every six hours. A cheap first step (`fpl due`) lets it go on
only within 36 hours of the next deadline, or once a day as a heartbeat, or when you
run it by hand. A run that goes on:

1. runs the test suite -- a refresh from broken code is not a refresh;
2. snapshots the FPL API (immutable, timestamped) and appends every player's injury
   flag, news and price to `data/external/availability_log.csv`;
3. fetches cup and European fixtures, and the bookmaker odds;
4. refits the model and rebuilds `site/data/live.json`;
5. writes the coming gameweek's projections to `data/external/projections/` **before
   the deadline**, so the live model can be scored honestly afterwards;
6. validates the payload (`fpl check`) and, if anything is wrong, republishes the
   last good copy instead of a broken one;
7. commits the refreshed data and deploys the site.

The three logs are the point of turning this on: they are the data the model is
currently missing, and they cannot be recovered later. The **Backtest** tab shows the
live scorecard growing from the projection log, gameweek by gameweek, next to the two
replays.

## Checking on it

* **Actions** tab: every run, green or red. A red run leaves the site as it was.
* The page itself says when the data was captured and warns if it is more than 36
  hours old, so a silently stalled refresh is visible from the site.

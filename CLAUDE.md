# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What we're building

**Transfer Edge** — a decision intelligence platform for football clubs covering the full player transfer lifecycle: find undervalued talent, price players accurately, optimise squad composition, and time sales for peak return. Think of it as a quant fund applied to football transfers, delivered as software.

Starting point: La Liga. Expand to other leagues once validated.

**MVP definition:** a Streamlit app where you can search any La Liga player, see their predicted market value with confidence interval, see a SHAP bar chart explaining the valuation, see their undervalue score vs Transfermarkt, and browse a leaderboard of the most undervalued players by position.

## Repository Layout

```
README.md    Setup, usage, layout
CLAUDE.md    This file (AGENTS.md is a symlink to it)
docs/        roadmap.md (positioning, GTM), handoff.md (state + next steps)
notebooks/   01_scrape_transfermarkt.ipynb → 02_scrape_understat.ipynb
             → 03_eda_dataset.ipynb → 04_train_model.ipynb
             05_scrape_fbref.ipynb (hand-saved pages; run after 02, before training)
data/        *.parquet  (gitignored — regenerate from notebooks)
models/      xgb_model.pkl  (gitignored — regenerate with src/models/train.py)
outputs/     pred_vs_real.png, shap_summary.png (gitignored)
src/
  data/      loader.py       — load_panel(), load_players(), load_understat(), load_predictions()
  features/  engineer.py     — build_features(), select_features(), temporal_split(),
                               FEATURE_COLS, PEAK_AGES, compute_undervalue_score()
  models/    train.py        — reproducible training entrypoint
             tune.py         — Optuna search, walk-forward CV → models/best_params.json
             predict.py      — load_model(), predict(), explain()
             valuation.py    — score_panel(), top_undervalued(), test_mape() (the app's data layer)
app/         main.py         — Streamlit app: player search (valuation, value history, full SHAP) + leaderboard
```

There is no build system. Order: notebook 01 → 02 → 05 → `python -m src.models.train`. `load_panel()` joins `data/fbref_matched.parquet` (notebook 05) onto `panel_final.parquet` when it exists.

```bash
# Install dependencies
pip install -r requirements.txt

# Re-tune hyperparameters (~10 min, 100 trials; writes models/best_params.json)
python -m src.models.tune

# Retrain the model (writes models/xgb_model.pkl + data/test_predictions.parquet)
python -m src.models.train

# Run Streamlit app
streamlit run app/main.py
```

`notebooks/03_eda_dataset.ipynb` is read-only exploration of `panel_final.parquet` —
schema, integrity checks, missingness, target shape, panel balance and the
position-aware views. It writes nothing.

`notebooks/04_train_model.ipynb` is kept for exploration and for regenerating
the plots in `outputs/`. It duplicates what `src/models/train.py` does — treat
the module as authoritative and the notebook as a scratchpad. Notebook outputs
are stripped before committing; keep them stripped so diffs stay readable.

**No API layer.** `api/` was removed — Streamlit imports `src/` in-process, so
FastAPI was a layer with no consumer. Re-add it when a second client exists.

## Data Schemas

### `data/players_panel.parquet` — Transfermarkt raw (6,677 rows, 2017–2025)
`year`, `club`, `player_id`, `player_name`, `link_html`, `birth_date` (ISO), `image_url` (portrait, "header" size), `nationality`, `height` (meters), `foot` (right/left/both), `position` (verbose TM string), `market_value` (EUR), `age`, `age_imputed`, `market_value_imputed`

One row per (season, club, player) — mid-season transfers appear twice; notebook 02 collapses them. `birth_date` and `image_url` come from the squad (roster) pages, not the profiles. `value_date` is the snapshot the season's `market_value` came from; `age` is exact, from `birth_date`, as of 30 June of year+1. Gaps are filled in notebook 01:
- **Market value:** midpoint of the nearest listed season before and after — *only* when both exist. Gaps at the edge of a player's record stay NaN and drop out at `build_features`. A one-sided fill would copy a later value into an earlier season (Lamine Yamal's 2022 read EUR 60M at age 15).
- **Age:** exact from `birth_date` as of 30 June of year+1 (the valuation date), else a neighbouring season's age plus the year difference.
- Filled values are **display only**: `market_value_imputed` rows are never a training/test target, never a lag and never in a club median (see Key Gotchas).

### `data/understat_stats.parquet` — Understat raw (5,189 rows, 2017–2025)
`understat_id`, `player_name_us`, `games`, `minutes`, `goals`, `xG`, `assists`, `xA`, `shots`, `key_passes`, `yellow_cards`, `red_cards`, `position_us`, `team_us`, `npg`, `npxG`, `xGChain`, `xGBuildup`, `year`, `name_norm`, `team_norm`

Saved *after* normalisation, so `name_norm` / `team_norm` are really in the file. There is no `match_score` here — that belongs to the merge, not to the raw stats.

### `data/panel_final.parquet` — merged panel (6,513 rows, 2017–2025)
All `players_panel` columns plus `name_norm`, `team_norm`, `match_score`, `understat_id`, `player_name_us`, `team_us`, `position_us`, `games`, `minutes`, `goals`, `xG`, `assists`, `xA`, `shots`, `key_passes`, `yellow_cards`, `red_cards`, `npg`, `npxG`, `xGChain`, `xGBuildup`.

**Exactly one row per (year, player_id)** — players transferred mid-season keep only their last club row. 69–72% of rows have Understat stats for 2017–2024 and 88% for 2025; the rest are NaN, which is correct (Understat only lists players who appeared).

### `data/fbref_stats.parquet` / `data/fbref_matched.parquet` — FBref (notebook 05)
`fbref_stats`: one row per (season, FBref player, club) from the hand-saved `misc` pages — tackles won, interceptions, fouls, fouls drawn, crosses, cards, minutes. `fbref_matched`: aggregated to player-season and matched to the panel's (year, player_id) — `fbref_id`, `fbref_match_score`, `fb_*` counts. ~75% of panel rows match. Matching (`src/data/fbref.py`): per season, one-to-one; when both sides have a birth year it must be equal (then name ≥ 70), otherwise the notebook-02 rule (name ≥ 85, club ≥ 60 for inexact names).

### `data/test_predictions.parquet` — test-set results (~1,945 rows, 2023–2024)
All `panel_final` columns plus: `pred_log` (log-scale prediction), `pred_value` (EUR, back-transformed), `error_pct`

## Scraping Endpoints

### Transfermarkt (`01_scrape_transfermarkt.ipynb`)
- League page (team listing): `https://www.transfermarkt.us/laliga/startseite/wettbewerb/ES1/plus/?saison_id={year}`
- Team page: `https://www.transfermarkt.us/{team-slug}/startseite/verein/{club-id}/saison_id/{year}`
- Player profile: `https://www.transfermarkt.us/{player-slug}/profil/spieler/{player_id}`
- Market value history API (JSON): `https://tmapi-alpha.transfermarkt.technology/player/{player_id}/market-value-history`
  - Returns `data.history[]` with per-snapshot: `marketValue.determined` (date), `marketValue.value` (EUR), `age`, `clubId`
- Uses `User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36`
- Temporal strategy: the season's value is the snapshot at the **end** of the season — June of year+1, else May, else July. Histories are fetched **once per player** into `data/_value_histories.csv` (one request returns every snapshot), not once per player-season.
- Why end of season: Understat stats cover the full season (Aug–May). A December value would be explained by goals scored after it was set.

### Understat (`02_scrape_understat.ipynb`)
- `POST https://understat.com/main/getPlayersStats/` — body: `{'league': 'La_liga', 'season': str(year)}`
- Required headers: `Referer: https://understat.com/`, `X-Requested-With: XMLHttpRequest`, `Content-Type: application/x-www-form-urlencoded`
- `time.sleep(1.5)` between season requests

## Fuzzy Name Matching

Transfermarkt and Understat use different name formats, so `02_scrape_understat.ipynb` fuzzy-joins them per season. `process.cdist` builds the whole score matrix at once; names use `token_sort_ratio` and clubs `token_set_ratio`.

- **One threshold, `MATCH_THRESHOLD = 85`**, applied inside the join. There is no second nullification pass — the old 80-in-the-function / 85-in-a-later-cell split is what let 157 sub-threshold rows reach the saved file.
- **Club corroboration** — a name scoring 85–99 also needs its club to score `TEAM_FLOOR = 60`. An exact name (100) skips that, because a mid-season transfer legitimately puts the two clubs out of step. Calibrated on real matches: everything below 60 was a different player, everything above 78 was correct.
- **One-to-one assignment** — candidates are sorted by name score (club breaks ties) and assigned greedily, each side used at most once. Without this, three different players could inherit one Understat stat line.
- **Name vs club normalisation differ.** Names delete punctuation; clubs turn it into a space, because Understat writes a transferred player's team as `'Getafe,Villarreal'` and deleting the comma would glue it into one token.

~28–31% of players remain unmatched per season. This is intentional — Understat only lists players who actually appeared, and wrong-player stats are worse than NaN.

## Current Model

- **Algorithm:** XGBoost (`models/xgb_model.pkl`)
- **Target:** `log(market_value)` — always back-transform with `np.exp()`
- **Temporal split:** train ≤2022, val 2023 (early stopping), test 2024–2025 (`TRAIN_END`, `VAL_YEAR` in `engineer.py`)
- **Performance (test 2024–2025, 1,306 rows; goalkeepers and interpolated values excluded):** R² 0.825, RMSE(log) 0.703, median APE 31.6% (Ridge baseline 55%). Split 2,985 / 638 / 1,306 (train ≤2022, val 2023). This is the first leakage-free number — earlier figures (24.6%, 27.0%) were flattered by one-sided fills, the player's own value inside `club_med_value`, and a mid-season target. FBref features: RMSE(log) 0.730 → 0.714, MAPE 32.1% → 31.8% (5-seed average), but both 95% bootstrap intervals include zero; the gain comes from fouls/fouls drawn, not from tackles/interceptions (DEF error unchanged).
- **Intervals:** `src/models/intervals.py` — a multi-quantile XGBoost (10/25/50/75/90%) whose 50% and 80% intervals are widened by the error seen on the validation season (conformalized quantile regression). Uncalibrated, its "80%" interval covered only ~60% of test players; calibrated: 78% (50% interval: 51%). Saved as `models/xgb_intervals.pkl`; `score_panel()` adds `low_80/low_50/high_50/high_80`. The app shows and judges by the **50% range** (product choice): undervalued/overvalued when the listed value is outside it — which by construction happens to about half of correctly priced players, so it reads as a lean, not a strong signal. The 80% range is still computed (`low_80/high_80`) but not shown.
- **Hyperparameters:** tuned with Optuna (`src/models/tune.py`), 100 trials, walk-forward CV (train on seasons before Y, validate on Y, Y = 2019–2023; test seasons never touched). `train.py` uses `models/best_params.json` when present, else the hand-set `XGB_PARAMS`. Current best: max_depth 6, learning_rate ≈ 0.012, subsample ≈ 0.66, colsample ≈ 0.57, early stopping on the validation season. CV RMSE(log) 0.718 → 0.701; on test RMSE(log) 0.714 → 0.704 (95% bootstrap interval excludes zero), median APE unchanged within noise — tuning cut the large misses, not the typical error.

### SHAP usage pattern
```python
import shap, pickle, numpy as np

# The pickle is {"model": estimator, "features": [...]} — unwrap it.
# src.models.predict.load_model() does this for you.
bundle = pickle.load(open('models/xgb_model.pkl', 'rb'))
model = bundle["model"]
explainer = shap.TreeExplainer(model)

baseline = np.exp(explainer.expected_value)  # baseline in €
sv = explainer.shap_values(X_player)         # shape: (1, n_features)
pred = np.exp(explainer.expected_value + sv.sum())
```
SHAP values are in log-space. To show per-feature EUR impact, you must work in log-space and only back-transform the final sum. The guarantee `baseline + sum(SHAP) = prediction` holds exactly in log-space.

## Feature Engineering (47 features)

**Position grouping** — TM verbose strings mapped to 4 groups:
- `GK`: Goalkeeper — **excluded from training** (`EXCLUDED_POSITIONS` in
  `temporal_split()`). Understat's outfield stats carry no signal for a
  keeper's value, and GK is the second-smallest group (706 rows) — not enough
  to support its own model yet. `build_features()` still computes every
  feature for GK rows (so the full panel stays usable for EDA); the exclusion
  happens only where training data is assembled. The app cannot score
  goalkeepers until a GK-specific model exists.
- `DEF`: Centre-Back, Full-Back, Wing-Back, **and Defensive Midfield** — DM
  was merged in rather than kept separate or folded into MID. Alone it was
  only 399 rows, too thin for reliable within-position percentiles, and a DM's
  defensive-actions profile sits closer to a centre-back's than to an
  attacking midfielder's.
- `MID`: Central/Attacking Midfield, Wide
- `ATT`: Centre-Forward, Second Striker, Winger

**Age.** Only `age` is a feature. `age2`, `years_to_peak` and `prime_window` were removed: for a tree they are the same information (any split on age² or peak-minus-age is a split on age) and only divided the age effect across several SHAP bars; removing them cost no measurable accuracy (test RMSE(log) +0.002, 95% CI across zero). `fit_peak_ages()` (training seasons only) and the `PEAK_AGES` priors remain for the EDA's age-curve view.

| Group | Features |
|---|---|
| Personal | `age` (the only age feature), `height`, `foot_enc` (right=1, left=2, both=3) |
| Per-90 stats | `minutes`, `games`, `goals_p90`, `xG_p90`, `assists_p90`, `xA_p90`, `shots_p90`, `key_passes_p90`, `npxG_p90`, `xGChain_p90`, `xGBuildup_p90` |
| Engineered | `goals_minus_xG`, `assists_minus_xA` (finishing quality); `xG_p90_pct`, `xA_p90_pct`, `npxG_p90_pct`, `minutes_pct` (percentile *within same year+position group*); `position_versatility` (distinct positions in `position_us`, `S` excluded); `club_med_value` (median of the player's teammates' values that season — never his own), `club_tier` (its quintile per year) |
| FBref | `tackles_won_p90`, `interceptions_p90`, `fouls_p90`, `fouled_p90`, `crosses_p90` (over FBref's own minutes), `def_actions_p90_pct` (tackles+interceptions per 90, percentile within year+position) — NaN without an FBref match |
| Lag | `log_value_lag1`, `xG_p90_lag1`, `goals_p90_lag1`, `assists_p90_lag1`, `minutes_lag1`, `value_growth` — **only filled when `year_diff == 1`** (consecutive seasons for the same player); NaN otherwise |
| One-hot | Position group (DEF/MID, ATT dropped as reference; no GK column — GK rows never reach training); top-10 nationalities + Other + Spain (Argentina dropped) |

## Key Gotchas

- **No leakage — every feature must be knowable when the target is set.** The rules, all enforced in `src/features/engineer.py`:
  - target = end-of-season value, so full-season stats precede it;
  - `club_med_value` = median of the player's *teammates*, never his own value;
  - interpolated values (built from the next season) never reach the model — `temporal_split()` drops them, `log_value_lag1` is NaN after one, club medians skip them;
  - anything fitted from the target (e.g. `fit_peak_ages()`, EDA only) sees training seasons (≤ `TRAIN_END`) only.
  Adding a feature? Ask whether it could be computed on the valuation date without the player's own value.

- **Write parquet with the same environment that reads it.** pyarrow >= 20 stores size-statistics histograms that pyarrow < 20 cannot parse — reading such a file fails with `OSError: Repetition level histogram size mismatch`. No writer option (`version`, `write_page_index`) suppresses them, so the only fix is to regenerate the file with the older pyarrow. This project runs on the Anaconda env (`/opt/anaconda3`, pyarrow 19); don't regenerate `data/*.parquet` from a different interpreter.

- **Lag features are NaN for a player's first season** and any year after a gap. Don't backfill.
- **Percentile features are within-group** (same year + position). A 0.5 xG/90 means very different things for a striker vs a keeper.
- **Club tier is recomputed each season** — reflects current squad strength, not historical reputation.
- **SHAP is in log-space** — never show raw SHAP values as EUR; always work through the log-space formula.
- **~28–31% missing Understat stats per year** — keep them as NaN. XGBoost learns a direction for missing values natively, and the missingness is *informative*: matched players are worth ~4x the median of unmatched ones, because Understat only lists players who appeared. Median-imputing (as the Ridge baseline does) erases that signal.
- **`position_versatility` is NaN without an Understat match, 0 for sub-only players, and constant 1 for goalkeepers** — it carries no signal for GK.

## Four Product Modules

### 1. Scout — find undervalued talent (MVP priority)
- **Undervalue score:** `(predicted_value - market_value) / market_value` — computable from existing data today
- Performance percentile rankings vs position peers (not whole dataset)
- "Hidden gem" leaderboard sorted by undervalue score per position
- Similar player finder (cosine similarity on performance feature vectors)

### 2. Value — fair market pricing (MVP priority)
- Transfer fee model (existing XGBoost, needs upgrades — see below)
- Confidence intervals: output "€24–31m (median €27m)", never a point estimate alone
- Salary benchmarking (requires Capology scraper, not yet built)

### 3. Fit — squad optimisation (post-MVP)
- Tactical system match score, age-balance audit, depth chart gap identification

### 4. Sell — peak sale timing (post-MVP)
- Value trajectory at +6m/+12m/+18m, contract cliff alerts (<18 months remaining)

## Build status

MVP is built: undervalue score, calibrated intervals, FBref basics, Streamlit app (search,
valuation with ranges, value history, full SHAP, leaderboard). Still open from the original
plan: position-specific models (untested — see handoff), FastAPI (deferred: no consumer).
**Prioritised next steps and the decisions already made live in [docs/handoff.md](docs/handoff.md).**

## Data Pipeline Gaps

- No scheduled refresh (currently one-off scrapes); needs weekly cron via APScheduler or GitHub Actions
- FBref is behind a Cloudflare challenge that blocks automated fetching — notebook 05 reads pages saved by hand into `data/_fbref/html/` (`{year}_{stat}.html`, 'Webpage, HTML only'). It checks each page's season and drops all-empty columns. Output `data/fbref_stats.parquet` is not yet joined into the panel.
- No Capology scraper yet (salary data — needed for wage/value ratio feature)

## Tech Stack

| Layer | Choice |
|---|---|
| Data pipeline | Python + `requests` + `BeautifulSoup` |
| Storage | Parquet → PostgreSQL once multi-user |
| Model | XGBoost + SHAP |
| API | FastAPI |
| Frontend MVP | Streamlit |
| Frontend V1 | React + Recharts |
| Scheduling | APScheduler or GitHub Actions cron |
| Deployment | Railway or Render → AWS later |

## Key Design Principles

1. **Explainability first** — every number needs a "why." SHAP bars are the mechanism. Black boxes won't be trusted by clubs.
2. **Confidence intervals always** — never show a point estimate alone. "€24–31m" is usable; "€27m" is not.
3. **Position-aware** — always benchmark players against position peers, never the full squad.
4. **Financial layer** — contract duration, wage/value ratio, FFP headroom — pure sports analytics tools miss this.
5. **La Liga first** — depth over breadth at MVP stage.

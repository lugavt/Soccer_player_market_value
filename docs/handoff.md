# Handoff — Transfer Edge (La Liga)

Snapshot of 2026-09-25. What exists, what was decided and why, and what to do next.
Technical reference (schemas, features, gotchas) is in [CLAUDE.md](../CLAUDE.md).

## Where things stand

| Piece | State |
|---|---|
| Transfermarkt scraper (notebook 01) | Done — rosters, photos, birth dates, profiles, per-player value histories; all cached |
| Understat scraper + join (notebook 02) | Done — one-to-one fuzzy join, name + club |
| FBref (notebook 05) | Done, **basics only** — hand-saved `misc` pages; FBref lost Opta's advanced data |
| EDA (notebook 03) | Done, current |
| Features (`src/features/engineer.py`) | 47 features, leakage rules enforced in code |
| Model (`src/models/train.py`, `tune.py`) | XGBoost, Optuna-tuned, walk-forward CV |
| Intervals (`src/models/intervals.py`) | Calibrated 50% / 80% ranges (conformalized quantile regression) |
| App (`app/main.py`) | Done — player search (current season), valuation + 50% range, value history, full SHAP, leaderboard |
| Tests / CI | None |
| Deployment | None — runs locally |
| Version control | **Nothing since the restructure is committed** |

## Model card

- **Target:** log of the Transfermarkt value at the **end** of each season (June of year+1).
- **Rows:** outfield player-seasons, 2017–2025. Goalkeepers excluded; DM folded into DEF;
  interpolated values never used.
- **Split:** train ≤2022 (2,985), validate 2023 (638), test 2024–2025 (1,306).
- **Test:** R² 0.825, RMSE(log) 0.703, median absolute % error 31.6% (Ridge baseline 55%).
- **Intervals on test:** the 80% range holds the listed value 78% of the time, the 50% range 51%.
  Wide by nature — median 80% range spans ~4.7× from low to high. Weakest for the most
  expensive quarter of players (~72% coverage).
- **Known blind spot:** young breakout players priced on hype (Lamine Yamal: model ~€126M vs
  €200M listed). No performance feature captures it.

## Decisions already made — don't reopen without new evidence

| Decision | Why |
|---|---|
| End-of-season target, not December | Full-season stats must precede the value they explain |
| Value gaps filled only two-sided, and never modelled | A one-sided fill copied future values backwards (Yamal 2022 read €60M) |
| `club_med_value` excludes the player | His own value inside a feature is target leakage |
| `age` is the only age feature | age², years-to-peak and prime-window were the same information for trees; they only split the SHAP bar |
| Keep last season's value (`log_value_lag1`) | Removing it: MAPE 32% → 44%; its "undervalued" list filled with aging veterans, not gems |
| Keep FBref features | Small gain (RMSE 0.730 → 0.714), CI touches zero; costs one hand-saved page a season |
| Optuna params | CV RMSE −2.5%; test RMSE −1.4% with CI below zero; typical error unchanged |
| No FastAPI layer | Nothing but Streamlit consumes predictions yet |

## Next steps, in priority order

1. **Commit and push.** Everything since the restructure lives only on this disk.
   GitHub SSH auth fails on this machine: add a key (`ssh-keygen`, then GitHub → Settings → SSH
   keys) or switch the remote to HTTPS. Commit in logical chunks: restructure, pipeline, model,
   app. *Effort: 30 min.*
2. **Contract length** — the biggest price driver the model can't see (a player with 18 months
   left and one with 4 years left are different assets). Start with the *current* contract:
   Transfermarkt profiles show "Contract expires", one `info_table_value()` label in notebook 01
   — right for the current season and for the Sell module. Historical contracts (for training)
   need the Wayback Machine or Football Manager databases; probe coverage on ~50 players first.
   *Effort: S for current, L for historical.*
3. **Leaderboard by confidence.** Rank by how far the listed value sits *below the range*, not by
   the raw ratio. The app uses the 50% range (about half of players fall outside it by design);
   the 80% range (`low_80`) is available for a stricter "strong signal" filter. *Effort: S.*
4. **Per-position models, as an experiment.** Error is 29% for DEF vs 34% for MID/ATT. Train one
   model per group with the same walk-forward CV and compare against the joint model on test —
   adopt only if the bootstrap interval clears zero. *Effort: S–M.*
5. **Similar-player finder** (Scout module): cosine similarity on within-position per-90
   percentiles, shown on the player page. *Effort: M.*
6. **Goalkeeper model.** Save one FBref `keepers` page (not `keepersadv`) and check that saves,
   save % and clean sheets are still filled. If so, a small GK-only model brings keepers back.
   *Effort: M.*
7. **Tests and CI.** Encode the invariants that bit us: one row per (year, player_id); every
   Understat/FBref id used once per season; no imputed row in any split; `club_med_value` never
   contains the player's value; test coverage of the 80% range ≥ 75%. *Effort: M.*
8. **Deploy the app** (Streamlit Community Cloud or Render). The parquets and pickles are
   gitignored, so ship a small app snapshot (current-season scores + model) or add a build step.
   *Effort: M.*
9. **Optional — a "performance-only" lens** in the app: the valuation without last season's
   value, shown next to the main one. A different question (what do his stats alone justify?),
   not a better answer.
10. **Before any commercial use: licensing.** Transfermarkt data and photos (IMAGO), Understat,
    and FBref's Opta-sourced stats are fine for a prototype, not for a product sold to clubs.

## Yearly update — every June, after the season ends

1. Add the new season to `YEARS` in notebooks 01, 02 and 05; delete `data/_rosters.parquet`
   and `data/_value_histories.csv` (values change every window).
2. Save one FBref page: `{year}_misc.html` into `data/_fbref/html/`.
3. Run notebook 01 → 02 → 05, then `python -m src.models.tune` (optional) and
   `python -m src.models.train`.
4. Move `TRAIN_END` / `VAL_YEAR` in `engineer.py` forward one season.
5. Check the train output: test R² and interval coverage close to the model card above.

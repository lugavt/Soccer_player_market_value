# Transfer Edge

Decision intelligence for football transfers, starting with La Liga. Predicts a
player's market value from performance and personal data, explains every
prediction with SHAP, and scores players against their Transfermarkt valuation
to surface undervalued talent.

Think of it as a quant fund applied to football transfers, delivered as software.

## Status

Data, model and a first version of the app all work.

| Layer | State |
|---|---|
| Scrapers (Transfermarkt, Understat) | Done — notebooks 01 and 02 |
| Feature pipeline (47 features) | Done — `src/features/engineer.py` |
| Model (XGBoost, Optuna-tuned, R²=0.825, MAPE=31.6%, leakage-free) | Done — `src/models/train.py` |
| Prediction + SHAP explanations | Done — `src/models/predict.py` |
| Streamlit app | Done — `streamlit run app/main.py` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or use Docker, below
pip install -r requirements.txt
```

The `data/` and `models/` directories ship empty — parquet files and the
pickled model are gitignored because of their size. Regenerate them by running
the notebooks in order:

1. `notebooks/01_scrape_transfermarkt.ipynb` → `data/players_panel.parquet`
2. `notebooks/02_scrape_understat.ipynb` → `data/understat_stats.parquet`,
   then fuzzy-matches the two into `data/panel_final.parquet`
3. `notebooks/03_eda_dataset.ipynb` → explore the dataset (read-only)
4. `notebooks/04_train_model.ipynb` → modelling exploration and plots
5. `notebooks/05_scrape_fbref.ipynb` → FBref stats from hand-saved pages (run after 02, before training)

Then train reproducibly, without touching Jupyter:

```bash
python -m src.models.train
```

That writes `models/xgb_model.pkl` and `data/test_predictions.parquet`.

## Using the model

```python
from src.data.loader import load_panel
from src.features.engineer import build_features, compute_undervalue_score
from src.models.predict import load_model, predict, explain

df = build_features(load_panel())     # build on the FULL panel — see note below
model = load_model()

player = df[df["player_name"] == "Nico Williams"].nlargest(1, "year")
predict(model, player)                # → array([<EUR>])
explain(model, player)                # → baseline, per-feature SHAP, prediction
```

Two things that will bite you if you skip them:

- **Build features on the whole panel, then filter.** Percentile ranks and club
  tier are computed relative to the rest of the league. Running
  `build_features()` on one player's row yields a 100th-percentile, tier-0
  player with no error raised.
- **SHAP values are in log-space.** `baseline + sum(shap) == prediction` holds
  exactly in log-space and not after `np.exp()`. Never label raw SHAP values as
  EUR without doing the conversion.

## Run with Docker

One image covers the app, the training jobs and the notebooks, with the same
package versions as the local environment (`requirements.txt` is pinned).
`data/` and `models/` are mounted, not copied, so Docker and a local Python
read and write the same files.

```bash
docker compose up --build                       # app -> http://localhost:8501
docker compose run --rm train                   # retrain (writes models/, data/)
docker compose run --rm tune                    # re-run the Optuna search (~10 min)
docker compose --profile notebooks up jupyter   # notebooks; the URL with its token is in the log
docker compose down                             # stop everything
```

Rebuild with `--build` after changing code. Notebook 05's hand-saved FBref pages
live in `data/_fbref/html/`, so they are visible inside the container too.

## Deploy (private Hugging Face Space)

`deploy/` holds an app-only image with the data and models baked in (~28 MB of
artifacts), because a Space has no mounted `data/` folder.

```bash
hf auth login                                   # once: a token with write access
python deploy/build_space.py                    # assemble build/hf_space/ and check it
docker build -t transfer-edge-space build/hf_space && \
  docker run --rm -p 7860:8501 transfer-edge-space   # optional local test -> :7860
python deploy/build_space.py --push             # upload as a PRIVATE Space
```

The Space is always created (or reset to) private: the data is not licensed for
redistribution. To give other people access, put the Space under a Hugging Face
organisation (`--repo my-org/transfer-edge`) and add them as members. Re-run
`--push` after retraining to redeploy.

## Run the app

```bash
streamlit run app/main.py
```

Two tabs. **Player search** (current season): valuation against the listed value
with a calibrated 50% range, value history, season stats with percentiles against
positional peers, and the full SHAP breakdown. **Leaderboard**: most undervalued
players by position, with a minimum-value filter.

After changing anything in `src/`, restart the app — Streamlit reloads
`app/main.py` on its own, but not the modules it imports.

## Layout

```
docs/handoff.md    State of the project, decisions already made, next steps
docs/roadmap.md    Positioning, go-to-market, MVP definition of done
notebooks/         01 scrape TM → 02 Understat + merge → 05 FBref → 03 EDA → 04 train
src/data/          Parquet loaders, FBref matching
src/features/      Feature engineering (leakage rules live here)
src/models/        Training, tuning, prediction, SHAP, intervals, app scoring
app/               Streamlit app (+ assets/favicon.png)
Dockerfile, docker-compose.yml   One image for app, jobs and notebooks
```

## Docs

- **[docs/roadmap.md](docs/roadmap.md)** — the four product modules, competitive
  positioning, go-to-market.
- **[docs/handoff.md](docs/handoff.md)** — current state, model card, decisions
  already made (and why), prioritised next steps, yearly update runbook.

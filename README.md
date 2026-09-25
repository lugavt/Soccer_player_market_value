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
| Feature pipeline (50 features) | Done — `src/features/engineer.py` |
| Model (XGBoost, Optuna-tuned, R²=0.825, MAPE=31.6%, leakage-free) | Done — `src/models/train.py` |
| Prediction + SHAP explanations | Done — `src/models/predict.py` |
| Streamlit app | Done — `streamlit run app/main.py` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
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

## Run the app

```bash
streamlit run app/main.py
```

Currently a title and two TODOs. [docs/handoff.md](docs/handoff.md) has a
screen-by-screen build plan.

## Layout

```
CLAUDE.md          Technical reference — schemas, endpoints, model, gotchas
AGENTS.md          Symlink to CLAUDE.md
docs/roadmap.md    Positioning, go-to-market, MVP definition of done
docs/handoff.md    Where the project stands and how to build the app next
notebooks/         01 scrape TM → 02 scrape Understat + merge → 03 EDA → 04 train
src/data/          Parquet loaders
src/features/      Feature engineering, undervalue score
src/models/        Training, prediction, SHAP
app/               Streamlit MVP
```

## Docs

- **[CLAUDE.md](CLAUDE.md)** — data schemas, scraping endpoints, fuzzy-matching
  rules, feature definitions, and the gotchas list. Read this before changing
  the pipeline.
- **[docs/roadmap.md](docs/roadmap.md)** — the four product modules, competitive
  positioning, go-to-market.
- **[docs/handoff.md](docs/handoff.md)** — current state and the concrete next
  steps for the Streamlit MVP.

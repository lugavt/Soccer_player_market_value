"""Draw the Transfer Edge file flow to docs/pipeline.png.

    python docs/pipeline_diagram.py

Matplotlib only (no Graphviz binary needed). Boxes are placed by hand: sources
on the left, the code that processes them in the middle, the files it writes on
the right. A dashed border marks what V2 adds.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

OUT = Path(__file__).with_name("pipeline.png")

COLORS = {               # fill, edge
    "source": ("#E8F1FB", "#2F6DB5"),
    "code": ("#FFF4E0", "#C77C0E"),
    "data": ("#E9F6EC", "#2E8B47"),
    "model": ("#F3E9FB", "#7A3DB8"),
    "app": ("#FDE8E8", "#B83232"),
}
LEGEND = {
    "source": "Web source",
    "code": "Notebook / Python module",
    "data": "Data file (data/)",
    "model": "Model / MLOps (models/, mlruns/)",
    "app": "App / deploy",
}

fig, ax = plt.subplots(figsize=(17, 22))
ax.set_xlim(0, 17)
ax.set_ylim(-1.3, 22.2)
ax.axis("off")
boxes = {}


def box(key, x, y, w, h, kind, title, lines=(), new=False):
    """A rounded box centred on (x, y): bold title, then smaller description lines."""
    fill, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=fill, edgecolor=edge, linewidth=2.2 if new else 1.6,
        linestyle=(0, (5, 3)) if new else "-",
    ))
    ax.text(x, y + h / 2 - 0.3, ("★ " if new else "") + title, ha="center", va="top",
            fontsize=11, fontweight="bold", color="#1F1F1F")
    for i, line in enumerate(lines):
        ax.text(x, y + h / 2 - 0.72 - i * 0.34, line, ha="center", va="top",
                fontsize=8.6, color="#3A3A3A")
    boxes[key] = (x, y, w, h)


def anchor(key, side):
    x, y, w, h = boxes[key]
    return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}[side]


def arrow(a, b, sa="r", sb="l", label=None, dashed=False, rad=0.0):
    p, q = anchor(a, sa), anchor(b, sb)
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#555555",
        linestyle=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2,
    ))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.15, label, ha="center", va="bottom",
                fontsize=8, style="italic", color="#555555",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))


ax.text(8.5, 22, "Transfer Edge — how the files connect", ha="center", va="top",
        fontsize=18, fontweight="bold")
ax.text(8.5, 21.35, "sources  →  code  →  files it writes        (dashed border = new in V2)",
        ha="center", va="top", fontsize=10.5, color="#555555")

XS, XC, XD = 2.2, 7.6, 13.4          # columns: sources, code, data


def arrow_xy(p, q, label=None, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#555555",
        linestyle=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2,
    ))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, label, ha="center", va="center",
                fontsize=8, style="italic", color="#555555",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))


# ── Config ────────────────────────────────────────────────────────────────
box("leagues", XC, 20.1, 4.6, 1.3, "code", "src/data/leagues.py",
    ["the 5 leagues: names, URLs, country, clubs per season",
     "read by notebooks 01, 02 and 05"], new=True)

# ── Scraping: one source, one notebook, one file ──────────────────────────
ax.add_patch(FancyBboxPatch(
    (10.75, 11.3), 5.3, 7.7, boxstyle="round,pad=0.02,rounding_size=0.2",
    facecolor="#F7F7F7", edgecolor="#BBBBBB", linewidth=1.2, zorder=0))
ax.text(13.4, 11.5, "raw scraped data — one file per source", ha="center", va="bottom",
        fontsize=8.6, style="italic", color="#666666")

box("tm", XS, 17.6, 3.4, 1.6, "source", "Transfermarkt",
    ["squads, profiles,", "market-value API (EUR)"])
box("nb01", XC, 17.6, 4.6, 1.6, "code", "01_scrape_transfermarkt.ipynb",
    ["rosters → profiles → value histories", "resumable checkpoints"])
box("tmfiles", XD, 17.6, 5.0, 2.3, "data", "players_panel.parquet",
    ["1 row per player-season + end-of-season value",
     "_rosters.parquet (squads, photo, birth date)",
     "_enrichment_checkpoint.csv (height, foot, position)",
     "_value_histories.csv (every value snapshot)"])

box("us", XS, 15.0, 3.4, 1.3, "source", "Understat", ["xG API"])
box("nb02", XC, 15.0, 4.6, 1.3, "code", "02_scrape_understat.ipynb",
    ["scrape stats; later merges TM + Understat"])
box("usfiles", XD, 15.0, 5.0, 1.3, "data", "understat_stats.parquet",
    ["goals, xG, xA, shots, minutes…"])

box("fb", XS, 12.7, 3.4, 1.6, "source", "FBref pages",
    ["saved by hand in", "data/_fbref/html/<league>/"])
box("nb05", XC, 12.7, 4.6, 1.3, "code", "05_scrape_fbref.ipynb",
    ["+ src/data/fbref.py: parse the 'misc' table"])
box("fbfiles", XD, 12.7, 5.0, 1.3, "data", "fbref_stats.parquet",
    ["tackles, interceptions, fouls, cards"])

# ── Linking the sources ───────────────────────────────────────────────────
box("map", XS, 9.6, 3.4, 1.5, "source", "Public FBref↔TM map",
    ["worldfootballR_data"], new=True)
box("xwalk", XC, 9.6, 4.6, 1.6, "code", "src/data/crosswalk.py",
    ["deterministic rules, no fuzzy matching", "club table · exact name · ID carry-over"], new=True)
box("xfiles", XD, 9.6, 5.0, 1.6, "data", "id_crosswalk.parquet",
    ["TM id ↔ Understat id ↔ FBref id",
     "manual/id_overrides.csv (human-checked leftovers)"], new=True)

box("panel", XD, 7.1, 5.0, 1.6, "data", "panel_final.parquet",
    ["TM + Understat joined by ids (notebook 02)",
     "fbref_matched.parquet: FBref joined by ids (05)"])

# ── Features and model ────────────────────────────────────────────────────
box("feat", XC, 7.1, 4.6, 1.6, "code", "loader.py + engineer.py",
    ["load panel + FBref; 37 features (+ league)", "leakage rules; train / val / test split"])

box("tune", XS, 4.8, 3.4, 1.3, "code", "src/models/tune.py",
    ["Optuna, 100 trials"])
box("train", XC, 4.8, 4.6, 1.3, "code", "src/models/train.py",
    ["XGBoost + calibrated ranges (intervals.py)"])
box("models", XD, 4.8, 5.0, 1.9, "model", "xgb_model.pkl",
    ["xgb_intervals.pkl (50% / 80% ranges)",
     "test_predictions.parquet",
     "best_params.json (from tune.py)"])

box("mlflow", XC, 2.5, 4.6, 1.6, "model", "MLflow + src/models/promote.py",
    ["every run logged (metrics per league)", "best version gets the 'champion' label"], new=True)
box("docker", XS, 2.5, 3.4, 1.6, "app", "Docker",
    ["runs app, training, tuning", "and notebooks locally"])
box("deploy", XD, 2.5, 5.0, 1.6, "app", "deploy/build_space.py",
    ["bundle → private repo transfer-edge-app", "→ Streamlit Cloud (public URL)"])

box("serve", XC, 0.2, 4.6, 1.6, "code", "predict.py + valuation.py",
    ["predict, SHAP explanation,", "scoring, leaderboard, percentiles"])
box("app", XD, 0.2, 5.0, 1.6, "app", "app/main.py (Streamlit)",
    ["search league → club → player", "undervalued leaderboard"])

# ── Arrows ────────────────────────────────────────────────────────────────
arrow("leagues", "nb01", "b", "t", dashed=True)
for src, nb, out in (("tm", "nb01", "tmfiles"), ("us", "nb02", "usfiles"), ("fb", "nb05", "fbfiles")):
    arrow(src, nb)
    arrow(nb, out)

arrow_xy((11.3, 11.3), anchor("xwalk", "t"), label="players from all 3 sources", rad=-0.15)
arrow("map", "xwalk")
arrow("xwalk", "xfiles")
arrow("xfiles", "panel", "b", "t", label="join by ids")

arrow("panel", "feat", "l", "r")
arrow("feat", "tune", "l", "t", rad=0.25)
arrow("feat", "train", "b", "t")
arrow("tune", "train", label="best params")
arrow("train", "models")
arrow("train", "mlflow", "b", "t", label="log run + register")
arrow("mlflow", "serve", "b", "t", label="champion")
arrow("serve", "app")
arrow("app", "deploy", "t", "b")

ax.text(XS, 0.7, "Read-only, outside the flow:\n03_eda_dataset.ipynb — explore the data\n"
        "04_train_model.ipynb — modelling experiments", ha="center", va="top",
        fontsize=8.6, color="#555555", linespacing=1.5)

handles = [Patch(facecolor=COLORS[k][0], edgecolor=COLORS[k][1], label=v) for k, v in LEGEND.items()]
handles.append(Patch(facecolor="white", edgecolor="#555555", linestyle="--", label="★ new in V2"))
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.025), ncol=6,
          frameon=False, fontsize=9.5)

fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")

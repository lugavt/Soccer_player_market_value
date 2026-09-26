---
title: Transfer Edge
emoji: ⚽
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
short_description: La Liga player valuation with explainable predictions
---

# Transfer Edge — La Liga player valuation

Look up any outfield La Liga player of the current season: the model's
valuation against the listed market value with a calibrated range, the value
history, season stats against positional peers, and a SHAP breakdown of what
drives the number. The leaderboard ranks the most undervalued players.

**Private demo.** Built from Transfermarkt, Understat and FBref data, which are
not licensed for redistribution — keep this Space private.

Deployed from the project repository with `deploy/build_space.py`.

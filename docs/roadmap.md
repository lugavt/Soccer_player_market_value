# Roadmap & Positioning

Product and business context for Transfer Edge. The technical reference —
schemas, model details, feature definitions, gotchas — lives in
[README](../README.md) and [handoff](handoff.md). This file holds what those leave
out: who we're competing with, how we'd sell it, and what counts as done.

## What "done" looks like for the MVP

A Streamlit app where you can:

- Search any La Liga player
- See their predicted market value with a confidence interval
- See a SHAP bar chart explaining the valuation in plain language
- See their undervalue score vs Transfermarkt
- Browse a leaderboard of the most undervalued players by position

## Competitive positioning

| Competitor | Gap we exploit |
|---|---|
| Wyscout | Video-focused, no financial/valuation layer |
| Transfermarkt | Manual values, no ML, no explainability |
| StatsBomb | Raw data only, no decisions or recommendations |
| Smarterscout | Performance only, no transfer pricing |
| **Us** | Full lifecycle: find → price → fit → sell timing, with SHAP explanations a club can use in a board meeting |

## Go-to-market (after MVP)

- **First target:** mid-table La Liga or Championship clubs — selling clubs with
  limited analytics staff.
- **Price point:** €20–40k/year for a pilot.
- **Lead gen:** publish a public "Transfer Market Efficiency Report" every
  window, ranking clubs by how well they buy and sell against model price. Gets
  picked up by media, read by club analysts.
- **Warm intros:** get 2–3 player agents using the tool for free. They use
  valuation data in contract negotiations and have direct access to sporting
  directors.
- **Key hire:** a football insider co-founder or advisor with club-side
  experience — sporting director, head of recruitment, or analytics lead.

## Build order beyond the MVP

The four product modules (Scout, Value, Fit, Sell) and the week-by-week build
order are tracked in [handoff.md](handoff.md). In short: Scout and Value are
MVP scope, Fit and Sell come after. Calibrated intervals and FBref basics are
done; the nearest milestones are contract length, a backtest of whether
"undervalued" players actually appreciate, and position-specific models.

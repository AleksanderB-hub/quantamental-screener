# Quantamental Stock Screener — CLAUDE.md

## Project Summary

Two-phase stock screening engine for S&P 500 stocks.

- **Phase A (Quantitative):** ✅ Complete — screen stocks using ~40 fundamental/technical rules, feature selection to identify 11 consensus rules, score-based ranking.
- **Phase B (Agentic):** In progress — LLM multi-agent pipeline for qualitative assessment of Phase A candidates (news/sentiment, ESG, management signals).

## Current State

- Main pipeline in `pipeline.py` — backtesting on historical snapshots
- Configuration in `config.py` — all tuneable parameters (dates, ticker sample size, paths, thresholds)
- SQLite feature store: `ml_stock_pipeline.db`
- CSV exports: `ML_Training_Data.csv`, `ML_Testing_Data.csv`
- Screener output: `reports/phase_b_candidates.csv` — stocks scoring ≥ 8, input for Phase B

## Key Scripts

- `pipeline.py` — data extraction, snapshot generation, feature computation
- `features_xgboost.py` — XGBoost feature engineering and training experiments
- `feature_selection.py` — RF importance / SHAP-based feature selection; identified 11 consensus features
- `model_comparison.py` — comparison of XGBoost, Random Forest, Lasso models
- `test_evaluation.py` — evaluation on test set (2025-06-30)
- `screener.py` — applies tiered consensus scoring, outputs `reports/phase_b_candidates.csv`

## Pipeline Architecture

```
yfinance (once per ticker) → Snapshot generation (per ticker × screening date)
  → List 2 batch scoring → ML-ready binary feature matrix
```

Key design: **feature store** — rules are computed as binary features and stored; raw accounting line items are NOT stored.

## Rules Overview

- **List 1:** Absolute binary rules (technicals, valuation, financial health) — computed per ticker
- **List 2:** Industry/universe relative rules — computed in batch, grouped by `(Screening_Date, Sector)`
- **List 3:** Historical streak rules (3yr/2yr/1yr fallback) — computed per ticker

## ML Outcome & Feature Selection

ML experiments (XGBoost, Random Forest, Lasso) all failed to generalise from training to test set. The real deliverable was **feature selection**: identifying which of the ~40 rules carry signal.

**11 consensus features with tiered weighting (max score = 15):**

| Tier | Points | Features |
|------|--------|----------|
| Tier 1 | 2 pts each | `50MA_Gt_200MA`, `Op_Margin_Gt_Hist_Avg`, `PE_Below_Industry`, `Gross_Margin_Above_Industry` |
| Tier 2 | 1 pt each | `ROE_Gt_15_Sustained`, `EPS_Current_Change_Above_Industry`, `Price_Gt_50MA`, `DE_Less_1`, `PE_Bottom_40_Pct`, `PB_Below_Industry`, `Assets_To_Liability_Ratio_Above_Industry` |

Stocks scoring ≥ 8 are passed to Phase B.

## ML Labelling Strategy

- **Feature selection:** Quintile buckets — top 20% = 1, bottom 20% = 0, middle 60% discarded
- **Prediction target:** Cross-sectional percentile rank of 6-month excess return (0.0–1.0, market-neutral)
- **Target variables:** `Forward_6m_Return`, `Forward_6m_Excess_Return`

## Screening Dates

- Training: `2023-12-31`, `2024-06-30`, `2024-12-31`
- Testing: `2025-06-30`

## Key Design Decisions

- **NaN propagation:** Missing data → `NaN`, never default to 0
- **90-day reporting lag:** Financial statements filtered to avoid look-ahead bias
- **No `.info` in training data:** PEG and ROE calculated from statements to avoid look-ahead bias
- **Dynamic fallback:** Growth/streak rules try 3yr → 2yr → 1yr data
- **Survivorship bias:** Acknowledged, accepted for portfolio-project scope

## Tech Stack

- Python, pandas, numpy, yfinance
- SQLite (feature store)
- scikit-learn, XGBoost, SHAP
- Phase B: LangChain/LangGraph, API-based inference (start sequential, graduate to LangGraph if needed)

## Phase B Plan

Agentic live assessment layer for candidates from `reports/phase_b_candidates.csv`.

**Agents:**
1. **News/Sentiment agent** — recent news headlines, analyst sentiment
2. **ESG agent** — environmental, social, governance signals
3. **Management agent** — insider activity, guidance, recent commentary
4. **Synthesis agent** — combines signals into a final qualitative score/narrative

**Architecture:** Start with a simple sequential LangChain chain. Graduate to LangGraph if parallelism or conditional branching is needed. API-based inference initially (no local vLLM).

## Coding Conventions

- **NaN for missing data:** Use `NaN` for genuinely missing or unavailable data. Only default to `0` when a rule is definitively not met (e.g. no dividend paid → dividend growth = 0 is wrong; the rule simply fails → `NaN`).
- **Dynamic fallback:** Historical metrics try 3yr → 2yr → 1yr data windows in order; use the longest available period, propagate `NaN` if none available.
- **All config in `config.py`:** Screening dates, ticker sample size, file paths, thresholds, and any other tuneable parameter must live in `config.py`. Never hardcode these values in `pipeline.py` or elsewhere.

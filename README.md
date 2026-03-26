# Quantamental Stock Screener (Python + SQLite)

## Project Overview

Two-phase stock screening engine for S&P 500 stocks. Phase A is complete — a quantitative rules engine with ML-driven feature selection. Phase B is in progress — an agentic qualitative assessment layer using LLMs.

---

## Phase A: Quantitative Screener (Complete)

### Architecture

```
yfinance (once per ticker) → Snapshot generation (per ticker × screening date)
  → List 2 batch scoring → ML-ready binary feature matrix → Tiered consensus scoring
```

Key design: **feature store** — rules are computed as binary features and stored in SQLite; raw accounting line items are not persisted.

### Rules Engine

- **List 1 — Absolute rules** (technicals, valuation, financial health): computed per ticker
- **List 2 — Industry/universe relative rules**: computed in batch, grouped by `(Screening_Date, Sector)`
- **List 3 — Historical streak rules**: 3yr/2yr/1yr dynamic fallback, computed per ticker

### ML Outcome

XGBoost, Random Forest, and Lasso all failed to generalise from training to test. The real deliverable was **feature selection**: identifying which of the ~40 rules carry predictive signal.

**11 consensus features with tiered weighting (max score = 15):**

| Tier | Points | Features |
|------|--------|----------|
| Tier 1 | 2 pts each | `50MA_Gt_200MA`, `Op_Margin_Gt_Hist_Avg`, `PE_Below_Industry`, `Gross_Margin_Above_Industry` |
| Tier 2 | 1 pt each | `ROE_Gt_15_Sustained`, `EPS_Current_Change_Above_Industry`, `Price_Gt_50MA`, `DE_Less_1`, `PE_Bottom_40_Pct`, `PB_Below_Industry`, `Assets_To_Liability_Ratio_Above_Industry` |

Stocks scoring ≥ 8 are written to `reports/phase_b_candidates.csv` as Phase B inputs.

### Key Scripts

| Script | Purpose |
|--------|---------|
| `pipeline.py` | Data extraction, snapshot generation, feature computation |
| `features_xgboost.py` | XGBoost feature engineering and training experiments |
| `feature_selection.py` | RF importance / SHAP-based feature selection; identified 11 consensus features |
| `model_comparison.py` | Comparison of XGBoost, Random Forest, and Lasso models |
| `test_evaluation.py` | Evaluation on test set (2025-06-30) |
| `screener.py` | Applies tiered consensus scoring; outputs `reports/phase_b_candidates.csv` |
| `config.py` | All tuneable parameters (dates, ticker sample size, file paths, thresholds) |

### Outputs

- `ml_stock_pipeline.db` — SQLite feature store
- `ML_Training_Data.csv`, `ML_Testing_Data.csv` — ML datasets
- `reports/phase_b_candidates.csv` — Phase B candidate stocks (score ≥ 8)

### Screening Dates

- Training: `2023-12-31`, `2024-06-30`, `2024-12-31`
- Testing: `2025-06-30`

---

## Phase B: Agentic Qualitative Assessment (In Progress)

Live assessment layer for candidates from `reports/phase_b_candidates.csv`.

### Planned Agents

1. **News/Sentiment agent** — recent news headlines, analyst sentiment
2. **ESG agent** — environmental, social, governance signals
3. **Management agent** — insider activity, guidance, recent commentary
4. **Synthesis agent** — combines signals into a final qualitative score/narrative

### Architecture

Start with a simple sequential LangChain chain. Graduate to LangGraph if parallelism or conditional branching is needed. API-based inference (Claude) initially — no local model serving.

### Next Steps

1. Set up LangChain/LangGraph project structure and dependencies
2. Implement News/Sentiment agent (news API + LLM summarisation)
3. Implement ESG agent
4. Implement Management agent (insider filings, earnings call transcripts)
5. Implement Synthesis agent — combine sub-scores into a final narrative
6. Wire agents into a sequential chain; test on current `phase_b_candidates.csv`
7. Evaluate whether parallelism or branching warrants migration to LangGraph

---

## Key Design Decisions

- **NaN propagation:** Missing data → `NaN`, never default to 0
- **90-day reporting lag:** Financial statements filtered to avoid look-ahead bias
- **No `.info` in training data:** PEG and ROE calculated from statements to avoid look-ahead bias
- **Dynamic fallback:** Growth/streak rules try 3yr → 2yr → 1yr data windows
- **All config in `config.py`:** No hardcoded parameters in pipeline scripts

## Tech Stack

- Python, pandas, numpy, yfinance
- SQLite (feature store)
- scikit-learn, XGBoost, SHAP
- Phase B: LangChain/LangGraph, Anthropic API

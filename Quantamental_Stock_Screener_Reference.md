# Quantamental Stock Screener — Project Reference

## Project Overview

An automated stock screening engine that combines quantitative rule-based filtering with an agentic AI "live assessment" layer. The project operates in two major phases:

- **Phase A (Quantitative Engine):** Screen S&P 500 stocks using ~40 fundamental/technical rules, use ML to identify which rules actually predict outperformance, then apply a transparent scoring system using the most robust rules.
- **Phase B (Agentic Live Layer):** Take the top-scored stocks from Phase A and run them through an LLM-powered multi-agent pipeline that gathers and synthesises qualitative signals (news, ESG, management changes) for a final investment assessment.

---

## Phase A — Complete Summary

### Pipeline Architecture

The pipeline uses a **feature store design** — rules and ratios are computed in Python and saved as pre-processed binary features. Raw accounting line items are NOT stored in the database.

```
┌──────────────────────────────────────────────────────────────┐
│  DATA COLLECTION (once per ticker via yfinance)              │
│                                                              │
│  stock.history("max")  →  full price history                 │
│  stock.financials      →  income statement (~4 annual rows)  │
│  stock.balance_sheet   →  balance sheet (~4 annual rows)     │
│  stock.cashflow        →  cash flow (~4 annual rows)         │
│  stock.info            →  current snapshot (sector, etc.)    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  SNAPSHOT GENERATION (per ticker × per screening date)       │
│                                                              │
│  1. Slice price history up to screening_date                 │
│  2. Filter financials with 90-day reporting lag              │
│  3. Compute all List 1 + List 3 rules → binary features      │
│  4. Compute raw ratios for List 2 → continuous values        │
│  5. Calculate 6-month forward return + SPY excess return     │
│  6. Output: one feature dict per (ticker, screening_date)    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  LIST 2 SCORING (batch, per screening_date cohort)           │
│                                                              │
│  1. Universe percentile rankings (grouped by Screening_Date) │
│  2. Industry-relative rules (grouped by Screening_Date +     │
│     Sector)                                                  │
│  3. Drop raw columns → ML-ready binary feature matrix        │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  ML ANALYSIS → FEATURE SELECTION → TRANSPARENT SCORING       │
└──────────────────────────────────────────────────────────────┘
```

### Data Scale
- **Training set:** 1,494 rows (S&P 500 × 3 screening dates: 2023-12-31, 2024-06-30, 2024-12-31)
- **Testing set:** 501 rows (S&P 500 × 1 screening date: 2025-06-30)
- **43 binary features** across Lists 1, 2, and 3

### Screening Timeline
- Training: `2023-12-31`, `2024-06-30`, `2024-12-31`
- Validation: `2024-12-31` (used for hyperparameter tuning, then merged into training for final models)
- Testing: `2025-06-30` (6-month forward return available through Dec 2025)

---

## ML Experiment Results

### Model Comparison (3 architectures tested)

```
Model               Val Spearman   Test Spearman   Top20% Hit   Top30% Hit   Top40% Hit
─────────────────────────────────────────────────────────────────────────────────────────
Lasso                  +0.190         -0.070          21.8%        27.2%        39.3%
Random Forest          +0.209         -0.086          17.8%        25.8%        36.8%
XGBoost (tuned)        +0.256         -0.074          18.8%        29.8%        40.8%
Random baseline            —          +0.000          20.0%        30.0%        40.0%
```

**Key finding:** All three models showed promising validation performance but collapsed to random-baseline on the out-of-sample test set. More model complexity did not help — XGBoost (most complex) performed marginally worse than Lasso (simplest) on test data. This is consistent with the limited training data covering a single market regime (2023-2024 bull market) that did not persist into the 2025 test period.

**Conclusion:** With the available data, reliable stock return prediction is not achievable. The ML models were used to identify which features are consistently important (feature selection), not for direct prediction. The final screening uses a transparent rule-counting approach.

### Feature Selection (4 independent methods)

| Method | What it measures | Features identified |
|--------|-----------------|-------------------|
| SHAP (XGBoost) | How each feature pushes predictions | Top 15 ranked by mean absolute SHAP |
| Permutation Importance | Performance drop when feature shuffled | Top 15 ranked by importance |
| Boruta (Random Forest) | Whether feature beats randomised shadow copies | 4 confirmed, 1 tentative |
| Lasso (L1 regression) | Which features survive regularisation | 29 non-zero coefficients |

**Consensus approach:** Features scoring 2+ out of 3 methods (SHAP top-15, Permutation top-15, Boruta confirmed) were selected. Lasso independently confirmed all 11 consensus features (all had non-zero coefficients).

### Lasso Coefficient Sign Analysis

Some "quality/value" features showed **negative** coefficients in the 2023-2024 training period. This reflects regime-specific dynamics (growth/momentum outperformed value in this period), not flawed feature logic:

**Positive coefficients (momentum/trend — worked in training period):**
- `50MA_Gt_200MA` (+0.080), `Op_Margin_Gt_Hist_Avg` (+0.083)
- `EPS_Current_Change_Above_Industry` (+0.020), `Price_Gt_50MA` (+0.010)

**Negative coefficients (quality/value — counter-cyclical in this period):**
- `ROE_Gt_15_Sustained` (-0.061), `PE_Below_Industry` (-0.037)
- `PB_Below_Industry` (-0.016), `DE_Less_1` (-0.004)

**Design decision:** Both momentum and quality/value features are retained in the screener. Stocks passing rules from both categories have both trend confirmation AND fundamental substance — the ideal quantamental combination. The scoring does not use coefficient signs; it counts how many screening criteria a stock meets.

---

## Final Screening System

### Selected Features (11 features, tiered weighting, max score: 15)

**Tier 1 — Boruta confirmed, strongest evidence (2 points each):**

| Feature | Category | What it measures |
|---------|----------|-----------------|
| `50MA_Gt_200MA` | Momentum | Price in sustained uptrend (golden cross) |
| `Op_Margin_Gt_Hist_Avg` | Quality | Operating margins improving vs own history |
| `PE_Below_Industry` | Value | Cheaper than sector peers |
| `Gross_Margin_Above_Industry` | Quality | Stronger competitive position than peers |

**Tier 2 — Consensus across SHAP + Permutation (1 point each):**

| Feature | Category | What it measures |
|---------|----------|-----------------|
| `ROE_Gt_15_Sustained` | Quality | Consistently high profitability (3yr/2yr) |
| `EPS_Current_Change_Above_Industry` | Momentum | Earnings growing faster than sector peers |
| `Price_Gt_50MA` | Momentum | Price above short-term trend |
| `DE_Less_1` | Value | Conservative debt levels |
| `PE_Bottom_40_Pct` | Value | Cheap relative to entire universe |
| `PB_Below_Industry` | Value | Book value discount vs sector peers |
| `Assets_To_Liability_Ratio_Above_Industry` | Value | Stronger balance sheet than peers |

### Scoring Logic
- For each stock: `Total_Score = (Tier 1 rules passed × 2) + (Tier 2 rules passed × 1)`
- NaN treated as not passed (conservative)
- Ranked by Total_Score descending, Tier_1_Score as tiebreaker
- Candidates with Total_Score ≥ 8 feed into Phase B agent layer

### Factor Balance
The 11 features span: Momentum (4 features), Value (5 features), Quality (3 features) — with some overlap. This diversification means the screener doesn't bet on a single factor regime.

---

## Complete Rules Inventory (43 features, 11 selected)

### List 1: Absolute Rules (Binary 0/1)

**Technical Indicators:**
- `Price_Gt_50MA` ✓ SELECTED (Tier 2)
- `50MA_Gt_200MA` ✓ SELECTED (Tier 1)
- `RSI_13W_Gt_RSI_25W`
- `OBV_20D_Positive`

**Valuation:**
- `PEG_01_to_05`
- `PEG_Less_1`
- `PEG_Less_1_5`
- `PB_Less_2`
- `Zero_Dividend`
- `MC_to_CF_Less_3`

**Financial Health:**
- `DE_Less_1` ✓ SELECTED (Tier 2)
- `Cash_Ratio_Gt_1`
- `Cash_Ratio_Improving`
- `FCF_Positive`
- `FCF_Growing`
- `OCF_Gt_NetIncome`
- `ROA_Positive`

### List 3: Historical Streaks (Dynamic 3yr→2yr→1yr fallback)

- `FCF_Growing_Sustained`
- `ROE_Gt_15_Sustained` ✓ SELECTED (Tier 2)
- `Net_Income_Growth_Gt_8pct`
- `Sales_Growth_Gt_RnD_Growth`
- `Op_Margin_Gt_Hist_Avg` ✓ SELECTED (Tier 1)
- `Sales_Growing_Sustained`

### List 2: Industry & Universe Relative

**Universe Percentile Rankings (per screening date):**
- `Market_Cap_Top_30_Pct` / `Market_Cap_Top_25_Pct`
- `PE_Bottom_40_Pct` ✓ SELECTED (Tier 2) / `PE_Bottom_20_Pct`
- `FCF_Top_30_Pct`

**Industry-Relative (per screening date × sector):**
- `PE_Below_Industry` ✓ SELECTED (Tier 1) / `PB_Below_Industry` ✓ SELECTED (Tier 2)
- `Div_Yield_Above_Industry`
- `Margin_Above_Industry` / `Gross_Margin_Above_Industry` ✓ SELECTED (Tier 1) / `Operating_Margin_Above_Industry`
- `ROE_Above_Industry`
- `Debt_To_Assets_Ratio_Below_Industry` / `Long_Term_Debt_To_Equity_Ratio_Below_Industry`
- `Assets_To_Liability_Ratio_Above_Industry` ✓ SELECTED (Tier 2) / `Sales_To_Assets_Ratio_Above_Industry`
- `Growth_Above_Industry` / `EPS_Growth_Above_Industry` / `EPS_Current_Change_Above_Industry` ✓ SELECTED (Tier 2)
- `Margin_Gt_Industry_Sustained`

---

## Known Biases & Limitations

### Data Limitations
- **Survivorship bias:** Using today's S&P 500 membership for all historical quarters
- **Restated financials:** yfinance serves current versions of historical statements
- **yfinance depth:** ~4 years of annual financials; dynamic fallback (3yr→2yr→1yr) handles thin data
- **Single regime:** Training data covers 2023-2024 (bull market); insufficient regime diversity for robust prediction

### Design Mitigations
- **90-day reporting lag:** Financial statements filtered to avoid look-ahead bias
- **No `.info` in training data:** PEG and ROE calculated from statements only
- **NaN propagation:** Missing data → NaN, never defaulted to 0 (except PEG rules where negative EPS growth definitively fails the criterion)
- **Transparent scoring over black-box prediction:** ML identified features; final screening is rule-based and explainable

---

## Phase B Architecture (Next — Agentic Live Layer)

### Overview
Takes the top-scored stocks from the screener (Total_Score ≥ 8) and runs qualitative assessment via LLM-powered agents.

### Proposed Agent Pipeline

```
┌──────────────────┐
│  Screener Output │  (Ranked candidates, score ≥ 8)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  News/Sentiment  │  Agent 1: fetch & summarise recent articles
│  Agent           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ESG Agent       │  Agent 2: retrieve ESG data/reports (RAG)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Management      │  Agent 3: CEO changes, insider trading,
│  Signal Agent    │  executive news
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Synthesis Agent │  Combines quant score + qualitative signals
│                  │  → final ranked assessment
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Final Report    │  Investment thesis per stock
└──────────────────┘
```

### Tech Stack (tentative — to be decided at Phase B start)
- Start simple: LangChain with tools + sequential chain
- Graduate to LangGraph if workflow needs branching/cycles
- Inference: API-based initially (Claude/OpenAI), swap to local vLLM (RTX 4070 Ti SUPER, 16GB) later
- Vector store: FAISS or Chroma for RAG components (ESG reports, etc.)

---

## Tech Stack Summary

### Phase A (Quantitative) — Complete:
- Python, pandas, numpy, yfinance
- SQLite (feature store)
- XGBoost, scikit-learn (Random Forest, Lasso)
- SHAP, Boruta (feature selection)
- Optuna (hyperparameter tuning)

### Phase B (Agentic — next):
- LangChain / LangGraph
- vLLM or API-based LLM inference
- FAISS/Chroma (vector store for RAG)
- sentence-transformers (embeddings)

### Hardware:
- RTX 4070 Ti SUPER (16GB VRAM) — for local inference in Phase B
- Phase A is CPU/disk bound

---

## Repository Structure

```
quantamental-screener/
├── config.py                     # All tuneable parameters and paths
├── pipeline.py                   # Data extraction + feature engineering
├── features_xgboost.py           # XGBoost training + SHAP analysis
├── feature_selection.py          # Boruta + Permutation Importance + consensus
├── model_comparison.py           # Lasso + RF + comparison report
├── test_evaluation.py            # Test set evaluation
├── screener.py                   # Final scoring system
├── data/
│   ├── ml_stock_pipeline.db      # SQLite feature store
│   ├── ML_Training_Regression.csv
│   └── ML_Testing_Regression.csv
├── models/
│   └── xgboost_model.json        # Saved XGBoost model
├── reports/
│   ├── feature_importance.csv    # SHAP + XGBoost Gain rankings
│   ├── feature_consensus.csv     # 4-method consensus table
│   ├── lasso_coefficients.csv    # Lasso feature selection
│   ├── model_comparison.csv      # 3-model comparison
│   ├── screener_results.csv      # Full scored stock list
│   ├── phase_b_candidates.csv    # Shortlist for agent layer
│   ├── shap_feature_importance.png
│   ├── val_predicted_vs_actual.png
│   └── test_predicted_vs_actual.png
├── agents/                       # Phase B (next)
├── .gitignore
├── CLAUDE.md
└── README.md
```

---

## Key Design Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Project direction | Stock screener over Skills Gap Analyzer | Existing codebase, broader skill showcase |
| Database design | Feature store (computed rules), not raw data | ML-ready output, avoids SQL complexity |
| Target variable | Cross-sectional percentile rank (0.0–1.0) | Market-neutral, retains all data |
| Feature selection | 4-method consensus (SHAP, Permutation, Boruta, Lasso) | Robust; no single method is reliable alone |
| Final screening | Transparent scoring, not ML prediction | ML failed to generalise; scoring is explainable and robust |
| Tier weighting | Boruta-confirmed = 2x, consensus-only = 1x | Evidence-proportional weighting |
| Negative Lasso coefficients | Kept features, ignored sign | Signs are regime-specific; features are fundamentally sound |
| Model comparison | Lasso + RF + XGBoost (simple → complex) | Demonstrated complexity doesn't help with limited data |
| NaN handling | Propagate NaN throughout, treat as 0 in scoring | Honest in training, conservative in screening |
| Phase B entry | Simple LangChain first | Beginner-friendly, avoid premature complexity |
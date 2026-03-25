# Quantamental Stock Screener — Project Reference

## Project Overview

An automated stock screening engine that combines quantitative rule-based filtering with an agentic AI "live assessment" layer. The project operates in two major phases:

- **Phase A (Quantitative Engine):** Screen S&P 500 stocks using ~40 fundamental/technical rules, use ML to identify which rules actually predict outperformance, then rank stocks using a trained model.
- **Phase B (Agentic Live Layer):** Take the top-ranked stocks from Phase A and run them through an LLM-powered multi-agent pipeline that gathers and synthesises qualitative signals (news, ESG, management changes) for a final investment assessment.

### Why this project:
- Builds on substantial existing codebase (rules engine, SQLite pipeline, data quality audit)
- Demonstrates breadth: data engineering + ML + agentic AI
- Phase A is a standalone portfolio piece even without Phase B
- Finance domain with practical, demo-able output
- Agentic layer introduces LangChain/RAG naturally where it adds value

---

## Current State

### What's built:
- **`test.py`** — Complete backtesting pipeline, tested on 10 random S&P 500 tickers
- Refactored `get_robust_financials` with new signature: `(ticker_symbol, hist, fin, bs, cash, info, screening_date)`
- `process_historical_snapshot` — handles date slicing, 90-day reporting lag, and forward return calculation
- `run_backtest_pipeline` — main loop fetching data once per ticker, generating snapshots per screening date
- `calculate_list_2_rules` — industry-relative scoring with NaN propagation, grouped by `(Screening_Date, Sector)`
- `audit_extraction` — data quality reporting per dataset
- Dynamic fallback extraction (3yr → 2yr → 1yr) for growth metrics and streak rules
- Consistent NaN propagation (missing data → NaN, not 0)
- Training/testing split: train on 2023-2024 snapshots, test on 2025-06-30

### What's next:
- Apply bug fixes (see Known Issues below)
- Run full S&P 500 extraction
- Feature selection (quintile labels) and prediction model (percentile ranking)

---

## Pipeline Architecture

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
│  OUTPUT: Two datasets                                        │
│                                                              │
│  Training Set (2023-2024):  features + Forward_6m_Return     │
│                             + Forward_6m_Excess_Return       │
│  Testing Set  (2025-06-30): features + Forward_6m_Return     │
│                             + Forward_6m_Excess_Return       │
│                                                              │
│  Stored in: ml_stock_pipeline.db + CSV exports               │
└──────────────────────────────────────────────────────────────┘
```

---

## Complete Rules Inventory

### List 1: Absolute Rules (Binary 0/1)

**Technical Indicators:**
- `Price_Gt_50MA` — Current price > 50-day moving average
- `50MA_Gt_200MA` — 50-day MA > 200-day MA (golden cross)
- `RSI_13W_Gt_RSI_25W` — 13-week RSI > 25-week RSI (momentum)
- `OBV_20D_Positive` — 20-day OBV change is positive (volume confirmation)

**Valuation:**
- `PEG_01_to_05` — PEG ratio between 0.1 and 0.5
- `PEG_Less_1` — PEG ratio < 1.0
- `PEG_Less_1_5` — PEG ratio < 1.5
- `PB_Less_2` — Price-to-book < 2
- `Zero_Dividend` — Dividend yield = 0% (reinvesting earnings)
- `MC_to_CF_Less_3` — Market cap / operating cash flow < 3

**Financial Health:**
- `DE_Less_1` — Debt-to-equity < 1
- `Cash_Ratio_Gt_1` — Cash / current liabilities > 1
- `Cash_Ratio_Improving` — Cash ratio improving year-over-year
- `FCF_Positive` — Free cash flow > 0
- `FCF_Growing` — FCF growing year-over-year
- `OCF_Gt_NetIncome` — Operating cash flow > net income (earnings quality)
- `ROA_Positive` — Return on assets > 0

### List 3: Historical Streaks (Binary 0/1, Dynamic Fallback)

All streak rules use dynamic extraction: try 3-year data first, fall back to 2-year if unavailable.

- `FCF_Growing_Sustained` — FCF growing consecutively (3yr or 2yr)
- `ROE_Gt_15_Sustained` — ROE > 15% sustained (3yr, 2yr, or 1yr)
- `Net_Income_Growth_Gt_8pct` — Net income CAGR > 8% (3yr, 2yr, or 1yr)
- `Sales_Growth_Gt_RnD_Growth` — Sales CAGR > R&D CAGR (matched timeframes)
- `Op_Margin_Gt_Hist_Avg` — Current operating margin > historical average
- `Sales_Growing_Sustained` — Revenue growing consecutively (3yr or 2yr)

### List 2: Industry & Universe Relative (Binary 0/1, computed in batch)

**Universe Percentile Rankings (per screening date):**
- `Market_Cap_Top_30_Pct` / `Market_Cap_Top_25_Pct`
- `PE_Bottom_40_Pct` / `PE_Bottom_20_Pct`
- `FCF_Top_30_Pct`

**Industry-Relative (per screening date × sector):**
- `PE_Below_Industry` / `PB_Below_Industry` — Valuation below sector median
- `Div_Yield_Above_Industry` — Dividend yield above sector median
- `Margin_Above_Industry` / `Gross_Margin_Above_Industry` / `Operating_Margin_Above_Industry`
- `ROE_Above_Industry`
- `Debt_To_Assets_Ratio_Below_Industry` / `Long_Term_Debt_To_Equity_Ratio_Below_Industry`
- `Assets_To_Liability_Ratio_Above_Industry` / `Sales_To_Assets_Ratio_Above_Industry`
- `Growth_Above_Industry` / `EPS_Growth_Above_Industry` / `EPS_Current_Change_Above_Industry`
- `Margin_Gt_Industry_Sustained` — Net margin > sector median for multiple consecutive years

### Raw Ratios (used by List 2, dropped from final ML dataset):
- `Market_Cap_Raw`, `PE_Raw`, `Free_Cash_Flow_Raw`, `PB_Raw`, `Div_Yield_Raw`, `ROE_Raw`
- `Net_Profit_Margin_Raw`, `Net_Profit_Margin_Prev_Raw`, `Net_Profit_Margin_Prev_2_Raw`
- `Gross_Profit_Margin_Raw`, `Operating_Margin_Raw`
- `Sales_3yr_Growth_Raw`, `EPS_3yr_Growth_Raw`, `Current_EPS_Change_Raw`
- `Debt_To_Assets_Ratio_Raw`, `Assets_To_Liability_Ratio_Raw`
- `Long_Term_Debt_To_Equity_Raw`, `Sales_To_Assets_Ratio_Raw`

---

## Labelling Strategy

### Feature Selection Phase: Quintile Buckets
- Rank all stocks by 6-month excess return within each quarterly cohort
- Top 20% → label 1, Bottom 20% → label 0, Middle 60% → discard
- Purpose: clean signal for identifying which rules matter

### Prediction Phase: Cross-Sectional Percentile Ranking
- Full 0.0–1.0 percentile rank of excess returns within each cohort
- Regression target (not classification)
- Market-neutral by construction: median is always 0.5 regardless of bull/bear
- Tree-based models (XGBoost/RF) naturally learn to separate extremes

### Target Variables:
- `Forward_6m_Return` — Raw 6-month stock return
- `Forward_6m_Excess_Return` — Stock return minus SPY return over same window
- Percentile ranking and quintile labels computed from excess return during ML phase

### Time Horizons:
- Primary: 6 months forward (calendar months via pd.DateOffset)
- Secondary: 12 months forward (future comparison)

---

## Screening Timeline

### Training Data:
- `2023-12-31` — Q4 2023 snapshot
- `2024-06-30` — Q2 2024 snapshot
- `2024-12-31` — Q4 2024 snapshot

### Testing Data:
- `2025-06-30` — Q2 2025 snapshot (6-month forward return available through Dec 2025)

### Why these dates:
- yfinance provides ~4 years of annual financials, so screening dates before ~2023 leave very limited historical depth after applying the 90-day reporting lag
- Training dates are spaced 6 months apart to avoid overlapping forward return windows
- Test date is the most recent point where the full 6-month forward return is observable

---

## Known Biases & Limitations

### Survivorship Bias
Using today's S&P 500 membership for all historical quarters. Companies that were dropped or delisted are excluded. Accepted for portfolio-project scope.

### Look-Ahead Bias (Mitigated)
- Financial statements filtered with 90-day reporting lag
- `.info` fallbacks removed for historical training data (PEG ratio and ROE now always calculated from statements)
- `.info` values (sector, etc.) that don't change over time are acceptable

### Restated Financials
yfinance serves current versions of historical statements. Rare in S&P 500; acknowledged.

### yfinance Data Depth
~4 years of annual financials. Dynamic fallback extraction (3yr → 2yr → 1yr) handles cases where fewer years are available after date filtering.

### Industry-Relative Rules at Small Scale
With only 10 tickers per sector, sector medians are unreliable. This resolves at full S&P 500 scale (~50+ stocks per major sector).

---

## Phase A — Remaining Roadmap

### Step 1: Apply Bug Fixes to test.py (documented in conversation)
- Percentile rankings grouped by Screening_Date
- Remove .info look-ahead bias
- Add API throttling
- Add SPY excess return
- Add back PB_Less_2, Zero_Dividend, Sales_Growing_Sustained

### Step 2: Full S&P 500 Extraction
- Run pipeline on all ~500 tickers
- Expected output: ~1,500 training rows (500 × 3 dates), ~500 test rows
- Estimated runtime: ~30-45 minutes with throttling
- Monitor with audit reports

### Step 3: Feature Selection
- Compute quintile labels from Forward_6m_Excess_Return per screening date
- Run RF importance + Boruta or SHAP on quintile-labelled data
- Identify top predictive rules

### Step 4: Prediction Model
- Train XGBoost/RF regression on percentile-ranked excess returns
- Selected features only
- Validate: do model-predicted top quintile stocks actually land in real top quintile?

### Step 5: Live Screening
- Run trained model on test set (2025-06-30 data)
- Output: ranked shortlist for Phase B agent layer

---

## Phase B Architecture (Future — Agentic Live Layer)

### Proposed Agent Pipeline:

```
Phase A Output (ranked shortlist)
        │
        ▼
News/Sentiment Agent  →  fetch & summarise recent articles
        │
        ▼
ESG Agent             →  retrieve ESG data/reports (RAG)
        │
        ▼
Management Signal     →  CEO changes, insider trading,
Agent                    executive news
        │
        ▼
Synthesis Agent       →  combines quant score + qualitative
                         signals → final assessment
        │
        ▼
Final Report          →  investment thesis per stock
```

### Tech Stack (tentative):
- Start simple: LangChain with tools + sequential chain
- Graduate to LangGraph if workflow needs branching/cycles
- Inference: API-based initially, swap to local vLLM (RTX 4070 Ti SUPER) later
- Vector store: FAISS or Chroma for RAG components

---

## Tech Stack Summary

### Phase A (Quantitative):
- Python, pandas, numpy
- SQLite (feature store)
- yfinance (data collection)
- scikit-learn (Random Forest, Boruta)
- XGBoost (prediction model)
- SHAP (model interpretability)

### Phase B (Agentic — future):
- LangChain / LangGraph
- vLLM or API-based LLM inference
- FAISS/Chroma (vector store for RAG)
- sentence-transformers (embeddings)

### Hardware:
- RTX 4070 Ti SUPER (16GB VRAM) — for local inference in Phase B
- Phase A is CPU/disk bound

---

## File Structure (Current → Target)

```
quantamental-screener/
├── test.py                       # Current: main pipeline script
├── ml_stock_pipeline.db          # SQLite: training + testing tables
├── ML_Training_Data.csv          # Exported training features
├── ML_Testing_Data.csv           # Exported testing features
├── missing_data_report_*.csv     # Audit reports
│
├── src/                          # Future: modularised version
│   ├── data_collector.py
│   ├── rules_engine.py
│   ├── snapshot_generator.py
│   ├── labeller.py
│   ├── feature_selector.py
│   └── predictor.py
├── agents/                       # Phase B
│   ├── news_agent.py
│   ├── esg_agent.py
│   ├── management_agent.py
│   └── synthesis_agent.py
├── config.py
├── CLAUDE.md
└── README.md
```

---

## Key Design Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Project direction | Stock screener over Skills Gap Analyzer | Existing codebase, broader skill showcase |
| Database design | Feature store (computed rules), not raw accounting data | Avoids rewriting Python logic as SQL; ML-ready output |
| Feature selection labelling | Quintile buckets (top/bottom 20%) | Clean signal, removes noisy borderline cases |
| Prediction labelling | Cross-sectional percentile rank (0.0–1.0) | Market-neutral, retains all data |
| Time horizon | 6-month primary | Matches fundamental signal horizon |
| Screening frequency | Semi-annual snapshots (3 training + 1 test) | Balances yfinance depth limitations vs sample size |
| Stock universe | S&P 500 (expand later) | Manageable size, good data coverage |
| Data source | yfinance (accept limitations) | Free, sufficient for portfolio project |
| Dynamic extraction | 3yr → 2yr → 1yr fallback for growth/streak rules | Maximises data availability within yfinance's depth |
| NaN handling | Propagate NaN (not default to 0) | Honest missing data for ML; avoids false negatives |
| .info usage | Removed from training data to prevent look-ahead bias | Manual calculations from statements are more reliable |
| Phase B entry point | Simple LangChain first, LangGraph if needed | Beginner-friendly, avoid premature complexity |
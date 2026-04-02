# Quantamental Stock Screener — Project Reference

## Project Overview

A two-phase stock screening engine combining quantitative rule-based filtering with an agentic AI qualitative assessment pipeline.

- **Phase A (Quantitative):** Screens stocks using ~43 binary rules, validates through ML feature selection (4 methods), and applies transparent tiered scoring using 9 consensus features.
- **Phase B (Agentic):** Four-stage hybrid pipeline using Claude API (Stages 1, 3, 4) and local Qwen 7B via vLLM (Stage 2) for web research, sentiment classification, RAG-based synthesis, and personalised advisory.

---

## Phase A — Quantitative Engine

### Pipeline

```
yfinance → 43 binary features → ML feature selection (SHAP + Permutation + Boruta + Lasso)
→ 9 consensus features → tiered scoring → ranked candidates
```

### Data

- Training: ~1,500+ rows (S&P 500 × multiple screening dates)
- Testing: ~500+ rows (held-out screening date)
- Multi-index experiments: S&P 500 + Nasdaq 100 + FTSE 100 + DAX 40
- 43 binary features across Lists 1 (absolute), 2 (industry/universe relative), and 3 (historical streaks)
- Target: cross-sectional percentile rank of 6-month forward excess return (0.0–1.0)

### ML Experiment Results

```
                S&P 500 Only          Multi-Index
              Val     Test          Val     Test
Lasso:       +0.190  -0.070       +0.150  -0.032
RF:          +0.209  -0.086       +0.144  -0.043
XGBoost:     +0.256  -0.074       +0.309  -0.110
```

All models failed to generalise — regime-specific training data, not model choice. ML used for feature selection only; final screening uses transparent rule-counting.

### Feature Selection (4 methods, cross-validated across multi-index)

| Method | What it measures |
|--------|-----------------|
| SHAP (XGBoost) | How each feature pushes predictions |
| Permutation Importance | Performance drop when feature shuffled |
| Boruta (Random Forest) | Whether feature beats randomised shadow copies |
| Lasso (L1 regression) | Which features survive regularisation |

### Final Screening Features (9 features, max score 13)

**Tier 1 — Stable across S&P 500 and multi-index (2 points each):**

| Feature | Category |
|---------|----------|
| `50MA_Gt_200MA` | Momentum — sustained uptrend |
| `Op_Margin_Gt_Hist_Avg` | Quality — improving operations |
| `EPS_Current_Change_Above_Industry` | Momentum — earnings beat peers |
| `Gross_Margin_Above_Industry` | Quality — competitive advantage |

**Tier 2 — Score=2 in multi-index consensus (1 point each):**

| Feature | Category |
|---------|----------|
| `Assets_To_Liability_Ratio_Above_Industry` | Value — balance sheet strength |
| `Zero_Dividend` | Quality — reinvesting earnings |
| `Operating_Margin_Above_Industry` | Quality — operating efficiency |
| `FCF_Growing_Sustained` | Quality — consistent cash flow growth |
| `PEG_01_to_05` | Value — attractively priced for growth |

### Feature Stability: S&P 500 → Multi-Index

Features that held: `50MA_Gt_200MA`, `Op_Margin_Gt_Hist_Avg`, `EPS_Current_Change_Above_Industry` (promoted to Score=3), `Gross_Margin_Above_Industry`, `Assets_To_Liability_Ratio_Above_Industry`.

Features that dropped: `PE_Below_Industry` (Tier 1 → Score=0), `PB_Below_Industry` (Score=0), `ROE_Gt_15_Sustained` (Score=1), `DE_Less_1` (Score=1), `PE_Bottom_40_Pct` (Score=1), `Price_Gt_50MA` (Score=1). These were regime/market-specific.

New features emerged: `Zero_Dividend`, `Operating_Margin_Above_Industry`, `FCF_Growing_Sustained`, `PEG_01_to_05`.

### Multi-Index Design

**Percentile rankings** (Market Cap, PE, FCF): grouped by `(Index_Source, Screening_Date)` — relative within each index.

**Industry-relative rules** (all sector median comparisons): grouped by `(Screening_Date, Sector)` globally — larger peer groups, more meaningful medians.

**Deduplication:** stocks appearing in multiple indices are scored in both, then deduplicated in the screener by keeping the highest score. `Index_Source` column records all source indices.

**Benchmarks:** each index uses its correct ETF — SPY (sp500), QQQ (nasdaq100), ISF.L (ftse100), EXS1.DE (dax40).

---

## Phase B — Agentic Pipeline

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Data Gathering                 [Claude Sonnet]    │
│  4 DuckDuckGo searches per stock (news, earnings, ESG,      │
│  management). Claude extracts structured signals.           │
│  Output: data/stage1_raw/{ticker}_research.json             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Snippet Classification         [Local Qwen 7B]    │
│  Each snippet: sentiment, category, relevance, summary.     │
│  Fallback: Claude Haiku when LOCAL_MODEL=False.             │
│  Output: data/stage2_processed/{ticker}_processed.json      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: RAG Synthesis                  [FAISS + Sonnet]   │
│  Embed snippets → FAISS vector store → hybrid retrieval     │
│  (metadata filter + semantic search) → Claude synthesis.    │
│  Output: data/stage3_reports/{ticker}_report.json           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: Personalised Advisory          [Claude Haiku]     │
│  10-question risk profiling → ChatPromptTemplate + chain    │
│  composition → BUY/HOLD/AVOID per stock per user profile.   │
│  Output: reports/advisory_{user_id}.json                    │
└─────────────────────────────────────────────────────────────┘
```

### Model Allocation

| Stage | Model | Why |
|-------|-------|-----|
| 1 | Claude Sonnet 4.6 (API) | Search reasoning, result evaluation |
| 2 | Qwen2.5-7B-Instruct-AWQ (local) or Haiku (fallback) | High-volume classification |
| 3 | Claude Sonnet 4.6 (API) | Complex multi-source synthesis |
| 4 | Claude Haiku 4.5 (API) | Constrained task, cost-efficient |

### RAG Implementation

- Embedding: all-MiniLM-L6-v2 (384 dims, CPU, ~80MB)
- Vector store: FAISS (in-memory, saved to disk per stock)
- Hybrid retrieval: metadata filtering (sentiment, category) + semantic similarity
- 4 targeted queries per stock (positive, risk, ESG, management), k=3 per query
- Documents: Stage 2 classified snippets with structured metadata

### LangChain Patterns

| Pattern | Where |
|---------|-------|
| `ChatAnthropic` / `ChatOpenAI` | All stages — unified LLM interface |
| `SystemMessage` / `HumanMessage` | Stages 1, 2, 3 — structured prompts |
| `ChatPromptTemplate` + pipe operator | Stage 4 — reusable template for batch |
| `DuckDuckGoSearchResults` | Stage 1 — web search tool |
| `HuggingFaceEmbeddings` | Stage 3 — sentence embeddings |
| `FAISS` with metadata filtering | Stage 3 — hybrid vector retrieval |

### Local LLM Setup

- Model: Qwen2.5-7B-Instruct-AWQ (4-bit quantised)
- Serving: vLLM in WSL2, OpenAI-compatible API on port 8000
- VRAM: ~15.9GB (model + pre-allocated KV cache)
- Hardware: RTX 4070 Ti SUPER (16GB)
- Toggle: `LOCAL_MODEL = True/False` in config.py

### Caching

Stages 1-3 check for existing output files before processing. Re-runs skip already-processed stocks by default. Use `--force-refresh` to reprocess everything. Stage 4 always runs (user-specific, cheap).

---

## Entry Points

### `run_phase_a.py` — Phase A Only

```bash
python run_phase_a.py --index sp500 --sample 5              # Live screening
python run_phase_a.py --run-experiments --index sp500        # Full experiments
python run_phase_a.py --run-experiments --index sp500 nasdaq100 ftse100 dax40  # Multi-index
python run_phase_a.py --run-experiments --skip-extraction    # Retrain on existing data
python run_phase_a.py --run-experiments --test-only          # Evaluate only
python run_phase_a.py --extract-test-only --index nasdaq100  # New test data + evaluate
```

### `run_phase_b.py` — Phase B Only

```bash
python run_phase_b.py --csv reports/screener_scores.csv      # Full (Stages 1-4)
python run_phase_b.py --csv reports/screener_scores.csv --skip-stage4   # Stages 1-3
python run_phase_b.py --csv reports/screener_scores.csv --skip-stage2   # No local LLM
python run_phase_b.py --csv reports/screener_scores.csv --showcase      # Progress bars
python run_phase_b.py --csv reports/screener_scores.csv --force-refresh # Reprocess all
python run_phase_b.py --stage4-only --user-id bob            # New user, existing reports
```

### `run_pipeline.py` — Full End-to-End

```bash
python run_pipeline.py --index sp500                         # Live + Phase B
python run_pipeline.py --index sp500 --full                  # Experiments + Live + Phase B
python run_pipeline.py --index sp500 nasdaq100 --sample 5 --showcase
```

---

## Repository Structure

```
quantamental-screener/
├── config.py                     # All tuneable parameters
├── pipeline.py                   # Data extraction, feature computation, List 2 rules
├── live_pipeline.py              # Live data extraction → screener
├── screener.py                   # Tiered scoring with deduplication
├── features_xgboost.py           # XGBoost training + SHAP
├── feature_selection.py          # Boruta + Permutation + consensus
├── model_comparison.py           # Lasso + RF + XGBoost comparison
├── test_evaluation.py            # Test set evaluation
├── stage_1_gather.py             # Web search + Claude extraction
├── stage_2_process.py            # Local LLM snippet classification
├── stage_3_synthesize.py         # FAISS RAG + Claude synthesis
├── stage_4_personal_advisor.py   # User profiling + personalised advice
├── run_phase_a.py                # Phase A orchestrator
├── run_phase_b.py                # Phase B orchestrator
├── run_pipeline.py               # Full pipeline orchestrator
├── data/                         # All generated data (gitignored)
├── models/                       # Saved ML models (gitignored)
├── reports/                      # Analysis outputs (gitignored)
├── .env                          # API key (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

**Phase A:** Python, pandas, numpy, yfinance, scikit-learn, XGBoost, SHAP, Boruta, Optuna

**Phase B:** LangChain (ChatAnthropic, ChatOpenAI, ChatPromptTemplate, FAISS, HuggingFaceEmbeddings, DuckDuckGoSearchResults), Claude API (Sonnet 4.6 + Haiku 4.5), vLLM + Qwen2.5-7B-Instruct-AWQ, sentence-transformers

**Hardware:** RTX 4070 Ti SUPER (16GB VRAM) for local inference via vLLM in WSL2

---

## Known Biases & Limitations

- Survivorship bias: current index membership for all historical dates
- Single regime training: 2023-2024 data; ML models don't generalise across regimes
- Restated financials: yfinance serves current versions
- DuckDuckGo snippets: not full articles
- 20 documents per stock in RAG: small-scale
- Risk profiling: simplified 10 questions, not regulatory-compliant
- 90-day reporting lag: conservative but doesn't eliminate all look-ahead bias

---

## Key Design Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Screening approach | Transparent scoring, not ML prediction | ML failed to generalise; scoring is explainable |
| Feature selection | 4-method consensus, validated multi-index | No single method reliable; cross-market stability matters |
| Tier weighting | Stable features = 2x, new consensus = 1x | Evidence-proportional |
| Multi-index grouping | Hybrid (per-index percentiles, global sector medians) | Size is index-relative; fundamentals are globally comparable |
| Phase B architecture | 4-stage hybrid (API + local LLM) | Cost-efficient, demonstrates full AI engineering stack |
| Local model | Qwen 7B via vLLM with Haiku fallback | Demonstrates local deployment; fallback ensures pipeline always works |
| RAG approach | Hybrid metadata + semantic retrieval | More precise than semantic-only |
| Stage 4 model | Haiku via ChatPromptTemplate | Constrained task, reusable template for batch processing |
| Output caching | Skip existing files, --force-refresh override | Saves API costs on re-runs |
| Deduplication | Post-scoring in screener, keep highest score | Preserves per-index calculations, clean final output |

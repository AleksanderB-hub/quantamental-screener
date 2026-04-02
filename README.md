# Quantamental Stock Screener

A two-phase stock screening engine that combines quantitative rule-based filtering with an AI-powered qualitative assessment pipeline. Phase A screens stocks using fundamental and technical rules validated through ML feature selection. Phase B researches the top candidates using web search, local LLM classification, RAG-based synthesis, and personalised investment advice.

## Architecture

```
PHASE A — Quantitative Screening
─────────────────────────────────
yfinance data → 43 binary rules → ML feature selection (4 methods)
→ 9 consensus features → tiered scoring → ranked candidates

PHASE B — Qualitative Assessment
─────────────────────────────────
Stage 1: Web search + Claude Sonnet extraction → raw research JSON
Stage 2: Local Qwen 7B classification → sentiment, category, relevance per snippet
Stage 3: FAISS vector store + Claude Sonnet synthesis → investment thesis per stock
Stage 4: User risk profiling + Claude Haiku → personalised BUY/HOLD/AVOID advice
```

## Quick Start

### Prerequisites

- Python 3.10+
- Anthropic API key (set in `.env` as `ANTHROPIC_API_KEY`)
- WSL2 + vLLM for local LLM (optional — Stage 2 falls back to Claude Haiku)

### Installation

```bash
git clone https://github.com/yourusername/quantamental-screener.git
cd quantamental-screener
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

### Running the Pipeline

**Screen stocks today and run qualitative analysis:**

```bash
python run_pipeline.py --index sp500
```

**Phase A only — extract and score:**

```bash
python run_phase_a.py --index nasdaq100 --sample 10
```

**Phase B only — research existing candidates:**

```bash
python run_phase_b.py --csv reports/screener_scores_2026-03-31.csv
```

**Re-run personalised advice for a different user:**

```bash
python run_phase_b.py --stage4-only --user-id bob
```

See [Usage Guide](#usage-guide) below for all options.

## Screening Features (9 features, max score 13)

Selected through consensus across 4 independent methods (SHAP, Permutation Importance, Boruta, Lasso) and validated across multi-index experiments (S&P 500 + Nasdaq 100 + FTSE 100 + DAX 40).

**Tier 1 — 2 points each (stable across all markets):**

| Feature | Category | What it measures |
|---------|----------|-----------------|
| `50MA_Gt_200MA` | Momentum | Price in sustained uptrend (golden cross) |
| `Op_Margin_Gt_Hist_Avg` | Quality | Operating margins improving vs own history |
| `EPS_Current_Change_Above_Industry` | Momentum | Earnings growing faster than sector peers |
| `Gross_Margin_Above_Industry` | Quality | Stronger competitive position than peers |

**Tier 2 — 1 point each (strong in multi-index consensus):**

| Feature | Category | What it measures |
|---------|----------|------------------|
| `Assets_To_Liability_Ratio_Above_Industry` | Value | Stronger balance sheet than peers |
| `Zero_Dividend` | Quality | Reinvesting earnings into growth |
| `Operating_Margin_Above_Industry` | Quality | Higher operating efficiency than peers |
| `FCF_Growing_Sustained` | Quality | Consistently growing free cash flow |
| `PEG_01_to_05` | Value | Attractively priced relative to growth |

## ML Experiment Results

Three models tested across a complexity spectrum. All showed promising validation performance but failed to generalise to out-of-sample test data, confirming the need for a transparent scoring approach over black-box prediction.

```
Model               Val Spearman   Test Spearman
Lasso                  +0.150         -0.032
Random Forest          +0.144         -0.043
XGBoost (tuned)        +0.309         -0.110
```

Key finding: with limited training data covering a single market regime, added model complexity hurts rather than helps. The ML models identified which features matter (feature selection), but the final screening uses transparent rule-counting.

Note that you should re-train the models, as the one provided in this repository is a simpler version used for testing.

## Multi-Index Support

Supported indices: `sp500`, `nasdaq100`, `ftse100`, `dax40`, `custom`

Each index uses its correct benchmark ETF for excess return calculation:

| Index      | Benchmark | Suffix  |
|------------|-----------|---------|
| S&P 500    | SPY       |    —    |
| Nasdaq 100 | QQQ       |    —    |
| FTSE 100   | ISF.L     |  `.L`   |
| DAX 40     | EXS1.DE   |  `.DE`  |

Industry-relative rules (margins, ROE, debt ratios) are compared against global sector peers across all indices. Universe percentile rankings (market cap, PE, FCF) are computed per-index. Duplicates across indices are resolved by keeping the highest score.

## Phase B — AI Pipeline Details

### Stage 1: Web Research (Claude Sonnet + DuckDuckGo)

4 targeted searches per stock (news, earnings, ESG, management). Claude extracts structured signals (sentiment, confidence, key findings) from raw search results. Output: one JSON per stock.

### Stage 2: Snippet Classification (Local Qwen 7B or Haiku fallback)

Each search result snippet is independently classified for sentiment (positive/neutral/negative), category (earnings/regulatory/ESG/etc.), and relevance (high/medium/low). Runs on local vLLM server (port 8000) or falls back to Claude Haiku when `LOCAL_MODEL=False` in config.

### Stage 3: RAG Synthesis (FAISS + Claude Sonnet)

Classified snippets are embedded (all-MiniLM-L6-v2) into a per-stock FAISS vector store. Hybrid retrieval (metadata filtering + semantic search) extracts the most relevant evidence for each assessment dimension. Claude synthesises an investment thesis combining the quantitative score with qualitative signals.

### Stage 4: Personalised Advisory (Claude Haiku)

10-question risk profiling quiz determines user's investment profile (Conservative / Moderate / Moderately Aggressive / Aggressive). Each stock's objective report is evaluated against the user's profile. ESG strict override: stocks with ESG concerns receive AVOID for users with strict ESG preferences.

## Usage Guide

### Entry Points

| Script | Purpose |
|--------|---------|
| `run_phase_a.py` | Phase A: extraction, scoring, and experiments |
| `run_phase_b.py` | Phase B: qualitative analysis (Stages 1-4) |
| `run_pipeline.py`| Combined: Phase A live → Phase B |

### run_phase_a.py

```bash
# Live screening
python run_phase_a.py --index sp500
python run_phase_a.py --index nasdaq100 --sample 10

# Full experiments (extract historical data + train models + feature selection)
python run_phase_a.py --run-experiments --index sp500

# Multi-index experiments
python run_phase_a.py --run-experiments --index sp500 nasdaq100 ftse100 dax40

# Re-run models on existing data
python run_phase_a.py --run-experiments --skip-extraction

# Test existing models on new data
python run_phase_a.py --run-experiments --test-only

# Extract test data for a new index
python run_phase_a.py --extract-test-only --index nasdaq100
```

### run_phase_b.py

```bash
# Full Phase B (Stages 1-4)
python run_phase_b.py --csv reports/screener_scores_2026-03-31.csv

# Without personalisation (Stages 1-3)
python run_phase_b.py --csv reports/screener_scores_2026-03-31.csv --skip-stage4

# Skip local LLM (falls back to Haiku for Stage 2)
python run_phase_b.py --csv reports/screener_scores_2026-03-31.csv --skip-stage2

# Stage 4 only for a new user
python run_phase_b.py --stage4-only --user-id bob

# Showcase mode (progress bars, minimal output)
python run_phase_b.py --csv reports/screener_scores_2026-03-31.csv --showcase

# Force reprocess all stocks
python run_phase_b.py --csv reports/screener_scores_2026-03-31.csv --force-refresh
```

### run_pipeline.py

```bash
# Standard: extract today + score + Phase B
python run_pipeline.py --index sp500

# Full: experiments + live + Phase B
python run_pipeline.py --index sp500 --full

# Multi-index with options
python run_pipeline.py --index sp500 nasdaq100 --sample 5 --skip-stage2 --showcase
```

## Output Files

```
data/
├── stage1_raw/{ticker}_research.json         # Web search results + Claude summary
├── stage2_processed/{ticker}_processed.json  # Classified snippets
├── stage3_reports/{ticker}_report.json       # Investment thesis
├── stage3_reports/advisory_{user_id}.json    # Personalised recommendations
├── vectorstores/{ticker}/                    # FAISS indices
├── user_profile/user_{id}.json               # Saved risk profiles
├── ML_Training_Regression.csv                # Historical training features
└── ML_Testing_Regression.csv                 # Test features

reports/
├── screener_scores_{date}.csv                # All stocks ranked by score
├── phase_b_candidates_{date}.csv             # Candidates above threshold
├── feature_consensus.csv                     # 4-method feature importance consensus
├── model_comparison.csv                      # Lasso vs RF vs XGBoost results
├── feature_importance.csv                    # SHAP + XGBoost Gain rankings
├── lasso_coefficients.csv                    # Lasso feature selection
└── advisory_{user_id}.json                   # Final recommendations per user

models/
└── xgboost_model.json                        # Saved XGBoost model
```

## Tech Stack

**Phase A:** Python, pandas, numpy, yfinance, scikit-learn, XGBoost, SHAP, Boruta, Optuna

**Phase B:** LangChain (ChatAnthropic, ChatOpenAI, ChatPromptTemplate, FAISS, HuggingFaceEmbeddings, DuckDuckGoSearchResults), Claude API (Sonnet + Haiku), vLLM + Qwen2.5-7B-Instruct-AWQ

**Hardware:** RTX 4070 Ti SUPER (16GB VRAM) for local inference via vLLM in WSL2

## Configuration

All tuneable parameters are centralised in `config.py`. Key settings:

- `LIVE_INDEX` / `EXPERIMENT_INDICES` — which index(es) to screen
- `TICKER_SAMPLE_SIZE` — random sample for testing (None = full index)
- `PHASE_B_MIN_SCORE` — minimum score for Phase B candidates
- `LOCAL_MODEL` — True for local Qwen, False for Haiku fallback
- `FORCE_REFRESH` — reprocess stocks even if outputs exist
- `TIER1_FEATURES` / `TIER2_FEATURES` — the consensus screening features

## Known Limitations

- **Survivorship bias:** uses current index membership for all historical dates
- **Single regime training:** 2023-2024 bull market data; ML models don't generalise
- **DuckDuckGo snippets:** search results are snippets, not full articles
- **20 documents per stock in RAG:** small-scale; pattern scales but not stress-tested
- **Risk profiling is simplified:** 10 questions, not regulatory-compliant

## License
Please see the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details

# Quantamental Stock Screener — CLAUDE.md

## Project Summary

Two-phase S&P 500 stock screening engine.

- **Phase A (Quantitative):** ✅ Complete — ~40 binary rules, RF/SHAP feature selection → 11 consensus features, tiered scoring.
- **Phase B (Agentic):** ✅ Complete — 4-stage LangChain pipeline for qualitative assessment of Phase A candidates.

## Key Scripts

**Phase A:**
- `pipeline.py` — data extraction, snapshot generation, feature computation
- `feature_selection.py` — RF importance / SHAP → 11 consensus features
- `screener.py` — tiered scoring → `reports/phase_b_candidates.csv`

**Phase B:**
- `stage_1_gather.py` — DuckDuckGo web search + Claude Sonnet 4.6 extraction → `data/stage1_raw/{ticker}_research.json`
- `stage_2_process.py` — local Qwen 2.5-7B (vLLM on localhost:8000) snippet classification → `data/stage2_processed/{ticker}_processed.json`
- `stage_3_synthesize.py` — HuggingFace embeddings + FAISS RAG + Claude Sonnet 4.6 synthesis → `data/stage3_reports/{ticker}_report.json`
- `stage_4_personal_advisor.py` — user risk profiling quiz + Claude Haiku personalization → `reports/advisory_{user_id}.json`

## Pipeline Architecture

```
Phase A: yfinance → feature store (SQLite) → screener.py → phase_b_candidates.csv
Phase B: phase_b_candidates.csv
  → stage_1_gather.py  (DuckDuckGo + Claude Sonnet 4.6)
  → stage_2_process.py (local Qwen 2.5-7B via vLLM)
  → stage_3_synthesize.py (FAISS RAG + Claude Sonnet 4.6)
  → stage_4_personal_advisor.py (Claude Haiku 4.5 + ChatPromptTemplate)
```

## LangChain Patterns Used

| Stage | Components |
|-------|-----------|
| 1 | `DuckDuckGoSearchResults`, `ChatAnthropic`, `SystemMessage`/`HumanMessage` |
| 2 | `ChatOpenAI(base_url=LOCAL_MODEL_URL)` — connects to vLLM OpenAI-compatible endpoint |
| 3 | `HuggingFaceEmbeddings(all-MiniLM-L6-v2)`, `FAISS`, `Document`, `ChatAnthropic` |
| 4 | `ChatPromptTemplate.from_messages()`, `ChatAnthropic(Haiku 4.5)` |

## 11 Consensus Features (max score = 15)

| Tier | Points | Features |
|------|--------|----------|
| Tier 1 | 2 pts | `50MA_Gt_200MA`, `Op_Margin_Gt_Hist_Avg`, `PE_Below_Industry`, `Gross_Margin_Above_Industry` |
| Tier 2 | 1 pt | `ROE_Gt_15_Sustained`, `EPS_Current_Change_Above_Industry`, `Price_Gt_50MA`, `DE_Less_1`, `PE_Bottom_40_Pct`, `PB_Below_Industry`, `Assets_To_Liability_Ratio_Above_Industry` |

Candidates scoring ≥ `PHASE_B_MIN_SCORE` (config.py, currently 11) proceed to Phase B.

## Coding Conventions

- **NaN for missing data:** Never default to 0 for genuinely missing values; propagate `NaN`.
- **Dynamic fallback:** Historical metrics try 3yr → 2yr → 1yr; propagate `NaN` if none available.
- **All config in `config.py`:** Dates, paths, thresholds, model names/URLs — never hardcode.
- **temperature=0** for all classification/extraction tasks; 0.1 only for generative personalization (stage 4).
- **JSON output from LLMs:** Prompt for JSON-only responses; strip markdown backticks before parsing (`response.strip().strip("```json").strip("```")`).
- **Local Qwen model:** served via vLLM in WSL2 on `localhost:8000` — not always running. Scripts using this endpoint must handle connection errors gracefully (catch `ConnectionError`/`httpx` errors, log a clear message, skip or fall back).

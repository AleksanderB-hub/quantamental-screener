#!/bin/bash
# ============================================================
# run.sh - Quantamental Stock Screener Launcher
# ============================================================
# Edit the variables below to configure your run, then execute:
#   bash run.sh
# ============================================================

# -- Index Selection ------------------------------------------
# Supported: sp500, nasdaq100, ftse100, dax40, custom
INDEX="nasdaq100"

# Only used if INDEX=custom - path to CSV with Ticker or Symbol column
CUSTOM_TICKERS_CSV=""

# -- Sample Size ----------------------------------------------
# 0 = full index, N = random sample of N stocks (useful for testing)
TICKER_SAMPLE=5

# -- Pipeline Scope -------------------------------------------
# Set to "true" to run Phase A (live data extraction + screening)
RUN_PHASE_A=true

# Set to "false" to run Phase A only, skipping all Phase B stages
PHASE_B=false

# -- Screener CSV ---------------------------------------------
# Only needed if RUN_PHASE_A=false
SCREENER_CSV="reports/screener_scores_2026-03-26.csv"

# -- Candidate Selection --------------------------------------
# Minimum screener score to pass into Phase B (max possible: 15)
MIN_SCORE=11

# -- User Profile ----------------------------------------------
# User ID for Stage 4 personalised advisory
USER_ID="default_user"

# -- Display Mode ----------------------------------------------
# Set to "true" for tqdm progress bars only (clean demo output)
SHOWCASE=true

# -- Local LLM ------------------------------------------------
# Set to "true" to skip Stage 2 (if vLLM is not running in WSL2)
SKIP_STAGE2=false

# ============================================================
# BUILD THE COMMAND (no need to edit below this line)
# ============================================================

CMD="python run_phase_b.py"

if [ "$RUN_PHASE_A" = true ]; then
    CMD="$CMD --run-phase-a"
else
    CMD="$CMD --csv $SCREENER_CSV"
fi

if [ -n "$CUSTOM_TICKERS_CSV" ]; then
    CMD="$CMD --custom-tickers $CUSTOM_TICKERS_CSV"
else
    CMD="$CMD --index $INDEX"
fi

if [ "$TICKER_SAMPLE" != "0" ]; then
    CMD="$CMD --ticker-sample $TICKER_SAMPLE"
fi

CMD="$CMD --min-score $MIN_SCORE"
CMD="$CMD --user-id $USER_ID"

if [ "$PHASE_B" = false ]; then
    CMD="$CMD --no-phase-b"
fi

if [ "$SHOWCASE" = true ]; then
    CMD="$CMD --showcase"
fi

if [ "$SKIP_STAGE2" = true ]; then
    CMD="$CMD --skip-stage2"
fi

# -- Run -------------------------------------------------------
echo ""
echo "============================================================"
echo "  Quantamental Stock Screener"
echo "============================================================"
echo "  Index:        $INDEX $([ -n "$CUSTOM_TICKERS_CSV" ] && echo "(custom: $CUSTOM_TICKERS_CSV)")"
echo "  Sample:       $TICKER_SAMPLE (0=full)"
echo "  Phase A:      $([ "$RUN_PHASE_A" = true ] && echo 'ON' || echo 'OFF')"
echo "  Phase B:      $([ "$PHASE_B" = true ] && echo 'ON' || echo 'OFF')"
echo "  Min Score:    $MIN_SCORE"
echo "  User ID:      $USER_ID"
echo "  Showcase:     $([ "$SHOWCASE" = true ] && echo 'ON' || echo 'OFF')"
echo "  Skip Stage 2: $([ "$SKIP_STAGE2" = true ] && echo 'YES' || echo 'NO')"
echo "  Command:      $CMD"
echo "============================================================"
echo ""

$CMD
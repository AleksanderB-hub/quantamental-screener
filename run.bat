@echo off
rem ============================================================
rem run.bat — Quantamental Stock Screener Launcher (Windows)
rem ============================================================
rem Edit the variables below to configure your run, then execute:
rem   run.bat
rem ============================================================

rem ── Index Selection ─────────────────────────────────────────
rem Supported: sp500, nasdaq100, ftse100, dax40, custom
set INDEX=nasdaq100

rem ── Sample Size ─────────────────────────────────────────────
rem 0 = full index, N = random sample of N stocks (useful for testing)
set TICKER_SAMPLE=10

rem ── Custom Tickers CSV ──────────────────────────────────────
rem Only used if INDEX=custom — path to CSV with Ticker or Symbol column
set CUSTOM_TICKERS_CSV=

rem ── Pipeline Scope ──────────────────────────────────────────
rem Set to "true" to run Phase A (live data extraction + screening)
rem Set to "false" to skip Phase A and use an existing screener CSV
set RUN_PHASE_A=true

rem Set to "false" to run Phase A only, skipping all Phase B stages
set PHASE_B=false

rem ── Screener CSV ────────────────────────────────────────────
rem Only needed if RUN_PHASE_A=false
rem If RUN_PHASE_A=true, this is ignored (uses freshly generated CSV)
set SCREENER_CSV=reports/screener_scores_2026-03-26.csv

rem ── Candidate Selection ────────────────────────────────────
rem Minimum screener score to pass into Phase B (max possible: 15)
set MIN_SCORE=11

rem ── User Profile ───────────────────────────────────────────
rem User ID for Stage 4 personalised advisory
rem Existing profiles are reused; new IDs trigger the risk quiz
set USER_ID=default_user

rem ── Display Mode ───────────────────────────────────────────
rem Set to "true" for tqdm progress bars only (clean demo output)
rem Set to "false" for full debug output
set SHOWCASE=true

rem ── Local LLM ──────────────────────────────────────────────
rem Set to "true" to skip Stage 2 (if vLLM is not running in WSL2)
set SKIP_STAGE2=false

rem ============================================================
rem BUILD THE COMMAND (no need to edit below this line)
rem ============================================================

set CMD=python run_phase_b.py

if /i "%RUN_PHASE_A%"=="true" (
    set CMD=%CMD% --run-phase-a
) else (
    set CMD=%CMD% --csv %SCREENER_CSV%
)

if not "%CUSTOM_TICKERS_CSV%"=="" (
    set CMD=%CMD% --custom-tickers %CUSTOM_TICKERS_CSV%
) else (
    set CMD=%CMD% --index %INDEX%
)

if not "%TICKER_SAMPLE%"=="0" (
    set CMD=%CMD% --ticker-sample %TICKER_SAMPLE%
)

set CMD=%CMD% --min-score %MIN_SCORE%
set CMD=%CMD% --user-id %USER_ID%

if /i "%PHASE_B%"=="false" (
    set CMD=%CMD% --no-phase-b
)

if /i "%SHOWCASE%"=="true" (
    set CMD=%CMD% --showcase
)

if /i "%SKIP_STAGE2%"=="true" (
    set CMD=%CMD% --skip-stage2
)

rem ── Run ────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Quantamental Stock Screener
echo ============================================================
echo   Index:        %INDEX%
echo   Sample:       %TICKER_SAMPLE% (0=full)
if /i "%RUN_PHASE_A%"=="true" (echo   Phase A:      ON) else (echo   Phase A:      OFF)
if /i "%PHASE_B%"=="true" (echo   Phase B:      ON) else (echo   Phase B:      OFF)
echo   Min Score:    %MIN_SCORE%
echo   User ID:      %USER_ID%
if /i "%SHOWCASE%"=="true" (echo   Showcase:     ON) else (echo   Showcase:     OFF)
if /i "%SKIP_STAGE2%"=="true" (echo   Skip Stage 2: YES) else (echo   Skip Stage 2: NO)
echo   Command:      %CMD%
echo ============================================================
echo.

%CMD%

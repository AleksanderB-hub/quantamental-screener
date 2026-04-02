"""
run_phase_b.py — Phase B: Qualitative Stock Analysis

Runs Stages 1-4 of the qualitative analysis pipeline on screened candidates.
Requires an existing screener scores CSV (from Phase A).

Usage:
  # Full Phase B (all 4 stages)
  python run_phase_b.py --csv reports/screener_scores_2026-03-31.csv

  # Without personalization (Stages 1-3 only)
  python run_phase_b.py --csv reports/screener_scores_2026-03-31.csv --skip-stage4

  # Skip local LLM classification (Stages 1, 3, 4 only)
  python run_phase_b.py --csv reports/screener_scores_2026-03-31.csv --skip-stage2

  # Showcase mode with custom score threshold
  python run_phase_b.py --csv reports/screener_scores_2026-03-31.csv --showcase --min-score 13

  # With personalization for specific user
  python run_phase_b.py --csv reports/screener_scores_2026-03-31.csv --user-id alice

  # Run Stage 4 only for a new user on existing Stage 3 reports
  python run_phase_b.py --stage4-only --user-id bob
  python run_phase_b.py --stage4-only --user-id carol
"""

import argparse
import sys
import os
import io
from contextlib import redirect_stdout
import pandas as pd
from tqdm import tqdm
import config as cfg


def load_tickers(csv_path: str, min_score: int) -> list:
    """Reads screener CSV and returns tickers with Score >= min_score."""
    df = pd.read_csv(csv_path)
    filtered = df[df["Score"] >= min_score]
    tickers = filtered["Ticker"].tolist()
    return tickers


def print_header(stage: str, showcase: bool = False):
    if not showcase:
        print(f"\n{'#'*70}")
        print(f"#  {stage}")
        print(f"{'#'*70}\n")


def suppress_output(func, *args, **kwargs):
    """Runs a function with stdout suppressed. Returns the function's result."""
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def main(csv_path=None, min_score=None, user_id=None, skip_stage2=False,
         skip_stage4=False, showcase=False, stage4_only=False, force_refresh=False):
    """
    Run Phase B pipeline. Can be called from CLI or imported by run_pipeline.py.
    """
    _min_score = min_score if min_score is not None else cfg.PHASE_B_RUNNER_MIN_SCORE
    _user_id = user_id or "default_user"

    # ── Stage 4 only mode ─────────────────────────────────────────
    if stage4_only:
        reports = list(cfg.STAGE3_DIR.glob("*_report.json"))
        print(f"\nStage 4 Only — Found {len(reports)} existing Stage 3 reports")
        if not reports:
            print("No Stage 3 reports found. Run Stages 1-3 first.")
            return []

        from engine.stage_4_personal_advisor import run_final_advisory_batch
        run_final_advisory_batch(user_id=_user_id)
        return []

    # ── Validate CSV (required for all other modes) ───────────────
    if csv_path is None or not os.path.exists(csv_path):
        print(f"\nError: No screener CSV found at '{csv_path}'.")
        print("Provide a valid --csv path or use run_pipeline.py for end-to-end.")
        return []

    # ── Load candidate tickers ─────────────────────────────────────
    tickers = load_tickers(csv_path, _min_score)

    if not tickers:
        print("No tickers passed the score filter. Exiting.")
        return []

    if showcase:
        print(f"\n Phase B — Processing {len(tickers)} candidates (score >= {_min_score})")
        print("-" * 50)
    else:
        print_header("PHASE B — Loading candidates")
        print(f"  Loaded {len(tickers)} tickers with Score >= {_min_score} from {csv_path}")

    # ── Stage 1 ────────────────────────────────────────────────────
    from engine.stage_1_gather import process_stock_stage_1, write_stage1_summary

    if showcase:
        stage1_success = []
        for ticker in tqdm(tickers, desc="Stage 1 — Web Research       "):
            try:
                suppress_output(process_stock_stage_1, ticker)
                stage1_success.append(ticker)
            except Exception:
                pass
        suppress_output(write_stage1_summary)
    else:
        print_header("STAGE 1 — Web Research & Claude Extraction")
        from engine.stage_1_gather import run_batch as stage1_batch
        stage1_success = stage1_batch(csv_path, _min_score)
        write_stage1_summary()

    # ── Stage 2 ────────────────────────────────────────────────────
    if skip_stage2:
        if not showcase:
            print_header("STAGE 2 — SKIPPED (--skip-stage2 flag set)")
        stage2_success = stage1_success
    else:
        from engine.stage_2_process import process_stock_stage_2, write_stage2_summary

        if showcase:
            stage2_success = []
            for ticker in tqdm(stage1_success, desc="Stage 2 — Local Classification"):
                try:
                    suppress_output(process_stock_stage_2, ticker)
                    stage2_success.append(ticker)
                except Exception:
                    pass
            suppress_output(write_stage2_summary)
        else:
            print_header("STAGE 2 — Local LLM Snippet Classification")
            from engine.stage_2_process import run_batch as stage2_batch
            stage2_success = stage2_batch(stage1_success)
            write_stage2_summary()

    # ── Stage 3 ────────────────────────────────────────────────────
    from engine.stage_3_synthesize import process_stock_stage_3, write_stage3_summary

    if showcase:
        stage3_success = []
        for ticker in tqdm(stage2_success, desc="Stage 3 — RAG Synthesis      "):
            try:
                suppress_output(process_stock_stage_3, ticker)
                stage3_success.append(ticker)
            except Exception:
                pass
        suppress_output(write_stage3_summary)
    else:
        print_header("STAGE 3 — FAISS RAG Synthesis & Claude Report")
        from engine.stage_3_synthesize import run_batch as stage3_batch
        stage3_success = stage3_batch(stage2_success)
        write_stage3_summary()

    if not stage3_success:
        print("\n[WARNING] No Stage 3 reports produced.")
        if not showcase:
            print("  Common causes: Stage 2 was skipped and no prior Stage 2 JSONs exist,")
            print("  or all tickers failed during Stage 3.")

    # ── Stage 4 (optional) ─────────────────────────────────────────
    if skip_stage4:
        if not showcase:
            print_header("STAGE 4 — SKIPPED (--skip-stage4 flag set)")
    else:
        if showcase:
            print()
        print_header("STAGE 4 — Personalized Advisory", showcase=False)

        from engine.stage_4_personal_advisor import run_final_advisory_batch
        run_final_advisory_batch(user_id=_user_id)

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Phase B complete.")
    print(f"{'='*70}")
    print(f"  Stage 1 output : {cfg.STAGE1_DIR}")
    print(f"  Stage 2 output : {cfg.STAGE2_DIR}")
    print(f"  Stage 3 output : {cfg.STAGE3_DIR}")
    if not skip_stage4:
        print(f"  Advisory output: {cfg.REPORTS_DIR / f'advisory_{_user_id}.json'}")
    print()

    return stage3_success


# ── CLI Entry Point ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase B: Qualitative stock analysis (Stages 1-4)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to screener scores CSV (not needed with --stage4-only)")
    parser.add_argument("--min-score", type=int, default=cfg.PHASE_B_RUNNER_MIN_SCORE)
    parser.add_argument("--user-id", type=str, default="default_user")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip local LLM classification")
    parser.add_argument("--skip-stage4", action="store_true", help="Skip personalized advisory")
    parser.add_argument("--showcase", action="store_true", help="Minimal output with progress bars")
    parser.add_argument("--stage4-only", action="store_true",
                        help="Run Stage 4 only for a new user on existing Stage 3 reports")
    parser.add_argument("--force-refresh", action="store_true",
                    help="Reprocess all stocks even if output files already exist")
    args = parser.parse_args()

    main(
        csv_path=args.csv if not args.stage4_only else None,
        min_score=args.min_score,
        user_id=args.user_id,
        skip_stage2=args.skip_stage2,
        skip_stage4=args.skip_stage4,
        showcase=args.showcase,
        stage4_only=args.stage4_only,
        force_refresh=args.force_refresh
    )


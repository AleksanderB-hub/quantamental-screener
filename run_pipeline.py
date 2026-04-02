"""
run_pipeline.py — Full End-to-End Pipeline

Usage:
  # Standard: live extraction + scoring + Phase B
  python run_pipeline.py --index sp500
  python run_pipeline.py --index sp500 nasdaq100 --sample 5 --showcase

  # Full: experiments + live extraction + scoring + Phase B
  python run_pipeline.py --index sp500 --full
  python run_pipeline.py --index sp500 nasdaq100 --full --showcase

  # With options
  python run_pipeline.py --index nasdaq100 --sample 10 --skip-stage2 --showcase
  python run_pipeline.py --index sp500 --full --user-id alice
"""

import argparse
import config as cfg


def main():
    parser = argparse.ArgumentParser(
        description="Quantamental Stock Screener — Full Pipeline"
    )

    # Index and sampling
    parser.add_argument("--index", type=str, nargs="+", default=None,
                        help="Index(es) to screen. E.g.: --index sp500 nasdaq100")
    parser.add_argument("--custom-tickers", type=str, default=None)
    parser.add_argument("--sample", type=int, default=0, help="0 = full index")

    # Pipeline scope
    parser.add_argument("--full", action="store_true",
                        help="Run experiments (historical extraction + model training) before live screening")

    # Phase B options
    parser.add_argument("--min-score", type=int, default=cfg.PHASE_B_RUNNER_MIN_SCORE)
    parser.add_argument("--user-id", type=str, default="default_user")
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Skip Stage 2 (local LLM). Use when vLLM is not running.")
    parser.add_argument("--skip-stage4", action="store_true",
                        help="Skip Stage 4 (personalized advisory)")
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument("--stage4-only", action="store_true",
                        help="Run Stage 4 only for a new user on existing Stage 3 reports")
    parser.add_argument("--force-refresh", action="store_true",
                    help="Reprocess all stocks even if output files already exist")

    args = parser.parse_args()

    # Live screening supports single or multiple indices
    resolved_index = args.index if args.index else cfg.LIVE_INDEX

    # ── Full mode: experiments first ──────────────────────────────
    if args.full:
        print(f"\n{'='*70}")
        print("  PHASE A — Experiments (Historical Extraction + Model Training)")
        print(f"{'='*70}")

        from run_phase_a import run_experiments

        exp_indices = args.index if args.index else cfg.EXPERIMENT_INDICES
        if isinstance(exp_indices, str):
            exp_indices = [exp_indices]

        run_experiments(
            indices=exp_indices,
            sample=args.sample,
        )

    # ── Live extraction + screening ───────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE A — Live Data Extraction & Screening")
    print(f"{'='*70}")

    from live_pipeline import main as run_phase_a
    run_phase_a(
        index=resolved_index,
        custom_csv=args.custom_tickers,
        ticker_sample=args.sample or None,
    )

    csv_path = str(cfg.SCREENER_SCORES_CSV)

    # ── Phase B ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE B — Qualitative Analysis Pipeline")
    print(f"{'='*70}")

    from run_phase_b import main as run_phase_b
    run_phase_b(
        csv_path=csv_path,
        min_score=args.min_score,
        user_id=args.user_id,
        skip_stage2=args.skip_stage2,
        skip_stage4=args.skip_stage4,
        showcase=args.showcase,
        stage4_only=args.stage4_only,
        force_refresh=args.force_refresh
    )

    print(f"\n{'='*70}")
    print("  Full pipeline complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

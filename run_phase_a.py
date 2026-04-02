"""
run_phase_a.py — Phase A Orchestrator

Usage:
  # Live screening
  python run_phase_a.py --index nasdaq100 --live --sample 5

  # Full experiments (extract + train + feature selection)
  python run_phase_a.py --run-experiments --index sp500

  # Multi-index experiments
  python run_phase_a.py --run-experiments --index sp500 nasdaq100

  # Re-run models on existing data (no extraction)
  python run_phase_a.py --run-experiments --skip-extraction

  # Test existing models on new data only
  python run_phase_a.py --run-experiments --test-only

  # Extract test data only and evaluate existing model (e.g. SP500-trained → Nasdaq100), note that only XGBoost is saved form previous experiments
  python run_phase_a.py --run-experiments --extract-test-only --index nasdaq100

  # Experiments then live screening
  python run_phase_a.py --run-experiments --live --index sp500
"""

import argparse
import config as cfg


def run_live_screening(index, sample, custom_csv=None):
    """Extracts today's data, scores stocks, saves candidates."""
    print(f"\n{'='*60}")
    print(f"Phase A — Live Screening ({index}, sample={sample or 'full'})")
    print(f"{'='*60}\n")

    from engine.live_pipeline import main as run_live
    run_live(index=index, custom_csv=custom_csv, ticker_sample=sample or None)
    print(f"\nScreener scores saved to: {cfg.SCREENER_SCORES_CSV}")


def run_experiments(indices, sample, skip_extraction=False, test_only=False,
                    extract_test_only=False, custom_csv=None):
    """
    Extracts historical data, trains models, runs feature selection.

    Modes:
      Default:           Extract data → train models → feature selection
      skip_extraction:   Use existing CSVs → train models → feature selection
      test_only:         Use existing models → evaluate on test CSV only
      extract_test_only: Extract test data only → evaluate existing models (cross-index)
    """

    if isinstance(indices, str):
        indices = [indices]

    # ── Extract-test-only mode: pull test data for each index, then evaluate ──
    if extract_test_only:
        original_lag = cfg.REPORTING_LAG_DAYS
        cfg.REPORTING_LAG_DAYS = 90

        print(f"\n{'='*60}")
        print(f"Phase A — Extract Test Data Only ({', '.join(indices)})")
        print(f"{'='*60}\n")

        from engine.pipeline import get_tickers, run_backtest_pipeline, calculate_list_2_rules
        import pandas as pd
        import random

        all_test_raw = []

        for idx_name in indices:
            print(f"\n--- Extracting test data: {idx_name} ---")
            benchmark = cfg.INDEX_BENCHMARKS.get(idx_name, "SPY")
            tickers = get_tickers(idx_name, custom_csv)  # type: ignore
            if sample:
                random.seed(cfg.RANDOM_SEED)
                tickers = random.sample(tickers, min(sample, len(tickers)))
                print(f"  Sample mode: {len(tickers)} tickers from {idx_name}")

            test_raw = run_backtest_pipeline(tickers, cfg.EXPERIMENT_TEST_DATES, benchmark_ticker=benchmark)
            if not test_raw.empty:
                test_raw["Index_Source"] = idx_name
                all_test_raw.append(test_raw)

        cfg.REPORTING_LAG_DAYS = original_lag

        if all_test_raw:
            combined_test_raw = pd.concat(all_test_raw, ignore_index=True)
            print(f"\nCombined test set: {len(combined_test_raw)} rows from {len(indices)} indices")
            combined_test = calculate_list_2_rules(combined_test_raw)
            cols_to_drop = [c for c in combined_test.columns if c.endswith('_Raw')]
            combined_test.drop(columns=cols_to_drop, errors='ignore').to_csv(cfg.TESTING_REGRESSION_CSV, index=False)
            print(f"Test data saved: {cfg.TESTING_REGRESSION_CSV}")

        print(f"\n[2/2] Evaluating existing models on new test data...")
        from engine.test_evaluation import main as run_test_eval
        run_test_eval()

        print(f"\nExperiments complete. Check reports/ for results.")
        return

    # ── Step 1-2: Data extraction (unless skipped) ──
    if not skip_extraction and not test_only:
        original_lag = cfg.REPORTING_LAG_DAYS
        cfg.REPORTING_LAG_DAYS = 90

        print(f"\n{'='*60}")
        print(f"Phase A — Historical Data Extraction ({', '.join(indices)})")
        print(f"{'='*60}\n")

        from engine.pipeline import get_tickers, run_backtest_pipeline, calculate_list_2_rules
        import pandas as pd
        import random

        all_train_raw = []
        all_test_raw = []

        for idx_name in indices:
            print(f"\n--- Extracting: {idx_name} ---")
            benchmark = cfg.INDEX_BENCHMARKS.get(idx_name, "SPY")
            tickers = get_tickers(idx_name, custom_csv)  # type: ignore
            if sample:
                random.seed(cfg.RANDOM_SEED)
                tickers = random.sample(tickers, min(sample, len(tickers)))
                print(f"  Sample mode: {len(tickers)} tickers from {idx_name}")

            # Training data
            train_raw = run_backtest_pipeline(tickers, cfg.EXPERIMENT_TRAIN_DATES, benchmark_ticker=benchmark, desc=f"{idx_name} — Training Data")
            if not train_raw.empty:
                train_raw["Index_Source"] = idx_name
                all_train_raw.append(train_raw)

            # Test data
            test_raw = run_backtest_pipeline(tickers, cfg.EXPERIMENT_TEST_DATES, benchmark_ticker=benchmark, desc=f"{idx_name} — Test Data")
            if not test_raw.empty:
                test_raw["Index_Source"] = idx_name
                all_test_raw.append(test_raw)

        cfg.REPORTING_LAG_DAYS = original_lag

        # Merge all indices then compute List 2 on combined universe
        if all_train_raw:
            combined_train_raw = pd.concat(all_train_raw, ignore_index=True)
            print(f"\nCombined training set: {len(combined_train_raw)} rows from {len(indices)} indices")
            combined_train = calculate_list_2_rules(combined_train_raw)
            combined_train['Target_Percentile'] = combined_train.groupby('Screening_Date')['Forward_6m_Return'].rank(pct=True)
            cols_to_drop = [c for c in combined_train.columns if c.endswith('_Raw')]
            combined_train.drop(columns=cols_to_drop, errors='ignore').to_csv(cfg.TRAINING_REGRESSION_CSV, index=False)
            print(f"Training data saved: {cfg.TRAINING_REGRESSION_CSV}")

        if all_test_raw:
            combined_test_raw = pd.concat(all_test_raw, ignore_index=True)
            print(f"Combined test set: {len(combined_test_raw)} rows from {len(indices)} indices")
            combined_test = calculate_list_2_rules(combined_test_raw)
            combined_test['Target_Percentile'] = combined_test.groupby('Screening_Date')['Forward_6m_Return'].rank(pct=True)
            cols_to_drop = [c for c in combined_test.columns if c.endswith('_Raw')]
            combined_test.drop(columns=cols_to_drop, errors='ignore').to_csv(cfg.TESTING_REGRESSION_CSV, index=False)
            print(f"Test data saved: {cfg.TESTING_REGRESSION_CSV}")

    else:
        print(f"\n{'='*60}")
        print(f"Phase A — Skipping extraction, using existing data")
        print(f"{'='*60}")
        print(f"  Train: {cfg.TRAINING_REGRESSION_CSV}")
        print(f"  Test:  {cfg.TESTING_REGRESSION_CSV}")

    # ── Step 3-5: Model training + feature selection (unless test-only) ──
    if not test_only:
        print(f"\n[3/5] Training XGBoost + SHAP analysis...")
        from engine.features_xgboost import main as run_xgboost
        run_xgboost()

        print(f"\n[4/5] Running model comparison (Lasso + RF)...")
        from engine.model_comparison import main as run_models
        run_models()

        print(f"\n[5/5] Running feature selection...")
        from engine.feature_selection import main as run_features
        run_features()
    else:
        print(f"\n{'='*60}")
        print(f"Phase A — Test-only mode: evaluating existing models on test data")
        print(f"{'='*60}")
        from engine.test_evaluation import main as run_test_eval
        run_test_eval()

    print(f"\nExperiments complete. Check reports/ for results.")


def main():
    parser = argparse.ArgumentParser(description="Phase A: Screening and experiments")
    parser.add_argument("--index", type=str, nargs="+", default=None,
                        help="Index(es) to use. Can specify multiple: --index sp500 nasdaq100")
    parser.add_argument("--custom-tickers", type=str, default=None, help="Path to custom ticker CSV")
    parser.add_argument("--sample", type=int, default=0, help="0 = full index")
    parser.add_argument("--min-score", type=int, default=cfg.PHASE_B_MIN_SCORE)
    parser.add_argument("--run-experiments", action="store_true",
                        help="Run historical extraction + models + feature selection")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip data extraction, use existing CSVs")
    parser.add_argument("--test-only", action="store_true",
                        help="Evaluate existing models on test data only (no training)")
    parser.add_argument("--extract-test-only", action="store_true",
                        help="Extract test data only (no training data, no model training)")
    parser.add_argument("--live", action="store_true",
                        help="Run live screening (default if --run-experiments not set)")
    args = parser.parse_args()

    if args.extract_test_only:
        args.run_experiments = True

    # Resolve index: pass as-is (live_pipeline handles both string and list)
    resolved_index = args.index if args.index else cfg.LIVE_INDEX

    if args.run_experiments:
        exp_indices = args.index if args.index else cfg.EXPERIMENT_INDICES
        if isinstance(exp_indices, str):
            exp_indices = [exp_indices]

        run_experiments(
            indices=exp_indices,
            sample=args.sample,
            skip_extraction=args.skip_extraction,
            test_only=args.test_only,
            extract_test_only=args.extract_test_only,
            custom_csv=args.custom_tickers,
        )

    if args.live or not args.run_experiments:
        run_live_screening(resolved_index, args.sample, custom_csv=args.custom_tickers)

    print(f"\nPhase A complete.")


if __name__ == "__main__":
    main()

"""
live_pipeline.py
================
The Production Inference Script. 
Fetches fundamental data for TODAY, cleans it, and prepares it for the screener.
"""

import sys, os
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import config as cfg
# Import your core extraction functions from your original pipeline
from .pipeline import get_tickers, run_backtest_pipeline, calculate_list_2_rules
from .screener import run_screener

def main(index=None, custom_csv=None, ticker_sample=None):
    idx = index or cfg.LIVE_INDEX
    csv = custom_csv or cfg.CUSTOM_TICKERS_CSV
    
    # Normalise to list
    indices = idx if isinstance(idx, list) else [idx]
    
    print(f"=== Starting LIVE Stock Screen for {cfg.LIVE_SCREEN_DATE[0]} ===\n")
    
    import random
    all_raw = []
    
    for idx_name in indices:
        benchmark = cfg.INDEX_BENCHMARKS.get(idx_name, "SPY")
        print(f"Fetching {idx_name} universe...")
        tickers = get_tickers(idx_name, csv)
        
        sample_size = ticker_sample if ticker_sample else cfg.TICKER_SAMPLE_SIZE
        if sample_size:
            tickers = random.sample(tickers, min(sample_size, len(tickers)))
            print(f"  TEST mode: {len(tickers)} stocks from {idx_name}")
        else:
            print(f"  FULL universe: {len(tickers)} stocks from {idx_name}")
        
        raw = run_backtest_pipeline(tickers, cfg.LIVE_SCREEN_DATE, benchmark_ticker=benchmark)
        if not raw.empty:
            raw["Index_Source"] = idx_name
            all_raw.append(raw)
    
    if not all_raw:
        print("Error: No data retrieved. Exiting.")
        return
    
    live_df_raw = pd.concat(all_raw, ignore_index=True)
    print(f"\nCombined: {len(live_df_raw)} stocks from {len(indices)} indices")
    
    # Calculate industry relative rules (hybrid grouping handles Index_Source)
    print("\nCalculating industry relative rules...")
    live_df_final = calculate_list_2_rules(live_df_raw)
    
    # Clean up
    cols_to_drop = [col for col in live_df_final.columns if col.endswith('_Raw')]
    if 'Forward_6m_Return' in live_df_final.columns:
        cols_to_drop.append('Forward_6m_Return')
    ml_ready_live_df = live_df_final.drop(columns=cols_to_drop, errors='ignore')
    
    ml_ready_live_df.to_csv(cfg.LIVE_FEATURES_CSV, index=False)
    print(f"\nLive data extraction complete! Saved to {cfg.LIVE_FEATURES_CSV}")
    
    run_screener(cfg.LIVE_FEATURES_CSV)

if __name__ == "__main__":
    main()

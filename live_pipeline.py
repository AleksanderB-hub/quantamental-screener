"""
live_pipeline.py
================
The Production Inference Script. 
Fetches fundamental data for TODAY, cleans it, and prepares it for the screener.
"""

import pandas as pd
import config as cfg
# Import your core extraction functions from your original pipeline
from pipeline import get_sp500_tickers, run_backtest_pipeline, calculate_list_2_rules
from screener import run_screener

def main():
    print(f"=== Starting LIVE Stock Screen for {cfg.LIVE_SCREEN_DATE[0]} ===\n")
    
    # 1. Get the Universe
    print("Fetching S&P 500 universe...")
    tickers = get_sp500_tickers()
    
    # Optional: If you still have TICKER_SAMPLE_SIZE set to 10 in config, it will test 10.
    # If it is set to None, it will run the full 500.
    import random
    if cfg.TICKER_SAMPLE_SIZE:
        tickers = random.sample(tickers, cfg.TICKER_SAMPLE_SIZE)
        print(f"Running in TEST mode: {len(tickers)} stocks.")
    else:
        print(f"Running FULL universe: {len(tickers)} stocks.")

    # 2. Extract Data for TODAY
    # Note: Because the date is today, your 'Forward_6m_Return' logic in get_robust_financials 
    # will naturally fail to find future prices and safely return NaN. This is perfectly fine!
    print("\n[1/2] Fetching fundamental snapshots...")
    live_df_raw = run_backtest_pipeline(tickers, cfg.LIVE_SCREEN_DATE)
    
    if live_df_raw.empty:
        print("Error: No data retrieved. Exiting.")
        return

    # 3. Calculate Industry Relative Rules
    print("\n[2/2] Calculating industry relative rules...")
    live_df_final = calculate_list_2_rules(live_df_raw)
    
    # 4. Clean up the dataset (Drop the _Raw columns so it matches the ML format)
    cols_to_drop = [col for col in live_df_final.columns if col.endswith('_Raw')]
    # Also drop the 6-month return column since it's irrelevant for live picking
    if 'Forward_6m_Return' in live_df_final.columns:
        cols_to_drop.append('Forward_6m_Return')
        
    ml_ready_live_df = live_df_final.drop(columns=cols_to_drop, errors='ignore')
    
    # 5. Save the Live Feature Matrix
    ml_ready_live_df.to_csv(cfg.LIVE_FEATURES_CSV, index=False)
    print(f"\nLive data extraction complete! Saved to {cfg.LIVE_FEATURES_CSV}")
    
    # 6. RUN THE SCREENER AUTOMATICALLY
    run_screener(cfg.LIVE_FEATURES_CSV)

if __name__ == "__main__":
    main()

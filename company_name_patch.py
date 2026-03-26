import pandas as pd
import yfinance as yf
import config as cfg

def patch_shortlist(csv_file_path):
    print(f"Loading {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    if 'Company_Name' in df.columns:
        print("Company_Name column already exists!")
        return

    tickers = df['Ticker'].unique()
    print(f"Fetching names for {len(tickers)} unique tickers...")

    # Create a dictionary mapping Ticker -> Name
    name_map = {}
    for t in tickers:
        try:
            # Ping yfinance for just the basic info
            info = yf.Ticker(t).info
            name = info.get('shortName', info.get('longName', 'Unknown'))
            name_map[t] = name
            print(f"  {t}: {name}")
        except Exception as e:
            name_map[t] = "Unknown"
            print(f"  {t}: Failed to fetch")

    # Insert the new column right after 'Ticker' (usually index 1 or 2)
    ticker_col_idx = df.columns.get_loc('Ticker')
    df.insert(ticker_col_idx + 1, 'Company_Name', df['Ticker'].map(name_map))
    
    # Save it back
    df.to_csv(csv_file_path, index=False)
    print("\n✅ CSV successfully updated with Company Names!")

if __name__ == "__main__":
    # Change this filename if yours has a specific date appended!
    target_csv = cfg.REPORTS_DIR / "phase_b_candidates_new.csv" 
    patch_shortlist(target_csv)
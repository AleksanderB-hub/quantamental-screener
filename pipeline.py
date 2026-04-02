import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import random
import time
from datetime import timedelta
from tqdm import tqdm
import config as cfg

#set the seed for reproducibility
random.seed(42)

def _find_ticker_column(df, candidates):
    """Return the first candidate column name present in df, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None

def get_tickers_single(index_name: str = "sp500", custom_csv: str = None) -> list: # type: ignore
    """
    Returns ticker list based on index name.
    Supported: sp500, nasdaq100, ftse100, dax40, custom

    Custom reads a CSV with a 'Ticker' or 'Symbol' column.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    TICKER_COLS = ['Ticker', 'Symbol', 'Ticker symbol', 'EPIC', 'Code']

    index_name = index_name.lower()

    if index_name == "custom":
        if not custom_csv:
            raise ValueError("custom_csv path required when index_name='custom'")
        df = pd.read_csv(custom_csv)
        col = _find_ticker_column(df, TICKER_COLS)
        if col is None:
            raise ValueError(
                f"Custom CSV has no recognized ticker column. "
                f"Expected one of: {TICKER_COLS}. Found: {list(df.columns)}"
            )
        return df[col].dropna().str.replace('.', '-', regex=False).tolist()

    index_configs = {
        "sp500": {
            "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "table_idx": 0,
            "suffix": "",
        },
        "nasdaq100": {
            "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
            "table_idx": None,  # auto-detect
            "suffix": "",
        },
        "ftse100": {
            "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
            "table_idx": None,  # auto-detect
            "suffix": ".L",
        },
        "dax40": {
            "url": "https://en.wikipedia.org/wiki/DAX",
            "table_idx": None,  # auto-detect
            "suffix": ".DE",
        },
    }

    if index_name not in index_configs:
        raise ValueError(
            f"Unknown index '{index_name}'. "
            f"Supported: {list(index_configs.keys()) + ['custom']}"
        )

    idx_cfg = index_configs[index_name]
    tables = pd.read_html(idx_cfg["url"], storage_options=headers)

    # Locate the right table
    if idx_cfg["table_idx"] is not None:
        df = tables[idx_cfg["table_idx"]]
        col = _find_ticker_column(df, TICKER_COLS)
    else:
        df, col = None, None
        for table in tables:
            c = _find_ticker_column(table, TICKER_COLS)
            if c is not None:
                df, col = table, c
                break

    if df is None or col is None:
        raise RuntimeError(
            f"Could not find a ticker column in any table on the {index_name} Wikipedia page. "
            f"Looked for: {TICKER_COLS}"
        )

    # Replace internal dots with dashes (e.g. BRK.B -> BRK-B), then append exchange suffix
    raw = df[col].dropna().str.strip()
    suffix = idx_cfg["suffix"]
    if suffix:
        # Only add suffix to tickers that don't already have it
        tickers = [
            t.replace('.', '-') + suffix if not t.endswith(suffix) else t
            for t in raw
        ]
    else:
        tickers = raw.str.replace('.', '-', regex=False).tolist()

    return tickers

def get_tickers(index_name, custom_csv=None):
    """
    Returns ticker list. Accepts a single index string or a list of indices.
    Deduplicates across indices.
    """
    if isinstance(index_name, list):
        all_tickers = []
        for idx in index_name:
            print(f"  Fetching {idx} tickers...")
            all_tickers.extend(get_tickers_single(idx, custom_csv)) # type: ignore
        # Deduplicate while preserving order
        seen = set()
        unique = [t for t in all_tickers if not (t in seen or seen.add(t))]
        print(f"  Combined universe: {len(unique)} unique tickers from {len(index_name)} indices")
        return unique
    else:
        return get_tickers_single(index_name, custom_csv) # type: ignore

def get_sp500_tickers():
    """Backward-compatible wrapper — returns S&P 500 tickers via get_tickers()."""
    return get_tickers("sp500")

def calculate_rsi(series, period=14):
    '''Calculate RSI using Wilder's smoothing method.'''
    
    delta = series.diff()
    # Gains and Losses 
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Wilder's Exponential Weighted Mean (1/period = alpha)
    
    # Subsequent averages: exponential smoothing
    # Wilder's smoothing: alpha = 1/period
    # adjust = False to mathc the formula 
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # RS
    rs = avg_gain / avg_loss
    
    # RSI
    return 100 - (100 / (1 + rs))

def calculate_obv(df):
    '''Function to calculate on Balance Volume (OBV)'''
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df['OBV']

def get_robust_financials(ticker_symbol, hist, fin, bs, cash, info, screening_date):
    features = {}
    
    try:
        # 1. EARLY EXIT IF DATA IS EMPTY
        if hist.empty or fin.empty or bs.empty: 
            return None
        
        # --- UNIVERSAL HELPER: Get Single Value Safely ---
        def get_val(df, keys, iloc_idx=0):
            for k in keys:
                if k in df.index:
                    try:
                        val = df.loc[k].iloc[iloc_idx]
                        return val
                    except IndexError:
                        return np.nan
            return np.nan

        # --- 1a. TECHNICAL INDICATORS ---
        current_price = hist['Close'].iloc[-1] if not hist.empty else np.nan
        ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else np.nan
        ma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else np.nan
        
        # RSI (Weekly)
        hist_weekly = hist['Close'].resample('W-FRI').last()
        rsi_13_wk = calculate_rsi(hist_weekly, period=13).iloc[-1] if len(hist_weekly) >= 14 else np.nan
        rsi_25_wk = calculate_rsi(hist_weekly, period=25).iloc[-1] if len(hist_weekly) >= 26 else np.nan
        
        # OBV
        obv_series = calculate_obv(hist)
        obv_20d_change = obv_series.iloc[-1] - obv_series.iloc[-21] if len(obv_series) > 20 else np.nan
        
        # Technical Binary Rules (NaN propagated)
        features['Price_Gt_50MA'] = np.nan if (pd.isna(current_price) or pd.isna(ma_50)) else (1 if current_price > ma_50 else 0)
        features['50MA_Gt_200MA'] = np.nan if (pd.isna(ma_50) or pd.isna(ma_200)) else (1 if ma_50 > ma_200 else 0)
        features['RSI_13W_Gt_RSI_25W'] = np.nan if (pd.isna(rsi_13_wk) or pd.isna(rsi_25_wk)) else (1 if rsi_13_wk > rsi_25_wk else 0)
        features['OBV_20D_Positive'] = np.nan if pd.isna(obv_20d_change) else (1 if obv_20d_change > 0 else 0)

        # --- 1b. VALUATION & RATIOS (Calculated) ---
        eps_curr = get_val(fin, ["Basic EPS"], iloc_idx=0)
        eps_prev = get_val(fin, ["Basic EPS"], iloc_idx=1)
        eps_prev_2 = get_val(fin, ["Basic EPS"], iloc_idx=2)
        eps_prev_3 = get_val(fin, ["Basic EPS"], iloc_idx=3)
        
        trailing_pe = np.nan
        if pd.notna(current_price) and pd.notna(eps_curr) and eps_curr > 0:
            trailing_pe = current_price / eps_curr
            
        # DYNAMIC EPS GROWTH EXTRACTION
        eps_growth_dynamic = np.nan
        if pd.notna(eps_curr):
            if pd.notna(eps_prev_3) and abs(eps_prev_3) > 0: # Try 3-Year
                eps_growth_dynamic = (eps_curr / eps_prev_3)**(1/3) - 1 if (eps_curr > 0 and eps_prev_3 > 0) else ((eps_curr - eps_prev_3) / abs(eps_prev_3)) / 3
            elif pd.notna(eps_prev_2) and abs(eps_prev_2) > 0: # Fallback to 2-Year
                eps_growth_dynamic = (eps_curr / eps_prev_2)**(1/2) - 1 if (eps_curr > 0 and eps_prev_2 > 0) else ((eps_curr - eps_prev_2) / abs(eps_prev_2)) / 2
            elif pd.notna(eps_prev) and abs(eps_prev) > 0: # Fallback to 1-Year
                eps_growth_dynamic = (eps_curr / eps_prev) - 1 if (eps_curr > 0 and eps_prev > 0) else (eps_curr - eps_prev) / abs(eps_prev)

        features['EPS_3yr_Growth_Raw'] = eps_growth_dynamic

        # PEG Ratio
        manual_peg = trailing_pe / (eps_growth_dynamic * 100) if (pd.notna(trailing_pe) and pd.notna(eps_growth_dynamic) and eps_growth_dynamic > 0) else np.nan
        final_peg = manual_peg
        # PEG Rules: 0 if PEG can't be calculated (negative/zero growth = doesn't qualify)
        if pd.notna(final_peg):
            features['PEG_01_to_05'] = 1 if 0.1 < final_peg <= 0.5 else 0
            features['PEG_Less_1'] = 1 if final_peg < 1 else 0
            features['PEG_Less_1_5'] = 1 if final_peg < 1.5 else 0
        elif pd.isna(eps_curr):
            # Genuinely missing EPS data → NaN (unknown)
            features['PEG_01_to_05'] = np.nan
            features['PEG_Less_1'] = np.nan
            features['PEG_Less_1_5'] = np.nan
        else:
            # EPS exists but growth is negative/zero → rule not met
            features['PEG_01_to_05'] = 0
            features['PEG_Less_1'] = 0
            features['PEG_Less_1_5'] = 0
        
        # Shares Outstanding (Expanded Fallbacks)
        shares_out = get_val(bs, ['Ordinary Shares Number', 'Share Issued', 'Basic Average Shares', 'Diluted Average Shares', 'Common Stock Shares Outstanding'], iloc_idx=0)
        mkt_cap = current_price * shares_out if (pd.notna(current_price) and pd.notna(shares_out) and shares_out > 0) else np.nan
            
        # Price to Book (Handles Negative Equity naturally by yielding NaN below if equity <= 0)
        equity_curr = get_val(bs, ["Total Stockholder Equity", "Stockholders Equity"], iloc_idx=0)
        
        # else np.nan
        if pd.isna(mkt_cap) or pd.isna(equity_curr):
            pb = np.nan
            features['PB_Less_2'] = np.nan
        elif equity_curr <= 0:
            pb = 0 
            features['PB_Less_2'] = 0
        else:
            pb = mkt_cap / equity_curr
            features['PB_Less_2'] = 1 if pb < 2 else 0
        
        # Dividend Yield (Fix: Set to 0 if missing)
        div_paid = get_val(cash, ['Cash Dividends Paid', 'Dividends Paid', 'Cash Dividend Paid'], iloc_idx=0)
        div_yield = 0.0 if pd.isna(div_paid) else (abs(div_paid) / mkt_cap if (pd.notna(mkt_cap) and mkt_cap > 0) else np.nan)
        
        ocf_curr = get_val(cash, ["Operating Cash Flow", "Total Cash From Operating Activities"], iloc_idx=0)
        if pd.isna(mkt_cap) or pd.isna(ocf_curr):
            features['MC_to_CF_Less_3'] = np.nan
        elif ocf_curr <= 0:
            features['MC_to_CF_Less_3'] = 0  # Burning cash = rule not met
        else:
            features['MC_to_CF_Less_3'] = 1 if (mkt_cap / ocf_curr) < 3 else 0
            
        features['Zero_Dividend'] = np.nan if pd.isna(div_yield) else (1 if div_yield == 0 else 0)

        # --- 1c. FINANCIAL HEALTH ---
        debt_curr = get_val(bs, ["Total Debt And Capital Lease Obligation", "Total Debt"], iloc_idx=0)
        equity_prev = get_val(bs, ["Total Stockholder Equity", "Stockholders Equity"], iloc_idx=1)
        equity_prev_2 = get_val(bs, ["Total Stockholder Equity", "Stockholders Equity"], iloc_idx=2)
        
        features['DE_Less_1'] = np.nan if (pd.isna(debt_curr) or pd.isna(equity_curr) or equity_curr == 0) else (1 if (debt_curr / equity_curr) < 1 else 0)
            
        cash_curr = get_val(bs, ["Cash And Cash Equivalents", "Cash Financial"], iloc_idx=0)
        cash_prev = get_val(bs, ["Cash And Cash Equivalents", "Cash Financial"], iloc_idx=1)
        cliab_curr = get_val(bs, ["Current Liabilities", "Total Current Liabilities"], iloc_idx=0)
        cliab_prev = get_val(bs, ["Current Liabilities", "Total Current Liabilities"], iloc_idx=1)
        
        if pd.notna(cash_curr) and pd.notna(cliab_curr) and cliab_curr != 0:
            cash_ratio = cash_curr / cliab_curr
            features['Cash_Ratio_Gt_1'] = 1 if cash_ratio > 1 else 0
            features['Cash_Ratio_Improving'] = np.nan if (pd.isna(cash_prev) or pd.isna(cliab_prev) or cliab_prev == 0) else (1 if cash_ratio > (cash_prev / cliab_prev) else 0)
        else:
            features['Cash_Ratio_Gt_1'] = np.nan
            features['Cash_Ratio_Improving'] = np.nan

        # Free Cash Flow
        capex_curr = get_val(cash, ["Capital Expenditure", "Capital Expenditures", "Capital Expenditure Reported"], iloc_idx=0)
        capex_prev = get_val(cash, ["Capital Expenditure", "Capital Expenditures", "Capital Expenditure Reported"], iloc_idx=1)
        capex_prev_2 = get_val(cash, ["Capital Expenditure", "Capital Expenditures", "Capital Expenditure Reported"], iloc_idx=2)
        ocf_prev = get_val(cash, ["Operating Cash Flow", "Total Cash From Operating Activities"], iloc_idx=1)
        ocf_prev_2 = get_val(cash, ["Operating Cash Flow", "Total Cash From Operating Activities"], iloc_idx=2)
        
        fcf_curr = (ocf_curr + capex_curr) if (pd.notna(ocf_curr) and pd.notna(capex_curr)) else np.nan
        fcf_prev = (ocf_prev + capex_prev) if (pd.notna(ocf_prev) and pd.notna(capex_prev)) else np.nan
        fcf_prev_2 = (ocf_prev_2 + capex_prev_2) if (pd.notna(ocf_prev_2) and pd.notna(capex_prev_2)) else np.nan
        
        features['FCF_Positive'] = np.nan if pd.isna(fcf_curr) else (1 if fcf_curr > 0 else 0)
        features['FCF_Growing'] = np.nan if (pd.isna(fcf_curr) or pd.isna(fcf_prev)) else (1 if fcf_curr > fcf_prev else 0)
        
        # adjust dynamically
        if pd.isna(fcf_curr) or pd.isna(fcf_prev):
            features['FCF_Growing_Sustained'] = np.nan
        else:
            if pd.notna(fcf_prev_2): # Has 3 years
                features['FCF_Growing_Sustained'] = 1 if (fcf_curr > fcf_prev) and (fcf_prev > fcf_prev_2) else 0
            else: # Has 2 years
                features['FCF_Growing_Sustained'] = 1 if fcf_curr > fcf_prev else 0
        
        # OCF > Net Income
        ni_curr = get_val(fin, ["Net Income", "Net Income Common Stockholders"], iloc_idx=0)
        ni_prev = get_val(fin, ["Net Income", "Net Income Common Stockholders"], iloc_idx=1) 
        ni_prev_2 = get_val(fin, ["Net Income", "Net Income Common Stockholders"], iloc_idx=2)
        ni_prev_3 = get_val(fin, ["Net Income", "Net Income Common Stockholders"], iloc_idx=3)
        
        features['OCF_Gt_NetIncome'] = np.nan if (pd.isna(ocf_curr) or pd.isna(ni_curr)) else (1 if ocf_curr > ni_curr else 0)
        
        assets_curr = get_val(bs, ["Total Assets"], iloc_idx=0)
        features['ROA_Positive'] = np.nan if (pd.isna(ni_curr) or pd.isna(assets_curr) or assets_curr <= 0) else (1 if ni_curr > 0 else 0)
        
        # --- 1d. List 3: Historical Data --- # adjust dynamically if needed
        roe_curr = (ni_curr / equity_curr) if (pd.notna(ni_curr) and pd.notna(equity_curr)) else np.nan
        roe_prev = (ni_prev / equity_prev) if (pd.notna(ni_prev) and pd.notna(equity_prev)) else np.nan
        roe_prev_2 = (ni_prev_2 / equity_prev_2) if (pd.notna(ni_prev_2) and pd.notna(equity_prev_2)) else np.nan
        
        # 2. Apply logic checking for BOTH missing data and negative equity
        if pd.isna(roe_curr):
            features['ROE_Gt_15_Sustained'] = np.nan
        elif pd.notna(equity_curr) and float(equity_curr) <= 0:
            features['ROE_Gt_15_Sustained'] = 0 # Fails standard ROE test if current equity is negative
        else:
            if pd.notna(roe_prev_2): # 3 years of data exist
                if equity_prev <= 0 or equity_prev_2 <= 0:
                    features['ROE_Gt_15_Sustained'] = 0 # Fails if historical equity was negative
                else:
                    features['ROE_Gt_15_Sustained'] = 1 if (roe_curr > 0.15 and roe_prev > 0.15 and roe_prev_2 > 0.15) else 0
                    
            elif pd.notna(roe_prev): # Only 2 years of data exist
                if equity_prev <= 0:
                    features['ROE_Gt_15_Sustained'] = 0 # Fails if previous equity was negative
                else:
                    features['ROE_Gt_15_Sustained'] = 1 if (roe_curr > 0.15 and roe_prev > 0.15) else 0
                    
            else: # Only 1 year of data exists
                features['ROE_Gt_15_Sustained'] = 1 if roe_curr > 0.15 else 0

        if pd.isna(ni_curr):
            features['Net_Income_Growth_Gt_8pct'] = np.nan
        else:
            # Fails if baseline OR current income is negative/zero
            if pd.notna(ni_prev_3):
                features['Net_Income_Growth_Gt_8pct'] = 0 if (ni_prev_3 <= 0 or ni_curr <= 0) else (1 if ((ni_curr / ni_prev_3)**(1/3) - 1) > 0.08 else 0)
            elif pd.notna(ni_prev_2):
                features['Net_Income_Growth_Gt_8pct'] = 0 if (ni_prev_2 <= 0 or ni_curr <= 0) else (1 if ((ni_curr / ni_prev_2)**(1/2) - 1) > 0.08 else 0)
            elif pd.notna(ni_prev):
                features['Net_Income_Growth_Gt_8pct'] = 0 if (ni_prev <= 0 or ni_curr <= 0) else (1 if ((ni_curr / ni_prev) - 1) > 0.08 else 0)
            else:
                features['Net_Income_Growth_Gt_8pct'] = np.nan
            
        # Sales vs R&D
        rev_curr = get_val(fin, ['Total Revenue', 'Operating Revenue'], iloc_idx=0)
        rev_prev = get_val(fin, ['Total Revenue', 'Operating Revenue'], iloc_idx=1)
        rev_prev_2 = get_val(fin, ['Total Revenue', 'Operating Revenue'], iloc_idx=2)
        rev_prev_3 = get_val(fin, ['Total Revenue', 'Operating Revenue'], iloc_idx=3) 
        
        rnd_curr = get_val(fin, ["Research And Development"], iloc_idx=0)
        rnd_prev = get_val(fin, ["Research And Development"], iloc_idx=1)
        rnd_prev_2 = get_val(fin, ["Research And Development"], iloc_idx=2)
        rnd_prev_3 = get_val(fin, ["Research And Development"], iloc_idx=3)
        
        sales_vs_rnd_flag = np.nan 
        if pd.notna(rev_curr) and pd.notna(rev_prev_3) and rev_prev_3 > 0 and pd.notna(rnd_curr) and pd.notna(rnd_prev_3) and rnd_prev_3 > 0:
            sales_vs_rnd_flag = 1 if ((rev_curr / rev_prev_3)**(1/3) - 1) > ((rnd_curr / rnd_prev_3)**(1/3) - 1) else 0
        elif pd.notna(rev_curr) and pd.notna(rev_prev_2) and rev_prev_2 > 0 and pd.notna(rnd_curr) and pd.notna(rnd_prev_2) and rnd_prev_2 > 0:
            sales_vs_rnd_flag = 1 if ((rev_curr / rev_prev_2)**(1/2) - 1) > ((rnd_curr / rnd_prev_2)**(1/2) - 1) else 0
        elif pd.notna(rev_curr) and pd.notna(rev_prev) and rev_prev > 0 and pd.notna(rnd_curr) and pd.notna(rnd_prev) and rnd_prev > 0:
            sales_vs_rnd_flag = 1 if ((rev_curr / rev_prev) - 1) > ((rnd_curr / rnd_prev) - 1) else 0
        elif pd.isna(rnd_curr) or rnd_curr == 0:
             if pd.notna(rev_curr) and pd.notna(rev_prev):
                 sales_vs_rnd_flag = 1 if rev_curr > rev_prev else 0

        features['Sales_Growth_Gt_RnD_Growth'] = sales_vs_rnd_flag
        
        # Sales Growing Streak (dynamic: 3yr -> 2yr)
        if pd.notna(rev_curr) and pd.notna(rev_prev):
            if pd.notna(rev_prev_2):  # Has 3 years
                features['Sales_Growing_Sustained'] = 1 if (rev_curr > rev_prev and rev_prev > rev_prev_2) else 0
            else:  # Has 2 years
                features['Sales_Growing_Sustained'] = 1 if rev_curr > rev_prev else 0
        else:
            features['Sales_Growing_Sustained'] = np.nan
        
        op_inc_curr = get_val(fin, ['Operating Income'], iloc_idx=0)
        op_inc_prev = get_val(fin, ['Operating Income'], iloc_idx=1)
        op_inc_prev_2 = get_val(fin, ['Operating Income'], iloc_idx=2)
        
        om_curr = (op_inc_curr / rev_curr) if (pd.notna(op_inc_curr) and pd.notna(rev_curr)) else np.nan
        om_prev = (op_inc_prev / rev_prev) if (pd.notna(op_inc_prev) and pd.notna(rev_prev)) else np.nan
        om_prev_2 = (op_inc_prev_2 / rev_prev_2) if (pd.notna(op_inc_prev_2) and pd.notna(rev_prev_2)) else np.nan
        
        available_margins = [m for m in [om_curr, om_prev, om_prev_2] if pd.notna(m)]
        if len(available_margins) >= 2: # Requires at least current and 1 past year
            avg_om = sum(available_margins) / len(available_margins)
            features['Op_Margin_Gt_Hist_Avg'] = 1 if om_curr > avg_om else 0
        else:
            features['Op_Margin_Gt_Hist_Avg'] = np.nan
        
        # --- 2. METADATA (Raw Data for List 2) ---
        gross_profit_curr = get_val(fin, ['Gross Profit'], iloc_idx=0)
        total_liab_curr = get_val(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities'], iloc_idx=0)
        if pd.isna(total_liab_curr):
             total_liab_curr = (assets_curr - equity_curr) if (pd.notna(assets_curr) and pd.notna(equity_curr)) else np.nan

        features['Ticker'] = ticker_symbol
        features['Company_Name'] = info.get('shortName', info.get('longName', 'Unknown'))
        features['Sector'] = info.get('sector', 'Unknown')
        features['Market_Cap_Raw'] = mkt_cap 
        features['PE_Raw'] = trailing_pe
        features['Free_Cash_Flow_Raw'] = fcf_curr
        features['PB_Raw'] = pb
        features['Div_Yield_Raw'] = div_yield
        
        # ROE HYBRID EXTRACTION (no .info, to avoid lookahead bias)
        features['ROE_Raw'] = ni_curr / equity_curr if (pd.notna(ni_curr) and pd.notna(equity_curr) and equity_curr > 0) else np.nan


        # C. Calculate Raw Ratios
        features['Net_Profit_Margin_Raw'] = ni_curr / rev_curr if (pd.notna(ni_curr) and pd.notna(rev_curr) and rev_curr > 0) else np.nan
        features['Net_Profit_Margin_Prev_Raw'] = ni_prev / rev_prev if (pd.notna(ni_prev) and pd.notna(rev_prev) and rev_prev > 0) else np.nan
        features['Net_Profit_Margin_Prev_2_Raw'] = ni_prev_2 / rev_prev_2 if (pd.notna(ni_prev_2) and pd.notna(rev_prev_2) and rev_prev_2 > 0) else np.nan
        features['Gross_Profit_Margin_Raw'] = gross_profit_curr / rev_curr if (pd.notna(gross_profit_curr) and pd.notna(rev_curr) and rev_curr > 0) else np.nan
        features['Operating_Margin_Raw'] = op_inc_curr / rev_curr if (pd.notna(op_inc_curr) and pd.notna(rev_curr) and rev_curr > 0) else np.nan

        # DYNAMIC SALES GROWTH EXTRACTION
        if pd.notna(rev_curr):
            if pd.notna(rev_prev_3) and rev_prev_3 > 0:
                features['Sales_3yr_Growth_Raw'] = (rev_curr / rev_prev_3)**(1/3) - 1
            elif pd.notna(rev_prev_2) and rev_prev_2 > 0:
                features['Sales_3yr_Growth_Raw'] = (rev_curr / rev_prev_2)**(1/2) - 1
            elif pd.notna(rev_prev) and rev_prev > 0:
                features['Sales_3yr_Growth_Raw'] = (rev_curr / rev_prev) - 1
            else:
                features['Sales_3yr_Growth_Raw'] = np.nan
        else:
            features['Sales_3yr_Growth_Raw'] = np.nan

        # Debt & Solvency Ratios
        features['Debt_To_Assets_Ratio_Raw'] = total_liab_curr / assets_curr if (pd.notna(total_liab_curr) and pd.notna(assets_curr) and assets_curr > 0) else np.nan
        features['Assets_To_Liability_Ratio_Raw'] = assets_curr / total_liab_curr if (pd.notna(assets_curr) and pd.notna(total_liab_curr) and total_liab_curr > 0) else np.nan
        features['Long_Term_Debt_To_Equity_Raw'] = debt_curr / equity_curr if (pd.notna(debt_curr) and pd.notna(equity_curr) and equity_curr != 0) else np.nan
        features['Sales_To_Assets_Ratio_Raw'] = rev_curr / assets_curr if (pd.notna(rev_curr) and pd.notna(assets_curr) and assets_curr > 0) else np.nan

        features['Current_EPS_Change_Raw'] = (eps_curr - eps_prev) / abs(eps_prev) if (pd.notna(eps_curr) and pd.notna(eps_prev) and abs(eps_prev) > 0) else np.nan
             
        features['Screening_Date'] = screening_date

        return features
    
    except Exception as e:
        return None

def calculate_list_2_rules(df):
    """Calculates Industry Relative and Percentile Rules (with strict NaN propagation)"""
    print("--- Calculating List 2 (Industry/Universe) Rules ---")

    multi_index = 'Index_Source' in df.columns

    # Percentile ranking group key: per-index when multi-index, otherwise date-only
    if multi_index:
        pct_group = ['Index_Source', 'Screening_Date']
    else:
        pct_group = ['Screening_Date']

    # Industry-relative group key: always global across all indices
    industry_group = ['Screening_Date', 'Sector']

    # --- 1. Percentile Rankings (Universe Wide) grouped by screening date ---
    df['Market_Cap_Percentile'] = df.groupby(pct_group)['Market_Cap_Raw'].rank(pct=True)
    df['Market_Cap_Top_30_Pct'] = np.where(df['Market_Cap_Raw'].isna(), np.nan, (df['Market_Cap_Percentile'] >= 0.70).astype(float))
    df['Market_Cap_Top_25_Pct'] = np.where(df['Market_Cap_Raw'].isna(), np.nan, (df['Market_Cap_Percentile'] >= 0.75).astype(float))

    df['PE_Percentile'] = df.groupby(pct_group)['PE_Raw'].rank(pct=True)
    df['PE_Bottom_40_Pct'] = np.where(df['PE_Raw'].isna(), np.nan, (df['PE_Percentile'] <= 0.40).astype(float))
    df['PE_Bottom_20_Pct'] = np.where(df['PE_Raw'].isna(), np.nan, (df['PE_Percentile'] <= 0.20).astype(float))

    df['Free_Cash_Flow_Percentile'] = df.groupby(pct_group)['Free_Cash_Flow_Raw'].rank(pct=True)
    df['FCF_Top_30_Pct'] = np.where(df['Free_Cash_Flow_Raw'].isna(), np.nan, (df['Free_Cash_Flow_Percentile'] >= 0.70).astype(float))

    # --- 2. Industry Relative Rules (Sector Specific) ---
    if 'Sector' in df.columns and 'Screening_Date' in df.columns:
        # PE < Industry Median
        df['Sector_Median_PE'] = df.groupby(industry_group)['PE_Raw'].transform('median')
        df['PE_Below_Industry'] = np.where(df['PE_Raw'].isna() | df['Sector_Median_PE'].isna(), np.nan, (df['PE_Raw'] < df['Sector_Median_PE']).astype(float))

        # PB < Industry Median
        df['Sector_Median_PB'] = df.groupby(industry_group)['PB_Raw'].transform('median')
        df['PB_Below_Industry'] = np.where(df['Sector_Median_PB'].isna(), np.nan, np.where(df['PB_Raw'].isna(), 0.0, (df['PB_Raw'] < df['Sector_Median_PB']).astype(float)))

        # Dividend Yield > Industry Median
        df['Sector_Median_Div_Yield'] = df.groupby(industry_group)['Div_Yield_Raw'].transform('median')
        df['Div_Yield_Above_Industry'] = np.where(df['Div_Yield_Raw'].isna() | df['Sector_Median_Div_Yield'].isna(), np.nan, (df['Div_Yield_Raw'] > df['Sector_Median_Div_Yield']).astype(float))

        # Margin > Industry Median
        df['Sector_Median_Margin'] = df.groupby(industry_group)['Net_Profit_Margin_Raw'].transform('median')
        df['Margin_Above_Industry'] = np.where(df['Net_Profit_Margin_Raw'].isna() | df['Sector_Median_Margin'].isna(), np.nan, (df['Net_Profit_Margin_Raw'] > df['Sector_Median_Margin']).astype(float))

        # Gross Margin > Industry Median
        df['Sector_Median_Gross_Margin'] = df.groupby(industry_group)['Gross_Profit_Margin_Raw'].transform('median')
        df['Gross_Margin_Above_Industry'] = np.where(df['Gross_Profit_Margin_Raw'].isna() | df['Sector_Median_Gross_Margin'].isna(), np.nan, (df['Gross_Profit_Margin_Raw'] > df['Sector_Median_Gross_Margin']).astype(float))

        # Operating Margin > Industry Median
        df['Sector_Median_Operating_Margin'] = df.groupby(industry_group)['Operating_Margin_Raw'].transform('median')
        df['Operating_Margin_Above_Industry'] = np.where(df['Operating_Margin_Raw'].isna() | df['Sector_Median_Operating_Margin'].isna(), np.nan, (df['Operating_Margin_Raw'] > df['Sector_Median_Operating_Margin']).astype(float))

        # ROE > Industry Average
        df['Sector_Median_ROE'] = df.groupby(industry_group)['ROE_Raw'].transform('median')
        df['ROE_Above_Industry'] = np.where(df['Sector_Median_ROE'].isna(), np.nan, np.where(df['ROE_Raw'].isna(), 0.0, (df['ROE_Raw'] > df['Sector_Median_ROE']).astype(float)))

        # Ratio of T Liabilities to T Assets < Industry
        df['Sector_Median_Debt_To_Assets_Ratio'] = df.groupby(industry_group)['Debt_To_Assets_Ratio_Raw'].transform('median')
        df['Debt_To_Assets_Ratio_Below_Industry'] = np.where(df['Debt_To_Assets_Ratio_Raw'].isna() | df['Sector_Median_Debt_To_Assets_Ratio'].isna(), np.nan, (df['Debt_To_Assets_Ratio_Raw'] < df['Sector_Median_Debt_To_Assets_Ratio']).astype(float))

        # Ratio of Long Term Debt to Equity < Industry
        df['Sector_Median_Long_Term_Debt_To_Equity'] = df.groupby(industry_group)['Long_Term_Debt_To_Equity_Raw'].transform('median')
        df['Long_Term_Debt_To_Equity_Ratio_Below_Industry'] = np.where(df['Long_Term_Debt_To_Equity_Raw'].isna() | df['Sector_Median_Long_Term_Debt_To_Equity'].isna(), np.nan, (df['Long_Term_Debt_To_Equity_Raw'] < df['Sector_Median_Long_Term_Debt_To_Equity']).astype(float))

        # Ratio of T Assets to T Liabilities > Industry
        df['Sector_Median_Assets_To_Liability'] = df.groupby(industry_group)['Assets_To_Liability_Ratio_Raw'].transform('median')
        df['Assets_To_Liability_Ratio_Above_Industry'] = np.where(df['Assets_To_Liability_Ratio_Raw'].isna() | df['Sector_Median_Assets_To_Liability'].isna(), np.nan, (df['Assets_To_Liability_Ratio_Raw'] > df['Sector_Median_Assets_To_Liability']).astype(float))

        # Ratio of Sales to T Assets > Industry
        df['Sector_Median_Sales_To_Assets'] = df.groupby(industry_group)['Sales_To_Assets_Ratio_Raw'].transform('median')
        df['Sales_To_Assets_Ratio_Above_Industry'] = np.where(df['Sales_To_Assets_Ratio_Raw'].isna() | df['Sector_Median_Sales_To_Assets'].isna(), np.nan, (df['Sales_To_Assets_Ratio_Raw'] > df['Sector_Median_Sales_To_Assets']).astype(float))

        # Growth > Industry Median
        df['Sector_Median_Growth'] = df.groupby(industry_group)['Sales_3yr_Growth_Raw'].transform('median')
        df['Growth_Above_Industry'] = np.where(df['Sales_3yr_Growth_Raw'].isna() | df['Sector_Median_Growth'].isna(), np.nan, (df['Sales_3yr_Growth_Raw'] > df['Sector_Median_Growth']).astype(float))

        # EPS Growth > Industry
        df['Sector_Median_EPS_Growth'] = df.groupby(industry_group)['EPS_3yr_Growth_Raw'].transform('median')
        df['EPS_Growth_Above_Industry'] = np.where(df['EPS_3yr_Growth_Raw'].isna() | df['Sector_Median_EPS_Growth'].isna(), np.nan, (df['EPS_3yr_Growth_Raw'] > df['Sector_Median_EPS_Growth']).astype(float))

        # Current Year EPS % Change > Industry
        df['Sector_Median_EPS_Current_Change'] = df.groupby(industry_group)['Current_EPS_Change_Raw'].transform('median')
        df['EPS_Current_Change_Above_Industry'] = np.where(df['Current_EPS_Change_Raw'].isna() | df['Sector_Median_EPS_Current_Change'].isna(), np.nan, (df['Current_EPS_Change_Raw'] > df['Sector_Median_EPS_Current_Change']).astype(float))

        # --- 3-Year Margin Streak Rule ---
        has_curr = df['Net_Profit_Margin_Raw'].notna() & df['Sector_Median_Margin'].notna()
        mask_curr_margin = df['Net_Profit_Margin_Raw'] > df['Sector_Median_Margin']

        df['Sector_Median_Margin_Prev'] = df.groupby(industry_group)['Net_Profit_Margin_Prev_Raw'].transform('median')
        has_prev = df['Net_Profit_Margin_Prev_Raw'].notna() & df['Sector_Median_Margin_Prev'].notna()
        mask_prev_margin = df['Net_Profit_Margin_Prev_Raw'] > df['Sector_Median_Margin_Prev']

        df['Sector_Median_Margin_Prev_2_yr'] = df.groupby(industry_group)['Net_Profit_Margin_Prev_2_Raw'].transform('median')
        has_prev_2 = df['Net_Profit_Margin_Prev_2_Raw'].notna() & df['Sector_Median_Margin_Prev_2_yr'].notna()
        mask_prev_2_margin = df['Net_Profit_Margin_Prev_2_Raw'] > df['Sector_Median_Margin_Prev_2_yr']

        streak_3yr = mask_curr_margin & mask_prev_margin & mask_prev_2_margin
        streak_2yr = mask_curr_margin & mask_prev_margin

        # If it has 3 years of data, test the 3yr streak. Otherwise, test the 2yr streak.
        df['Margin_Gt_Industry_Sustained'] = np.where(
            ~has_curr | ~has_prev, 
            np.nan, # Fails if it doesn't even have 2 years of history
            np.where(has_prev_2, streak_3yr.astype(float), streak_2yr.astype(float))
        )   
        
    # 3. Cleanup: Drop temporary medians and percentiles
    cols_to_drop = [
        'Market_Cap_Percentile', 'PE_Percentile', 'Free_Cash_Flow_Percentile',
        'Sector_Median_PE', 'Sector_Median_PB', 'Sector_Median_Div_Yield',
        'Sector_Median_Margin', 'Sector_Median_Gross_Margin', 'Sector_Median_Operating_Margin',
        'Sector_Median_ROE', 'Sector_Median_Debt_To_Assets_Ratio', 
        'Sector_Median_Long_Term_Debt_To_Equity', 'Sector_Median_Assets_To_Liability',
        'Sector_Median_Sales_To_Assets', 'Sector_Median_Growth', 'Sector_Median_EPS_Growth',
        'Sector_Median_EPS_Current_Change', 'Sector_Median_Margin_Prev', 'Sector_Median_Margin_Prev_2_yr'
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    print("Success: List 2 Rules Added (NaNs propagated securely).")
    return df


def process_historical_snapshot(ticker, full_hist, full_fin, full_bs, full_cash, info, screening_date_str, bench_hist=None):

    """
    Slices data to simulate point-in-time knowledge, extracts features, 
    and calculates the 6-month forward return for Machine Learning labels.
    """
    screening_date = pd.to_datetime(screening_date_str)
    
    # Strip timezones
    if full_hist is not None and not full_hist.empty and full_hist.index.tz is not None:
        full_hist.index = full_hist.index.tz_localize(None)
    
    # 1. SLICE PRICE HISTORY (For Features)
    hist_sliced = full_hist[full_hist.index <= screening_date].copy()
    if len(hist_sliced) < cfg.MIN_PRICE_HISTORY:
        return None 

    # 2. FILTER FINANCIALS (90-Day Lag)
    cutoff_date = screening_date - timedelta(days=cfg.REPORTING_LAG_DAYS)
    
    def filter_statement(df):
        if df is None or df.empty: return df
        valid_cols = [col for col in df.columns if pd.to_datetime(col) <= cutoff_date]
        return df[valid_cols]

    fin_filtered = filter_statement(full_fin)
    bs_filtered = filter_statement(full_bs)
    cash_filtered = filter_statement(full_cash)

    # 3. EXTRACT FEATURES
    features = get_robust_financials(
        ticker_symbol=ticker, 
        hist=hist_sliced, 
        fin=fin_filtered, 
        bs=bs_filtered, 
        cash=cash_filtered, 
        info=info, 
        screening_date=screening_date_str
    )
    
    # 4. CALCULATE THE ML LABEL (6-Month Forward Return)
    if features is not None:
        forward_date = screening_date + pd.DateOffset(months=6)
        current_price = hist_sliced.iloc[-1]['Close']
        
        future_hist = full_hist[full_hist.index >= forward_date]
        
        if not future_hist.empty:
            future_price = future_hist.iloc[0]['Close']
            stock_return = (future_price - current_price) / current_price
            features['Forward_6m_Return'] = stock_return
            
            # Excess return over benchmark for the same window
            if bench_hist is not None and not bench_hist.empty:
                bench_at_screen = bench_hist[bench_hist.index <= screening_date]
                bench_at_forward = bench_hist[bench_hist.index >= forward_date]

                if not bench_at_screen.empty and not bench_at_forward.empty:
                    bench_return = (bench_at_forward.iloc[0]['Close'] - bench_at_screen.iloc[-1]['Close']) / bench_at_screen.iloc[-1]['Close']
                    features['Forward_6m_Excess_Return'] = stock_return - bench_return
                else:
                    features['Forward_6m_Excess_Return'] = np.nan
            else:
                features['Forward_6m_Excess_Return'] = np.nan
        else:
            features['Forward_6m_Return'] = np.nan
            features['Forward_6m_Excess_Return'] = np.nan
            
    return features
    
def run_backtest_pipeline(tickers, screening_dates, benchmark_ticker=None, desc=None):
    """
    Main loop to generate multiple historical snapshots for your dataset.
    """
    verbose = cfg.VERBOSE
    all_snapshots = []
    iterator = tqdm(tickers, desc=desc) if not verbose else tickers

    # Fetch benchmark (index-aware)
    bench = benchmark_ticker or "SPY"
    
    if verbose:
        print(f"Fetching {bench} benchmark data...")
    bench_ticker = yf.Ticker(bench)
    bench_hist = bench_ticker.history(period="max")
    if bench_hist.index.tz is not None: # type: ignore
        bench_hist.index = pd.DatetimeIndex(bench_hist.index).tz_localize(None)

    if bench_hist.empty:
        print(f"  WARNING: Could not fetch benchmark {bench}. Excess returns will be NaN.")
        
    for ticker in iterator:
        if verbose:
            print(f"Fetching raw data for {ticker}...")
            
        stock = yf.Ticker(ticker)
        
        try:
            full_hist = stock.history(period="max")
            full_fin = stock.financials
            full_bs = stock.balance_sheet
            full_cash = stock.cashflow
            info = stock.info
        except Exception as e:
            if verbose:
                print(f"  Failed to fetch {ticker}: {e}")
            continue
            
        for date_str in screening_dates:
            row_data = process_historical_snapshot(
                ticker, full_hist, full_fin, full_bs, full_cash, info, date_str, bench_hist=bench_hist
            )
            
            if row_data:
                all_snapshots.append(row_data)
        
        # Throttle to avoid yfinance rate limiting
        time.sleep(random.uniform(0.5, 1.0))
                
    final_df = pd.DataFrame(all_snapshots)
    return final_df

def audit_extraction(df, dataset_name):
    """
    Analyzes a cleaned ML DataFrame for missing values in the feature columns.
    """
    print(f"\n===EXTRACTION AUDIT: {dataset_name} ===")
    print(f"Total Snapshots Captured: {len(df)}")
    
    # Exclude identifiers and the target label from the feature audit
    ignore_cols = ['Ticker', 'Sector', 'Screening_Date', 'Forward_6m_Return', 'Forward_6m_Excess_Return']
    feature_cols = [col for col in df.columns if col not in ignore_cols]
    
    # Calculate percentage of missing data per FEATURE column
    missing_by_col = df[feature_cols].isnull().mean() * 100
    missing_cols = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    
    if not missing_cols.empty:
        print(f"WARNING: The following ML features are missing data in {dataset_name}:")
        print(missing_cols.head(15).round(1).astype(str) + '%')
    else:
        print(f"100% Data Retrieval for {dataset_name} Features! No missing values detected.")
        
    # Generate the detailed CSV report for this specific dataset
    file_name = f"missing_data_report_{dataset_name.replace(' ', '_')}.csv"
    
    audit_file = cfg.REPORTS_DIR / file_name
    
    def get_missing_features(row):
        # Only list the missing features, not missing identifiers/labels
        missing = row[feature_cols].isna()
        return ", ".join(missing.index[missing].tolist())
        
    audit_df = df[['Ticker', 'Screening_Date', 'Sector']].copy()
    audit_df['Missing_Feature_Count'] = df[feature_cols].isnull().sum(axis=1)
    audit_df['Missing_Features_List'] = df.apply(get_missing_features, axis=1)
    
    # Sort by the worst offenders
    audit_df = audit_df.sort_values(by=['Missing_Feature_Count', 'Ticker'], ascending=[False, True])
    
    audit_df.to_csv(audit_file, index=False)
    print(f"Detailed row-by-row audit saved to: {audit_file}\n")

def main():
    conn = sqlite3.connect(cfg.DB_PATH)
    
    print("Fetching S&P 500 tickers...")
    all_tickers = get_sp500_tickers()
    
    if cfg.TICKER_SAMPLE_SIZE is not None:
        random.seed(cfg.RANDOM_SEED)
        tickers = random.sample(all_tickers, cfg.TICKER_SAMPLE_SIZE)
        print(f"Testing Pipeline on sample of {cfg.TICKER_SAMPLE_SIZE}: {tickers}")
    else:
        tickers = all_tickers
        print(f"Running FULL extraction on {len(tickers)} tickers")
    
    # Training
    print(f"\n[1/2] Building TRAINING Dataset ({cfg.TRAIN_DATES[0]} to {cfg.TRAIN_DATES[-1]})...")
    train_df_raw = run_backtest_pipeline(tickers, cfg.TRAIN_DATES)
    
    if not train_df_raw.empty:
        train_df_final = calculate_list_2_rules(train_df_raw)
        
        # cleanup (we only way rules and tickers)
        cols_to_drop = [col for col in train_df_final.columns if col.endswith('_Raw')]
        ml_ready_train_df = train_df_final.drop(columns=cols_to_drop)
        rule_cols = [col for col in ml_ready_train_df.columns if col not in ['Ticker', 'Sector', 'Screening_Date', 'Forward_6m_Return']]
        ml_ready_train_df[rule_cols] = ml_ready_train_df[rule_cols].astype(float)
        
        # save the dataset
        ml_ready_train_df.to_sql('training_data', conn, if_exists='replace', index=False)
        ml_ready_train_df.to_csv(cfg.TRAINING_CSV, index=False)
        
        # run the audit function on the train set 
        audit_extraction(ml_ready_train_df, "Training Set")
        print(f"Training Dataset saved! ({len(train_df_final)} rows)")
    
    # ==========================================
    # PHASE B: BUILD TESTING DATASET
    # ==========================================
    print(f"\n[2/2] Building TESTING Dataset ({cfg.TEST_DATES[0]} to {cfg.TEST_DATES[-1]})...")
    test_df_raw = run_backtest_pipeline(tickers, cfg.TEST_DATES)
    
    if not test_df_raw.empty:
        test_df_final = calculate_list_2_rules(test_df_raw)
        cols_to_drop = [col for col in test_df_final.columns if col.endswith('_Raw')]
        ml_ready_test_df = test_df_final.drop(columns=cols_to_drop)
        rule_cols = [col for col in ml_ready_test_df.columns if col not in ['Ticker', 'Sector', 'Screening_Date', 'Forward_6m_Return']]
        ml_ready_test_df[rule_cols] = ml_ready_test_df[rule_cols].astype(float)
        ml_ready_test_df.to_sql('testing_data', conn, if_exists='replace', index=False)
        ml_ready_test_df.to_csv(cfg.TESTING_CSV, index=False)
        print(f"Testing Dataset saved! ({len(test_df_final)} rows)")
        
        # run the audit function on the test set
        audit_extraction(ml_ready_test_df, "Testing Set")

    print("\nPipeline Complete")
    conn.close()

if __name__ == "__main__":
    main()
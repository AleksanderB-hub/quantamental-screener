# Quantamental Stock Screener (Python + SQLite)

## 📌 Project Overview
This project is an automated, robust stock screening engine designed to evaluate companies based on a strict set of Technical, Fundamental, and Industry-Relative rules. It uses `yfinance` to fetch live and historical accounting data, stores it safely in a local SQLite database, and runs a multi-phase rules engine to identify high-quality, reasonably priced growth stocks.

## 🏗 System Architecture
The screener operates in two distinct phases to solve the "Industry Median" calculation problem (you cannot know the industry average until you have downloaded data for the whole industry).

* **Phase 1: Data Collection & Absolute Rules (List 1 & 3)**
    * Iterates through a list of stock tickers.
    * Fetches Price History, Financials, Balance Sheet, and Cash Flow via `yfinance`.
    * Calculates absolute metrics (e.g., "Is P/E < 15?", "Is Debt/Equity < 1?").
    * Saves the raw values and binary rule scores (1 = Pass, 0 = Fail) to `market_data_universe.db`.
* **Phase 2: Contextual Analysis (List 2)**
    * Triggered by the `CALCULATE_LIST_2 = True` flag.
    * Loads the entire universe from the database into a Pandas DataFrame.
    * Uses `groupby('Sector')` to calculate industry medians.
    * Scores each stock against its peers (e.g., "Is Margin > Sector Median?").
    * Outputs the final dataset to `final_strategy_results.csv`.

---

## 📜 The Rules Engine

### List 1: Absolute Company Data (Snapshot)
Calculated stock-by-stock during data collection.
* **Technicals:** Price > 50MA, 50MA > 200MA, 13W RSI > 25W RSI, OBV slope positive.
* **Valuation:** 0.1 < PEG <= 0.5, PEG < 1.0, PEG < 1.5, P/B < 2, Dividend Yield = 0%, Market Cap / OCF < 3.
* **Financial Health:** Debt/Equity < 1, Cash Ratio > 1, Current FCF > Prev FCF, OCF > Net Income, FCF > 0, ROA > 0, Cash Ratio Improving.

### List 2: Industry & Universe Relative
Calculated via batch processing after data collection.
* **Universe Rankings:** Market Cap in Top 30% / 25%, FCF in Top 30%, P/E in Bottom 40%.
* **Industry Valuation:** P/E < Industry Median, P/B < Industry Median, Div Yield > Industry Median.
* **Industry Profitability:** Net Margin, Gross Margin, Operating Margin, and ROE > Industry Median.
* **Industry Structure:** Liabilities/Assets < Median, LT Debt/Equity < Median, Assets/Liabilities > Median, Asset Turnover (Sales/Assets) > Median.
* **Industry Growth:** 3-Year Sales Growth > Median, 3-Year EPS Growth > Median, Current EPS Change > Median.
* **The Streak:** Net Margin > Industry Median for 3 consecutive years.

### List 3: Historical Consistency (3-Year Streaks)
Adapted from 5-year rules to fit `yfinance` API limits (4-year maximum history).
* **Growth Streaks:** Sales growing 3 years in a row, FCF growing 3 years in a row.
* **Profitability:** ROE > 15% for the past 3 years.
* **Targeted Growth:** 3-Year Net Income CAGR > 8%, 3-Year Sales CAGR > 3-Year R&D CAGR.
* **Averages:** Current Operating Margin > 3-Year Average.

---

## 🧮 Key Math & Calculation Logic
To ensure robustness against messy financial data, the script uses specific mathematical handling:

1.  **The "Safe Fetch" (`get_val`)**: Bypasses `IndexError` crashes if a company has less than 4 years of operating history by safely returning `NaN`.
2.  **Handling Negative Earnings Growth**: 
    * Standard CAGR `(End/Start)^(1/3) - 1` is used for profitable growth.
    * If earnings are negative (e.g., Loss narrowing or Turnarounds), the script falls back to an **Annualized Linear Growth** formula: `((End - Start) / |Start|) / 3`. This prevents math errors while preserving the correct ranking order for List 2.
3.  **Missing P/E Protection**: If Yahoo Finance fails to provide a `trailingPE`, the script manually calculates `Current Price / Basic EPS`. Crucially, it forces P/E to `NaN` if EPS is negative to prevent broken valuation rankings.
4.  **R&D vs. Sales Growth**: Synchronizes timeframes. If a company only has 2 years of R&D data, the script compares it against 2 years of Sales data to ensure an "apples-to-apples" comparison.

---

## 📂 File Structure & Toolkit

* **`main.py`**: The central orchestrator. Contains the collection loop, database saving logic, and the `CALCULATE_LIST_2` switch for Phase 1 vs. Phase 2.
* **`engine.py`** *(or grouped in main)*: Contains the heavy-lifting functions:
    * `get_robust_financials(stock)`: Executes List 1 and 3.
    * `calculate_list_2_rules(df)`: Executes List 2.
    * `calculate_rsi()`, `calculate_obv()`: Technical indicator helpers.
* **`generate_audit_report.py`**: A standalone diagnostic tool. Reads the database and generates a CSV detailing exactly which data points (if any) are missing for every single scraped ticker.
* **`market_data_universe.db`**: The SQLite database where raw and calculated data is accumulated safely.
* **`final_strategy_results.csv`**: The ultimate output containing the binary scores (0 or 1) for every rule across the screened universe.

---

## 🚀 Workflow Guide

1.  **Mass Data Collection:** Set `CALCULATE_LIST_2 = False` in `main.py`. Feed it massive lists of tickers (S&P 500, Russell 2000). Run it in batches. The DB handles duplicate prevention and appending.
2.  **Data Quality Audit:** Run `generate_audit_report.py` to identify sectors with systemic missing data or to purge "ghost" tickers (e.g., newly listed SPACs with no history).
3.  **Contextual Generation:** Set `CALCULATE_LIST_2 = True` in `main.py` and run it once. It will read the entire database, rank everything, and spit out the finalized CSV ready for portfolio selection.
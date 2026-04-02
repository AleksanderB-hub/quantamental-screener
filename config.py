# config.py
from pathlib import Path
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime as dt

# Load the hidden .env file
load_dotenv()

# Securely grab the key
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "ml_stock_pipeline.db"
TRAINING_CSV = DATA_DIR / "ML_Training_Data.csv"
TESTING_CSV = DATA_DIR / "ML_Testing_Data.csv"

# ── Data Collection ────────────────────────────────────
REPORTING_LAG_DAYS = 90        # Needs to be set up to 0 for live data extraction, otherwise 90
MIN_PRICE_HISTORY = 200        # minimum trading days needed

# ── Ticker Selection ───────────────────────────────────
# Set to None for full S&P 500, or an integer for random sample
TICKER_SAMPLE_SIZE = None      # None = all, 10 = test mode
RANDOM_SEED = 42

# ── Index Selection ────────────────────────────────────
LIVE_INDEX = ["sp500", "nasdaq100", "ftse100", "dax40"]       # Example: LIVE_INDEX = ["sp500", "nasdaq100"]
CUSTOM_TICKERS_CSV = None     # Path to CSV with Ticker or Symbol column (used when INDEX=custom)
SUPPORTED_INDICES = ["sp500", "nasdaq100", "ftse100", "dax40", "custom"]
INDEX_BENCHMARKS = {
    "sp500": "SPY",
    "nasdaq100": "QQQ",
    "ftse100": "ISF.L",     # iShares Core FTSE 100
    "dax40": "EXS1.DE",     # iShares Core DAX
    "custom": "SPY",        # default fallback
}

# ── Screening Dates ───────────────────────────────────
TRAIN_DATES = ["2023-12-31",
               "2024-06-30",
               "2024-12-31"] 

TEST_DATES = [
    "2025-06-30",
]

# ── Experiment Parameters ──────────────────────────────
EXPERIMENT_TRAIN_DATES = TRAIN_DATES
EXPERIMENT_TEST_DATES = TEST_DATES
VERBOSE = False                 # This disables the individual stock progress tracking in favour of TQDM

# ── Labelling (used in feature selection step, not extraction) ─
QUINTILE_TOP = 0.80
QUINTILE_BOTTOM = 0.20

# ── XGBoost / ML Training ──────────────────────────────────
TRAINING_REGRESSION_CSV = DATA_DIR / "ML_Training_Regression.csv"
TESTING_REGRESSION_CSV  = DATA_DIR / "ML_Testing_Regression.csv"
MODELS_DIR              = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

N_OPTUNA_TRIALS   = 100
TRAIN_SPLIT_DATES = ["2023-12-31", "2024-06-30"]
VAL_SPLIT_DATE    = "2024-12-31"
META_COLS = ['Ticker', 'Sector', 'Screening_Date', 'Forward_6m_Return', 
             'Forward_6m_Excess_Return', 'Index_Source', 'Company_Name']
# Multi-index experiments
EXPERIMENT_INDICES = ["sp500", "nasdaq100", "ftse100", "dax40"]  # Default single index
# Example: EXPERIMENT_INDICES = ["sp500", "nasdaq100", "ftse100", "dax40"]
TARGET_COL        = 'Target_Percentile'
# SKIP_EXTRACTION   = True # if this is enabled the models will use the data already provided under the TRAINING_REGRESSION_CSV TESTING_REGRESSION_CSV

# ── Screener (Phase B candidate selection) ─────────────────
PHASE_B_CANDIDATES_CSV = REPORTS_DIR / "phase_b_candidates_new_new.csv" # adjust accordingly as per request
PHASE_B_MIN_SCORE = 10        # Threshold for feature selection (max score is 13)

# ── Phase 1 Extraction
TODAY_STR = dt.today().strftime('%Y-%m-%d')
LIVE_SCREEN_DATE = [TODAY_STR]
# Unless you want a specific date for test then simply specify a date in the yy-mm-dd format. 

# Dynamically name the output files so historical runs are never overwritten
LIVE_FEATURES_CSV = DATA_DIR / f"Live_Screen_Data_{TODAY_STR}.csv"
SCREENER_SCORES_CSV = REPORTS_DIR / f"screener_scores_{TODAY_STR}.csv"

# When using screener for live extraction, remember to input the directory path to your most recently extracted stocks e.g.,
# python screener.py data/Live_Screen_Data_2026-03-26.csv


# Tier 1 features contribute 2 point each (max 8)
TIER1_FEATURES = ["50MA_Gt_200MA", "Op_Margin_Gt_Hist_Avg", "EPS_Current_Change_Above_Industry", "Gross_Margin_Above_Industry"]

# Tier 2 features contribute 1 point each (max 5)
TIER2_FEATURES = ["Assets_To_Liability_Ratio_Above_Industry", "Zero_Dividend", "Operating_Margin_Above_Industry", "FCF_Growing_Sustained", "PEG_01_to_05"]

## OLD FEATURES (BASED ON THE S&P500 only)
# TIER2_FEATURES = [
#     "ROE_Gt_15_Sustained",
#     "EPS_Current_Change_Above_Industry",
#     "Price_Gt_50MA",
#     "DE_Less_1",
#     "PE_Bottom_40_Pct",
#     "PB_Below_Industry",
#     "Assets_To_Liability_Ratio_Above_Industry",
# ]

# # Tier 1 features contribute 2 points each (max 8)
# TIER1_FEATURES = [
#     "50MA_Gt_200MA",
#     "Op_Margin_Gt_Hist_Avg",
#     "PE_Below_Industry",
#     "Gross_Margin_Above_Industry",
# ]


# PHASE B CONFIGURATION
# STAGE 1
STAGE1_DIR = Path("data/stage1_raw")
STAGE1_DIR.mkdir(parents=True, exist_ok=True)
STAGE1_SUMMARY_CSV = DATA_DIR / "stage1_summary.csv"
MAX_RESULTS_PER_SEARCH = 5  # limit the results to the most recent news, you can expand it further but it will increase API costs
DELAY_BETWEEN_SEARCHES = 2  # seconds
SEARCH_LOOKBACK_MONTHS = 6
SEARCH_DATE_REFERENCE  = dt.now()

# STAGE 2
LOCAL_MODEL = True         # Remember to Set to False if you do not have LLM run locally, the script then reverts to the HAIKU Model
STAGE2_DIR = Path("data/stage2_processed")
STAGE2_DIR.mkdir(parents=True, exist_ok=True)
STAGE2_SUMMARY_CSV = DATA_DIR / "stage2_summary.csv"
LOCAL_MODEL_URL = "http://localhost:8000/v1"
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# STAGE 3
STAGE3_DIR = Path("data/stage3_reports")
STAGE3_SUMMARY_CSV = DATA_DIR / "stage3_summary.csv"
VECTORSTORE_DIR = Path("data/vectorstores")
STAGE3_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# STAGE 4
USER_PROFILE_DIR = Path("data/user_profile")
USER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# Phase B batch runner threshold (used by run_phase_b.py)
# Adjust accordingly, in my case 12 was a sweet spot for top picks but 11 also show valid candidates.
PHASE_B_RUNNER_MIN_SCORE = 12

# Enabling this will redo all of the individual stock extractions 
FORCE_REFRESH = False 
# config.py
from pathlib import Path
import os
from pathlib import Path
from dotenv import load_dotenv
import datetime

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
BENCHMARK_TICKER = "SPY"
REPORTING_LAG_DAYS = 0        # Needs to be set up to 0 for live data extraction, otherwise 90
MIN_PRICE_HISTORY = 200        # minimum trading days needed

# ── Ticker Selection ───────────────────────────────────
# Set to None for full S&P 500, or an integer for random sample
TICKER_SAMPLE_SIZE = None       # None = all, 10 = test mode
RANDOM_SEED = 42

# ── Screening Dates ───────────────────────────────────
TRAIN_DATES = ["2023-12-31",
               "2024-06-30",
               "2024-12-31"] 

TEST_DATES = [
    "2025-06-30",
]

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
META_COLS         = ['Ticker', 'Sector', 'Screening_Date', 'Forward_6m_Excess_Return']
TARGET_COL        = 'Target_Percentile'

# ── Screener (Phase B candidate selection) ─────────────────
PHASE_B_CANDIDATES_CSV = REPORTS_DIR / "phase_b_candidates_new.csv" # adjust accordingly as per request
PHASE_B_MIN_SCORE = 11        # Threshold for feature selection 

# Tier 1 features contribute 2 points each (max 8)
TIER1_FEATURES = [
    "50MA_Gt_200MA",
    "Op_Margin_Gt_Hist_Avg",
    "PE_Below_Industry",
    "Gross_Margin_Above_Industry",
]

# ── Phase 1 Extraction
TODAY_STR = datetime.datetime.today().strftime('%Y-%m-%d')
LIVE_SCREEN_DATE = [TODAY_STR]

# Dynamically name the output files so historical runs are never overwritten
LIVE_FEATURES_CSV = DATA_DIR / f"Live_Screen_Data_{TODAY_STR}.csv"
SCREENER_SCORES_CSV = REPORTS_DIR / f"screener_scores_{TODAY_STR}.csv"

# When using screener for live extraction, remember to input the directory path to your most recently extracted stocks e.g.,
# python screener.py data/Live_Screen_Data_2026-03-26.csv

# PHASE_B_CANDIDATES_CSV = REPORTS_DIR / f"phase_b_candidates_{TODAY_STR}.csv"

# Tier 2 features contribute 1 point each (max 7)
TIER2_FEATURES = [
    "ROE_Gt_15_Sustained",
    "EPS_Current_Change_Above_Industry",
    "Price_Gt_50MA",
    "DE_Less_1",
    "PE_Bottom_40_Pct",
    "PB_Below_Industry",
    "Assets_To_Liability_Ratio_Above_Industry",
]

# PHASE B CONFIGURATION
# STAGE 1
STAGE1_DIR = Path("data/stage1_raw")
STAGE1_DIR.mkdir(parents=True, exist_ok=True)
MAX_RESULTS_PER_SEARCH = 5  # limit the results to the most recent news, you can expand it further but it will increase API costs
DELAY_BETWEEN_SEARCHES = 2  # seconds 

# STAGE 2

STAGE2_DIR = Path("data/stage2_processed")
STAGE2_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODEL_URL = "http://localhost:8000/v1"
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# STAGE 3

STAGE3_DIR = Path("data/stage3_reports")
VECTORSTORE_DIR = Path("data/vectorstores")
STAGE3_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# STAGE 4
USER_PROFILE_DIR = Path("data/user_profile")
USER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
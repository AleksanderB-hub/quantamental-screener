# config.py
from pathlib import Path

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
REPORTING_LAG_DAYS = 90
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
import sys
import pandas as pd
import config as cfg

def run_screener(input_csv_path):
    """Loads a feature dataset, applies tiered scoring, and exports candidates."""
    print(f"\n=== Running Screener on {input_csv_path} ===")
    df = pd.read_csv(input_csv_path)
    
    # ── Validate features present ──
    all_features = cfg.TIER1_FEATURES + cfg.TIER2_FEATURES
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required feature columns: {missing}")

    # ── Compute tiered score ──
    tier1_scores = df[cfg.TIER1_FEATURES].fillna(0).mul(2)
    tier2_scores = df[cfg.TIER2_FEATURES].fillna(0).mul(1)
    df["Score"] = (tier1_scores.sum(axis=1) + tier2_scores.sum(axis=1)).astype(int)

    # ── Build scored output ──
    meta_cols = [c for c in ["Ticker", "Sector", "Screening_Date"] if c in df.columns]
    
    scored = (
        df[meta_cols + ["Score"] + cfg.TIER1_FEATURES + cfg.TIER2_FEATURES]
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )
    scored.index += 1

    # ── Save all scored stocks ──
    scored.to_csv(cfg.SCREENER_SCORES_CSV, index_label="Rank")
    print(f"All {len(scored)} stocks ranked and saved to: {cfg.SCREENER_SCORES_CSV}")

    # ── Save Phase B candidates ──
    if cfg.PHASE_B_MIN_SCORE is not None:
        candidates = scored[scored["Score"] >= cfg.PHASE_B_MIN_SCORE].copy()
        candidates.to_csv(cfg.PHASE_B_CANDIDATES_CSV, index_label="Rank")
        print(f"{len(candidates)} candidates (score >= {cfg.PHASE_B_MIN_SCORE}) saved to: {cfg.PHASE_B_CANDIDATES_CSV}")
        
        # Preview top candidates
        print(f"\nTop Candidates:")
        print(candidates.head(15)[meta_cols + ["Score"]].to_string())

# ── Command Line Execution ──
if __name__ == "__main__":
    # If run via terminal, use the provided argument or default to the test set
    target_path = sys.argv[1] if len(sys.argv) > 1 else cfg.TESTING_REGRESSION_CSV
    run_screener(target_path)
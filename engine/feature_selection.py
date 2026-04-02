"""
feature_selection.py
=====================
Runs two additional feature importance methods (Permutation Importance and
Boruta) alongside existing SHAP results, then builds a consensus table.

Sections:
  A - Load data, split by Screening_Date, load best hyperparams from saved model
  B - Permutation Importance (XGBoost retrained on train split)
  C - Boruta (Random Forest, -999 imputation, full train split)
  D - Consensus Table (SHAP + Permutation + Boruta)

Output files:
  reports/permutation_importance.csv
  reports/feature_consensus.csv
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import subprocess, sys

# ── Auto-install boruta if missing ────────────────────────────
try:
    import boruta  # noqa: F401
except ImportError:
    print("boruta not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boruta"])

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from boruta import BorutaPy

import os
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAINING_REGRESSION_CSV,
    MODELS_DIR, REPORTS_DIR,
    TRAIN_SPLIT_DATES, VAL_SPLIT_DATE,
    META_COLS, TARGET_COL, RANDOM_SEED,
)


def main():
    # ─────────────────────────────────────────────────────────────
    # A — Load Data & Hyperparams
    # ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("A — Loading data and best hyperparameters")
    print("=" * 60)

    df = pd.read_csv(TRAINING_REGRESSION_CSV)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    FEATURE_COLS = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    train_mask = df["Screening_Date"].isin(TRAIN_SPLIT_DATES)
    val_mask   = df["Screening_Date"] == VAL_SPLIT_DATE

    df_train = df[train_mask].copy()
    df_val   = df[val_mask].copy()

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    X_val   = df_val[FEATURE_COLS]
    y_val   = df_val[TARGET_COL]

    print(f"Train rows : {len(X_train)}  (dates: {TRAIN_SPLIT_DATES})")
    print(f"Val rows   : {len(X_val)}   (date: {VAL_SPLIT_DATE})")

    # Load best hyperparameters from the saved final model
    model_path = MODELS_DIR / "xgboost_model.json"
    _ref_model = XGBRegressor()
    _ref_model.load_model(str(model_path))
    best_params = _ref_model.get_params()

    # Drop params that vary between sklearn API and booster config, or that
    # are irrelevant for a re-fit from scratch
    _drop = {"n_jobs", "verbosity", "validate_parameters", "base_score",
             "missing", "nthread", "gpu_id", "predictor", "use_label_encoder",
             "early_stopping_rounds", "callbacks", "eval_metric",
             "feature_types", "feature_weights", "enable_categorical",
             "max_cat_to_onehot", "max_cat_threshold", "multi_strategy",
             "device", "sampling_method", "interaction_constraints",
             "monotone_constraints", "scale_pos_weight"}
    best_params = {k: v for k, v in best_params.items() if k not in _drop}
    best_params["random_state"] = RANDOM_SEED
    best_params["tree_method"]  = "hist"
    best_params["verbosity"]    = 0

    print(f"\nBest hyperparameters loaded from {model_path.name}:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")


    # ─────────────────────────────────────────────────────────────
    # B — Permutation Importance
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("B — Permutation Importance (XGBoost on train split, scored on val)")
    print("=" * 60)

    # Retrain with best params on train split only (XGBoost handles NaN natively)
    perm_model = XGBRegressor(**best_params)
    perm_model.fit(X_train, y_train)

    print("Computing permutation importance (n_repeats=20)...")
    perm_result = permutation_importance(
        perm_model,
        X_val,
        y_val,
        n_repeats=20,
        scoring="neg_mean_squared_error",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "Feature":          FEATURE_COLS,
        "Perm_Importance":  perm_result.importances_mean,
        "Perm_Std":         perm_result.importances_std,
    }).sort_values("Perm_Importance", ascending=False).reset_index(drop=True)

    perm_df["Perm_Rank"] = range(1, len(perm_df) + 1)

    perm_path = REPORTS_DIR / "permutation_importance.csv"
    perm_df.to_csv(perm_path, index=False)
    print(f"Permutation importance saved to: {perm_path}")
    print("\nTop 15 by Permutation Importance:")
    print(perm_df.head(15).to_string(index=False))


    # ─────────────────────────────────────────────────────────────
    # C — Boruta
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("C — Boruta (Random Forest, full train split)")
    print("=" * 60)

    # Boruta requires no NaN — impute with -999 (tree-safe sentinel value)
    X_train_imputed = X_train.fillna(-999).values.astype(np.float64)
    y_train_arr     = y_train.values.astype(np.float64)

    rf_estimator = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    boruta_selector = BorutaPy(
        estimator=rf_estimator,
        n_estimators="auto", # type: ignore
        random_state=RANDOM_SEED,
        verbose=1,
    )

    print("Running Boruta (this may take a few minutes)...")
    boruta_selector.fit(X_train_imputed, y_train_arr)

    # Decode support/tentative arrays into labels
    boruta_labels = []
    for confirmed, tentative in zip(boruta_selector.support_, boruta_selector.support_weak_):
        if confirmed:
            boruta_labels.append("Confirmed")
        elif tentative:
            boruta_labels.append("Tentative")
        else:
            boruta_labels.append("Rejected")

    boruta_df = pd.DataFrame({
        "Feature":       FEATURE_COLS,
        "Boruta_Status": boruta_labels,
        "Boruta_Ranking": boruta_selector.ranking_,
    }).sort_values("Boruta_Ranking").reset_index(drop=True)

    print("\nBoruta results:")
    print(boruta_df.to_string(index=False))

    confirmed_count  = boruta_labels.count("Confirmed")
    tentative_count  = boruta_labels.count("Tentative")
    rejected_count   = boruta_labels.count("Rejected")
    print(f"\nSummary — Confirmed: {confirmed_count}  Tentative: {tentative_count}  Rejected: {rejected_count}")


    # ─────────────────────────────────────────────────────────────
    # D — Consensus Table
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("D — Consensus Table")
    print("=" * 60)

    # Load SHAP rankings
    shap_df = pd.read_csv(REPORTS_DIR / "feature_importance.csv")
    shap_df = shap_df.sort_values("SHAP_Mean_Abs", ascending=False).reset_index(drop=True)
    shap_df["SHAP_Rank"] = range(1, len(shap_df) + 1)

    # Build perm rank lookup
    perm_rank_map = dict(zip(perm_df["Feature"], perm_df["Perm_Rank"]))

    # Build Boruta status lookup
    boruta_status_map = dict(zip(boruta_df["Feature"], boruta_df["Boruta_Status"]))

    # Assemble consensus table on the full FEATURE_COLS list
    consensus_rows = []
    for feat in FEATURE_COLS:
        shap_row = shap_df[shap_df["Feature"] == feat]
        shap_rank     = int(shap_row["SHAP_Rank"].values[0])     if len(shap_row) else len(FEATURE_COLS) + 1
        shap_val      = float(shap_row["SHAP_Mean_Abs"].values[0]) if len(shap_row) else 0.0
        perm_rank     = perm_rank_map.get(feat, len(FEATURE_COLS) + 1)
        boruta_status = boruta_status_map.get(feat, "Rejected")

        score = 0
        if shap_rank  <= 15: score += 1
        if perm_rank  <= 15: score += 1
        if boruta_status == "Confirmed": score += 1

        consensus_rows.append({
            "Feature":           feat,
            "SHAP_Rank":         shap_rank,
            "SHAP_Mean_Abs":     shap_val,
            "Perm_Rank":         perm_rank,
            "Boruta_Status":     boruta_status,
            "Consensus_Score":   score,
        })

    consensus_df = (
        pd.DataFrame(consensus_rows)
        .sort_values(["Consensus_Score", "SHAP_Rank"], ascending=[False, True])
        .reset_index(drop=True)
    )

    consensus_path = REPORTS_DIR / "feature_consensus.csv"
    consensus_df.to_csv(consensus_path, index=False)
    print(f"Consensus table saved to: {consensus_path}")

    print("\nFull Consensus Table:")
    print(consensus_df.to_string(index=False))

    # Recommended feature set
    recommended = consensus_df[consensus_df["Consensus_Score"] >= 2]["Feature"].tolist()
    print("\n" + "=" * 60)
    print(f"Recommended feature set (Consensus_Score >= 2)  [{len(recommended)} features]:")
    print("=" * 60)
    for f in recommended:
        row = consensus_df[consensus_df["Feature"] == f].iloc[0]
        print(f"  {f:<45}  score={int(row['Consensus_Score'])}  "
              f"SHAP_rank={int(row['SHAP_Rank'])}  "
              f"Perm_rank={int(row['Perm_Rank'])}  "
              f"Boruta={row['Boruta_Status']}")

    print("\nDone.")


if __name__ == "__main__":
    main()

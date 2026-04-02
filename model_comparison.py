"""
model_comparison.py
====================
Trains Lasso Regression and Random Forest alongside the existing XGBoost results,
evaluates all three on the test set, and produces a comparison report.

Sections:
  A - Load data, median imputation, temporal split
  B - Lasso Regression (LassoCV, feature selection)
  C - Random Forest (Optuna tuning, 50 trials)
  D - Comparison report (table + Lasso vs consensus overlap)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from config import (
    TRAINING_REGRESSION_CSV, TESTING_REGRESSION_CSV,
    MODELS_DIR, REPORTS_DIR,
    TRAIN_SPLIT_DATES, VAL_SPLIT_DATE,
    META_COLS, TARGET_COL, RANDOM_SEED,
)

RF_OPTUNA_TRIALS = 50


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label=""):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    rho  = spearmanr(y_true, y_pred).statistic
    if label:
        print(f"  [{label}]  RMSE={rmse:.4f}  MAE={mae:.4f}  Spearman={rho:.4f}")
    return rmse, mae, rho


def hit_rate(y_true, y_pred, threshold):
    """Fraction of top-threshold% predicted that are actually in real top-threshold%."""
    n = len(y_true)
    k = max(1, int(np.ceil(n * threshold)))
    pred_top_idx = np.argsort(y_pred)[-k:]
    actual_top   = set(np.where(y_true >= np.quantile(y_true, 1 - threshold))[0])
    hits = len(set(pred_top_idx) & actual_top)
    return hits / k


def hit_rates_dict(y_true, y_pred):
    return {t: hit_rate(y_true, y_pred, t) for t in [0.20, 0.30, 0.40]}


def main():
    # ─────────────────────────────────────────────────────────────
    # A — Load Data, Imputation, Split
    # ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("A — Loading data and preparing splits")
    print("=" * 60)

    df_train_full = pd.read_csv(TRAINING_REGRESSION_CSV)
    df_test_full  = pd.read_csv(TESTING_REGRESSION_CSV)

    print(f"Training CSV: {len(df_train_full)} rows, {df_train_full.shape[1]} columns")
    print(f"Testing  CSV: {len(df_test_full)} rows, {df_test_full.shape[1]} columns")
    print(f"Screening dates: {sorted(df_train_full['Screening_Date'].unique())}")

    FEATURE_COLS = [c for c in df_train_full.columns if c not in META_COLS + [TARGET_COL]]
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    train_mask = df_train_full['Screening_Date'].isin(TRAIN_SPLIT_DATES)
    val_mask   = df_train_full['Screening_Date'] == VAL_SPLIT_DATE

    df_train    = df_train_full[train_mask].copy()
    df_val      = df_train_full[val_mask].copy()
    df_trainval = df_train_full.copy()   # train + val combined for final model

    X_train    = df_train[FEATURE_COLS].values
    y_train    = df_train[TARGET_COL].values

    X_val      = df_val[FEATURE_COLS].values
    y_val      = df_val[TARGET_COL].values

    X_trainval = df_trainval[FEATURE_COLS].values
    y_trainval = df_trainval[TARGET_COL].values

    X_test     = df_test_full[FEATURE_COLS].values
    y_test     = df_test_full[TARGET_COL].values

    print(f"\nTrain rows : {len(X_train)}  (dates: {TRAIN_SPLIT_DATES})")
    print(f"Val rows   : {len(X_val)}   (date:  {VAL_SPLIT_DATE})")
    print(f"TrainVal   : {len(X_trainval)} rows total")
    print(f"Test rows  : {len(X_test)}")

    # Fit median imputer on training set only
    train_medians = np.nanmedian(X_train, axis=0)

    def impute(X, medians):
        X_imp = X.copy().astype(float)
        for j in range(X_imp.shape[1]):
            mask = np.isnan(X_imp[:, j])
            X_imp[mask, j] = medians[j]
        return X_imp

    X_train_imp    = impute(X_train, train_medians)
    X_val_imp      = impute(X_val, train_medians)
    X_trainval_imp = impute(X_trainval, train_medians)
    X_test_imp     = impute(X_test, train_medians)

    nan_before = np.isnan(X_train).sum()
    nan_after  = np.isnan(X_train_imp).sum()
    print(f"\nImputation (train set): {nan_before} NaNs → {nan_after} NaNs after median fill")


    # ─────────────────────────────────────────────────────────────
    # B — Lasso Regression
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("B — Lasso Regression (LassoCV, 5-fold CV)")
    print("=" * 60)

    lasso_cv = LassoCV(cv=5, random_state=RANDOM_SEED, max_iter=10_000, n_jobs=-1)
    lasso_cv.fit(X_train_imp, y_train)

    print(f"  Selected alpha: {lasso_cv.alpha_:.6f}")

    # Validation metrics (train-only model)
    val_pred_lasso_cv = lasso_cv.predict(X_val_imp)
    lasso_val_rmse, lasso_val_mae, lasso_val_rho = evaluate(
        y_val, val_pred_lasso_cv, label="Lasso val"
    )
    lasso_val_hr = hit_rates_dict(y_val, val_pred_lasso_cv)

    # Non-zero coefficients — Lasso's built-in feature selection
    coef = lasso_cv.coef_
    nonzero_mask = coef != 0.0
    nonzero_features = [FEATURE_COLS[i] for i in range(len(FEATURE_COLS)) if nonzero_mask[i]]
    nonzero_coefs    = coef[nonzero_mask]

    # Sort by absolute magnitude
    order = np.argsort(np.abs(nonzero_coefs))[::-1]
    nonzero_features_sorted = [nonzero_features[i] for i in order]
    nonzero_coefs_sorted    = nonzero_coefs[order]

    print(f"\n  Features surviving Lasso ({len(nonzero_features)} / {len(FEATURE_COLS)}):")
    for feat, c in zip(nonzero_features_sorted, nonzero_coefs_sorted):
        print(f"    {feat:50s}  coef={c:+.6f}")

    if len(nonzero_features) == 0:
        print("  (No features survived — alpha may be too large)")

    # Retrain final Lasso on train + val combined
    lasso_final = LassoCV(cv=5, random_state=RANDOM_SEED, max_iter=10_000, n_jobs=-1)
    lasso_final.fit(X_trainval_imp, y_trainval)

    # Test set evaluation
    test_pred_lasso = lasso_final.predict(X_test_imp)
    lasso_test_rmse, lasso_test_mae, lasso_test_rho = evaluate(
        y_test, test_pred_lasso, label="Lasso test"
    )
    lasso_test_hr = hit_rates_dict(y_test, test_pred_lasso)

    print(f"\nLasso — Test Quintile Hit Rates:")
    for t in [0.20, 0.30, 0.40]:
        print(f"  Top {t*100:.0f}%: {lasso_test_hr[t]*100:.1f}%  (random: {t*100:.1f}%)")


    # ─────────────────────────────────────────────────────────────
    # C — Random Forest Regression (Optuna tuning)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"C — Random Forest (Optuna, {RF_OPTUNA_TRIALS} trials, Spearman objective)")
    print("=" * 60)

    def rf_objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 500),
            max_depth        = trial.suggest_int("max_depth", 3, 10),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 50),
            max_features     = trial.suggest_float("max_features", 0.3, 1.0),
            random_state     = RANDOM_SEED,
            n_jobs           = -1,
        )
        model = RandomForestRegressor(**params)
        model.fit(X_train_imp, y_train)
        preds = model.predict(X_val_imp)
        return spearmanr(y_val, preds).statistic


    study_rf = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study_rf.optimize(rf_objective, n_trials=RF_OPTUNA_TRIALS, show_progress_bar=True)

    best_rf_params   = study_rf.best_params
    best_rf_spearman = study_rf.best_value

    print(f"\nBest Spearman (val): {best_rf_spearman:.4f}")
    print(f"Best params: {best_rf_params}")

    # Val-set evaluation with best params (train-only model)
    rf_val_model = RandomForestRegressor(
        **best_rf_params, random_state=RANDOM_SEED, n_jobs=-1
    )
    rf_val_model.fit(X_train_imp, y_train)
    val_pred_rf = rf_val_model.predict(X_val_imp)
    rf_val_rmse, rf_val_mae, rf_val_rho = evaluate(y_val, val_pred_rf, label="RF val")
    rf_val_hr = hit_rates_dict(y_val, val_pred_rf)

    # Final RF trained on train + val combined
    rf_final = RandomForestRegressor(
        **best_rf_params, random_state=RANDOM_SEED, n_jobs=-1
    )
    rf_final.fit(X_trainval_imp, y_trainval)

    # Test set evaluation
    test_pred_rf = rf_final.predict(X_test_imp)
    rf_test_rmse, rf_test_mae, rf_test_rho = evaluate(
        y_test, test_pred_rf, label="RF test"
    )
    rf_test_hr = hit_rates_dict(y_test, test_pred_rf)

    print(f"\nRandom Forest — Test Quintile Hit Rates:")
    for t in [0.20, 0.30, 0.40]:
        print(f"  Top {t*100:.0f}%: {rf_test_hr[t]*100:.1f}%  (random: {t*100:.1f}%)")

    # Feature importances (Gini)
    fi_rf = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": rf_final.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    fi_rf_path = REPORTS_DIR / "rf_feature_importance.csv"
    fi_rf.to_csv(fi_rf_path, index=False)
    print(f"\nRF feature importances saved to: {fi_rf_path}")
    print("\nTop 15 features by Gini importance:")
    print(fi_rf.head(15).to_string(index=False))


    # ─────────────────────────────────────────────────────────────
    # D — Comparison Report
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("D — Model Comparison Report")
    print("=" * 60)

    # Load XGBoost model and compute metrics dynamically
    xgb_model_path = MODELS_DIR / "xgboost_model.json"
    print(f"  Loading XGBoost model from: {xgb_model_path}")
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(xgb_model_path))

    # Val predictions (train-only model was used for val; use the saved final model
    # on the val split for a consistent comparison)
    xgb_val_preds   = xgb_model.predict(X_val_imp)
    _, _, xgb_val_rho = evaluate(y_val, xgb_val_preds, label="XGBoost val")
    xgb_val_hr      = hit_rates_dict(y_val, xgb_val_preds)

    # Test predictions
    xgb_test_preds   = xgb_model.predict(X_test_imp)
    _, _, xgb_test_rho = evaluate(y_test, xgb_test_preds, label="XGBoost test")
    xgb_test_hr      = hit_rates_dict(y_test, xgb_test_preds)

    def fmt_rho(v):
        return f"{v:+.3f}" if v is not None else "—"

    def fmt_hr(v):
        return f"{v*100:.1f}%" if v is not None else "—"

    header = (
        f"{'Model':<20} {'Val Spearman':>13} {'Test Spearman':>14} "
        f"{'Top20% Hit':>11} {'Top30% Hit':>11} {'Top40% Hit':>11}"
    )
    sep = "─" * len(header)

    rows = [
        ("Lasso",          lasso_val_rho,  lasso_test_rho,  lasso_test_hr),
        ("Random Forest",  rf_val_rho,     rf_test_rho,     rf_test_hr),
        ("XGBoost",        xgb_val_rho,    xgb_test_rho,    xgb_test_hr),
        ("Random baseline", None,          0.0,             {0.20:0.20, 0.30:0.30, 0.40:0.40}),
    ]

    print("\n" + header)
    print(sep)
    for name, val_rho, test_rho, hrs in rows:
        val_str  = fmt_rho(val_rho) if val_rho is not None else "—"
        test_str = fmt_rho(test_rho)
        print(
            f"{name:<20} {val_str:>13} {test_str:>14} "
            f"{fmt_hr(hrs[0.20]):>11} {fmt_hr(hrs[0.30]):>11} {fmt_hr(hrs[0.40]):>11}"
        )

    # Save comparison to CSV
    comparison_records = []
    for name, val_rho, test_rho, hrs in rows:
        comparison_records.append({
            "Model":          name,
            "Val_Spearman":   round(val_rho, 4) if val_rho is not None else None,
            "Test_Spearman":  round(test_rho, 4),
            "Top20_Hit_Rate": round(hrs[0.20], 4),
            "Top30_Hit_Rate": round(hrs[0.30], 4),
            "Top40_Hit_Rate": round(hrs[0.40], 4),
        })

    comparison_df = pd.DataFrame(comparison_records)
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison table saved to: {comparison_path}")


    # ─────────────────────────────────────────────────────────────
    # D2 — Lasso Feature Selection vs Consensus Overlap
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("D2 — Lasso Feature Selection vs Consensus Overlap")
    print("=" * 60)

    consensus_df  = pd.read_csv(REPORTS_DIR / "feature_consensus.csv")
    consensus_set = set(consensus_df[consensus_df["Consensus_Score"] >= 2]["Feature"].tolist())
    lasso_set     = set(nonzero_features)

    both         = sorted(lasso_set & consensus_set)
    consensus_only = sorted(consensus_set - lasso_set)
    lasso_only   = sorted(lasso_set - consensus_set)

    print(f"\nLasso selected {len(lasso_set)} features out of {len(FEATURE_COLS)}")
    print(f"Consensus features (score ≥ 2): {len(consensus_set)}")

    print(f"\nIn BOTH Lasso and consensus ({len(both)}) — strong agreement:")
    for f in both:
        idx = FEATURE_COLS.index(f)
        print(f"  {f}  (coef={coef[idx]:+.6f})")

    print(f"\nIn consensus but ZEROED by Lasso ({len(consensus_only)}):")
    for f in consensus_only:
        print(f"  {f}")

    print(f"\nSelected by Lasso but NOT in consensus ({len(lasso_only)}):")
    for f in lasso_only:
        idx = FEATURE_COLS.index(f)
        print(f"  {f}  (coef={coef[idx]:+.6f})")

    # Save Lasso coefficients
    in_consensus_map = {f: (f in consensus_set) for f in FEATURE_COLS}
    lasso_coef_df = pd.DataFrame({
        "Feature":        FEATURE_COLS,
        "Coefficient":    coef.tolist(),
        "Abs_Coefficient":[abs(c) for c in coef],
        "In_Consensus":   [in_consensus_map[f] for f in FEATURE_COLS],
    }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)

    lasso_coef_path = REPORTS_DIR / "lasso_coefficients.csv"
    lasso_coef_df.to_csv(lasso_coef_path, index=False)
    print(f"\nLasso coefficients saved to: {lasso_coef_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

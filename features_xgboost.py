"""
features_xgboost.py
====================
Trains an XGBoost regression model on ML_Training_Regression.csv to predict
Target_Percentile (cross-sectional percentile rank of 6-month excess return).

Sections:
  A - Load data and split by Screening_Date
  B - Baseline model (default XGBoost params)
  C - Hyperparameter tuning with Optuna (50 trials, Spearman objective)
  D - Final model trained on train + validation combined
  E - SHAP feature importance analysis
  F - Validation predictions, scatter plot, quintile hit-rate metrics
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from config import (
    TRAINING_REGRESSION_CSV, TESTING_REGRESSION_CSV,
    MODELS_DIR, REPORTS_DIR,
    N_OPTUNA_TRIALS, TRAIN_SPLIT_DATES, VAL_SPLIT_DATE,
    META_COLS, TARGET_COL, RANDOM_SEED,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label=""):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    rho  = spearmanr(y_true, y_pred).statistic # type: ignore
    if label:
        print(f"  [{label}]  RMSE={rmse:.4f}  MAE={mae:.4f}  Spearman={rho:.4f}")
    return rmse, mae, rho


def hit_rate(y_true, y_pred, threshold):
    """Fraction of top-threshold% predicted that are actually in real top-threshold%."""
    n = len(y_true)
    k = max(1, int(np.ceil(n * threshold)))
    pred_top_idx  = np.argsort(y_pred)[-k:]
    actual_top    = set(np.where(y_true >= np.quantile(y_true, 1 - threshold))[0])
    hits = len(set(pred_top_idx) & actual_top)
    return hits / k


def main():
    # ─────────────────────────────────────────────────────────────
    # A — Load & Split
    # ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("A — Loading data and splitting by Screening_Date")
    print("=" * 60)

    df = pd.read_csv(TRAINING_REGRESSION_CSV)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"Screening dates present: {sorted(df['Screening_Date'].unique())}")

    FEATURE_COLS = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    train_mask   = df['Screening_Date'].isin(TRAIN_SPLIT_DATES)
    val_mask     = df['Screening_Date'] == VAL_SPLIT_DATE

    df_train = df[train_mask].copy()
    df_val   = df[val_mask].copy()
    df_all   = df.copy()   # train + val combined for final model

    X_train    = df_train[FEATURE_COLS]
    y_train    = df_train[TARGET_COL]
    meta_train = df_train[META_COLS + ['Screening_Date']].reset_index(drop=True)

    X_val    = df_val[FEATURE_COLS]
    y_val    = df_val[TARGET_COL]
    meta_val = df_val[META_COLS + ['Screening_Date']].reset_index(drop=True)

    X_trainval = df_all[FEATURE_COLS]
    y_trainval = df_all[TARGET_COL]

    print(f"\nTrain rows : {len(X_train)}  (dates: {TRAIN_SPLIT_DATES})")
    print(f"Val rows   : {len(X_val)}   (date: {VAL_SPLIT_DATE})")
    print(f"TrainVal   : {len(X_trainval)} rows total")


    # ─────────────────────────────────────────────────────────────
    # B — Baseline Model
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("B — Baseline model (default XGBoost params)")
    print("=" * 60)

    baseline = XGBRegressor(
        n_estimators=200,
        random_state=RANDOM_SEED,
        tree_method="hist",
        verbosity=0,
    )
    baseline.fit(X_train, y_train)
    evaluate(y_val.values, baseline.predict(X_val), label="Baseline val")


    # ─────────────────────────────────────────────────────────────
    # C — Hyperparameter Tuning with Optuna
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"C — Optuna tuning ({N_OPTUNA_TRIALS} trials, Spearman objective)")
    print("=" * 60)

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
            max_depth         = trial.suggest_int("max_depth", 2, 8),
            learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 5.0),
            random_state      = RANDOM_SEED,
            tree_method       = "hist",
            verbosity         = 0,
            early_stopping_rounds = 50,
        )

        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        rho   = spearmanr(y_val.values, preds).statistic # type: ignore

        # Store best_iteration alongside params for later retrieval
        trial.set_user_attr("best_n_estimators", model.best_iteration + 1)

        return rho


    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    best_params           = study.best_params.copy()
    best_n_estimators     = study.best_trial.user_attrs["best_n_estimators"]
    best_spearman         = study.best_value

    print(f"\nBest Spearman (val): {best_spearman:.4f}")
    print(f"Best n_estimators (from early stopping): {best_n_estimators}")
    print(f"Best params: {best_params}")

    # Rebuild best trial model cleanly with fixed n_estimators (no early stopping)
    # so SHAP and val evaluation are deterministic
    best_params_clean = {k: v for k, v in best_params.items()
                         if k != "n_estimators"}
    best_params_clean["n_estimators"] = best_n_estimators

    val_model = XGBRegressor(
        **best_params_clean,
        random_state=RANDOM_SEED,
        tree_method="hist",
        verbosity=0,
    )
    val_model.fit(X_train, y_train)

    print("\nVal-set metrics (best params, train-only model):")
    evaluate(y_val.values, val_model.predict(X_val), label="Best-params val")


    # ─────────────────────────────────────────────────────────────
    # D — Final Model (trained on train + val combined)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("D — Final model (train + val combined, fixed n_estimators)")
    print("=" * 60)

    final_model = XGBRegressor(
        **best_params_clean,
        random_state=RANDOM_SEED,
        tree_method="hist",
        verbosity=0,
    )
    final_model.fit(X_trainval, y_trainval)

    model_path = MODELS_DIR / "xgboost_model.json"
    final_model.save_model(str(model_path))
    print(f"Final model saved to: {model_path}")
    print(f"  Trained on {len(X_trainval)} rows ({len(df['Screening_Date'].unique())} screening dates)")


    # ─────────────────────────────────────────────────────────────
    # E — SHAP Feature Importance
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("E — SHAP feature importance (on val-set model)")
    print("=" * 60)

    explainer   = shap.TreeExplainer(val_model)
    shap_values = explainer.shap_values(X_val)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    xgb_gain      = val_model.get_booster().get_score(importance_type="gain")

    importance_df = pd.DataFrame({
        "Feature":        FEATURE_COLS,
        "SHAP_Mean_Abs":  mean_abs_shap,
        "XGB_Gain":       [xgb_gain.get(f, 0.0) for f in FEATURE_COLS],
    }).sort_values("SHAP_Mean_Abs", ascending=False).reset_index(drop=True)

    importance_path = REPORTS_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")
    print("\nTop 15 features by mean |SHAP|:")
    print(importance_df.head(15).to_string(index=False))

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_imp = importance_df.sort_values("SHAP_Mean_Abs")
    ax.barh(sorted_imp["Feature"], sorted_imp["SHAP_Mean_Abs"], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance — XGBoost (SHAP, Validation Set)")
    plt.tight_layout()
    shap_chart_path = REPORTS_DIR / "shap_feature_importance.png"
    fig.savefig(str(shap_chart_path), dpi=150)
    plt.close(fig)
    print(f"SHAP chart saved to: {shap_chart_path}")


    # ─────────────────────────────────────────────────────────────
    # F — Validation Set Analysis & Hit-Rate Metrics
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("F — Validation set analysis & quintile hit-rate metrics")
    print("=" * 60)

    val_preds = val_model.predict(X_val)

    # Save predictions alongside meta cols
    val_output = meta_val.copy()
    val_output[TARGET_COL]            = y_val.values
    val_output["Predicted_Percentile"] = val_preds

    val_preds_path = REPORTS_DIR / "validation_predictions.csv"
    val_output.to_csv(val_preds_path, index=False)
    print(f"Validation predictions saved to: {val_preds_path}")

    # Scatter plot: predicted vs actual
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_val.values, val_preds, alpha=0.35, s=18, color="steelblue")
    lims = [0, 1]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Target_Percentile")
    ax.set_ylabel("Predicted Target_Percentile")
    ax.set_title(f"Predicted vs Actual — Validation ({VAL_SPLIT_DATE})")
    ax.legend()
    plt.tight_layout()
    scatter_path = REPORTS_DIR / "val_predicted_vs_actual.png"
    fig.savefig(str(scatter_path), dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved to: {scatter_path}")

    # Quintile hit-rate metrics
    print(f"\n--- Quintile Hit-Rate Metrics (Validation: {VAL_SPLIT_DATE}) ---")
    for threshold in [0.20, 0.30, 0.40]:
        hr = hit_rate(y_val.values, val_preds, threshold)
        pct = threshold * 100
        print(
            f"  Top {pct:.0f}% predicted → {hr*100:.1f}% actually in real top {pct:.0f}%"
            f"  (random baseline: {pct:.1f}%)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

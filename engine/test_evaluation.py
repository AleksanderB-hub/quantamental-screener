"""
test_evaluation.py
==================
Evaluates the saved XGBoost model on the held-out 2025-06-30 test set.

Sections:
  A - Load model and test data
  B - Run predictions
  C - Regression metrics (RMSE, MAE, Spearman)
  D - Quintile hit-rate metrics (top 20%, 30%, 40%)
  E - Save predictions CSV and scatter plot
  F - Top-20 shortlist (Phase B preview)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

import sys, os
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TESTING_REGRESSION_CSV,
    MODELS_DIR, REPORTS_DIR,
    META_COLS, TARGET_COL,
)


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
    pred_top_idx = np.argsort(y_pred)[-k:]
    actual_top   = set(np.where(y_true >= np.quantile(y_true, 1 - threshold))[0])
    hits = len(set(pred_top_idx) & actual_top)
    return hits / k


def main():
    # ─────────────────────────────────────────────────────────────
    # A — Load model and test data
    # ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("A — Loading model and test data")
    print("=" * 60)

    model_path = MODELS_DIR / "xgboost_model.json"
    model = XGBRegressor()
    model.load_model(str(model_path))
    print(f"Loaded model from: {model_path}")

    df_test = pd.read_csv(TESTING_REGRESSION_CSV)
    print(f"Loaded {len(df_test)} rows, {df_test.shape[1]} columns")
    print(f"Screening dates present: {sorted(df_test['Screening_Date'].unique())}")

    FEATURE_COLS = [c for c in df_test.columns if c not in META_COLS + [TARGET_COL]]
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    X_test    = df_test[FEATURE_COLS]
    y_test    = df_test[TARGET_COL]
    meta_test = df_test[META_COLS + ['Screening_Date']].reset_index(drop=True)


    # ─────────────────────────────────────────────────────────────
    # B — Run predictions
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("B — Running predictions on test set")
    print("=" * 60)

    test_preds = model.predict(X_test)
    print(f"Predictions generated for {len(test_preds)} stocks")
    print(f"  Pred range: [{test_preds.min():.4f}, {test_preds.max():.4f}]  "
          f"mean={test_preds.mean():.4f}")


    # ─────────────────────────────────────────────────────────────
    # C — Regression metrics
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("C — Regression metrics")
    print("=" * 60)

    evaluate(y_test.values, test_preds, label="Test set")


    # ─────────────────────────────────────────────────────────────
    # D — Quintile hit-rate metrics
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("D — Quintile hit-rate metrics")
    print("=" * 60)

    test_date = df_test['Screening_Date'].iloc[0]
    print(f"--- Quintile Hit-Rate Metrics (Test: {test_date}) ---")
    for threshold in [0.20, 0.30, 0.40]:
        hr  = hit_rate(y_test.values, test_preds, threshold)
        pct = threshold * 100
        print(
            f"  Top {pct:.0f}% predicted → {hr*100:.1f}% actually in real top {pct:.0f}%"
            f"  (random baseline: {pct:.1f}%)"
        )


    # ─────────────────────────────────────────────────────────────
    # E — Save predictions CSV and scatter plot
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("E — Saving predictions and scatter plot")
    print("=" * 60)

    # Build output dataframe
    test_output = meta_test.copy()
    test_output[TARGET_COL]             = y_test.values
    test_output["Predicted_Percentile"] = test_preds

    preds_path = REPORTS_DIR / "test_predictions.csv"
    test_output.to_csv(preds_path, index=False)
    print(f"Predictions saved to: {preds_path}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test.values, test_preds, alpha=0.35, s=18, color="darkorange")
    lims = [0, 1]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Target_Percentile")
    ax.set_ylabel("Predicted Target_Percentile")
    ax.set_title(f"Predicted vs Actual — Test Set ({test_date})")
    ax.legend()
    plt.tight_layout()
    scatter_path = REPORTS_DIR / "test_predicted_vs_actual.png"
    fig.savefig(str(scatter_path), dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved to: {scatter_path}")


    # ─────────────────────────────────────────────────────────────
    # F — Top-20 shortlist (Phase B preview)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("F — Top-20 stocks by predicted percentile (Phase B preview)")
    print("=" * 60)

    test_output["Actual_Rank"]    = y_test.values
    test_output["Predicted_Rank"] = test_preds

    top20 = (
        test_output
        .sort_values("Predicted_Percentile", ascending=False)
        .head(20)
        [["Ticker", "Sector", "Predicted_Percentile", "Actual_Rank", "Forward_6m_Excess_Return"]]
        .reset_index(drop=True)
    )
    top20.index += 1

    # Pretty-print with aligned columns
    print(f"\n{'Rank':<5} {'Ticker':<8} {'Sector':<30} {'Pred%ile':>8} {'Act%ile':>8} {'Fwd6m Exc Ret':>14}")
    print("-" * 80)
    for rank, row in top20.iterrows():
        print(
            f"{rank:<5} {row['Ticker']:<8} {str(row['Sector']):<30} "
            f"{row['Predicted_Percentile']:>8.3f} {row['Actual_Rank']:>8.3f} "
            f"{row['Forward_6m_Excess_Return']:>14.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

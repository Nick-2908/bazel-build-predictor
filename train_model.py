"""
train_model.py

Replicates the Stanford paper's methodology:
  1. Bucketize categorical features (prefix, type) into integer indices
  2. One-hot encode them
  3. Train Linear Regression WITHOUT feature crossing
  4. Train Linear Regression WITH feature crossing (prefix × type)
  5. Evaluate both on train / val / test using RMSE across 9 "periods"
     (sub-epochs on shuffled train folds, matching the paper's RMSE vs Periods plots)
  6. Also benchmark XGBoost as a stronger baseline
  7. Generate all plots
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

PLOTS_DIR = "/home/claude/bazel-build-predictor/plots"
DATA_DIR  = "/home/claude/bazel-build-predictor/data"

# ── Load data ─────────────────────────────────────────────────────────────────
def load_splits():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    val   = pd.read_csv(f"{DATA_DIR}/val.csv")
    test  = pd.read_csv(f"{DATA_DIR}/test.csv")
    return train, val, test


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, cross: bool = False):
    """
    Return feature matrix X and target y.
    cross=True adds the prefix×type interaction feature.
    """
    X_prefix = df[["most_common_prefix"]]
    X_type   = df[["most_common_type"]]

    ohe_prefix = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                                categories=[sorted(df["most_common_prefix"].unique())])
    ohe_type   = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                                categories=[sorted(df["most_common_type"].unique())])

    return X_prefix, X_type, ohe_prefix, ohe_type


def encode(train, val, test, cross: bool):
    """Fit encoders on train, transform all splits. Return (X_tr, X_val, X_te)."""
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np

    all_prefixes = sorted(train["most_common_prefix"].unique())
    all_types    = sorted(train["most_common_type"].unique())

    ohe_p = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                           categories=[all_prefixes])
    ohe_t = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                           categories=[all_types])

    def transform(df):
        P = ohe_p.transform(df[["most_common_prefix"]])
        T = ohe_t.transform(df[["most_common_type"]])
        if not cross:
            return np.hstack([P, T])
        # Feature cross: outer product of each prefix one-hot with each type one-hot
        # Results in |prefix| × |type| additional binary features
        cross_feat = P[:, :, None] * T[:, None, :]        # (N, |P|, |T|)
        cross_feat = cross_feat.reshape(len(df), -1)       # (N, |P|×|T|)
        return np.hstack([P, T, cross_feat])

    ohe_p.fit(train[["most_common_prefix"]])
    ohe_t.fit(train[["most_common_type"]])

    return (transform(train),
            transform(val),
            transform(test),
            train["cpu_time_ms"].values,
            val["cpu_time_ms"].values,
            test["cpu_time_ms"].values)


# ── Simulate "periods" (matching paper's x-axis) ──────────────────────────────
def train_with_periods(X_tr, y_tr, X_val, y_val, X_te, y_te,
                       n_periods: int = 9,
                       lr: float = 0.01,
                       batch_size: int = 16):
    """
    Train SGD linear regression in mini-batches for n_periods.
    Record RMSE on train and val after each period.
    Returns (train_rmses, val_rmses, test_rmse_final, model)
    """
    from sklearn.linear_model import SGDRegressor

    model = SGDRegressor(
        loss="squared_error",
        learning_rate="constant",
        eta0=lr,
        max_iter=1,
        warm_start=True,
        fit_intercept=True,
        random_state=42
    )

    train_rmses, val_rmses = [], []
    n = len(X_tr)

    for period in range(n_periods):
        # Shuffle each period
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X_tr[idx], y_tr[idx]

        # Mini-batch SGD
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            model.partial_fit(X_shuf[start:end], y_shuf[start:end])

        tr_pred  = model.predict(X_tr)
        val_pred = model.predict(X_val)
        tr_rmse  = np.sqrt(mean_squared_error(y_tr, tr_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_rmses.append(tr_rmse)
        val_rmses.append(val_rmse)

    test_pred = model.predict(X_te)
    test_rmse = np.sqrt(mean_squared_error(y_te, test_pred))
    return train_rmses, val_rmses, test_rmse, model


# ── Plotting ──────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
}

def plot_rmse_vs_periods(train_rmses, val_rmses, title, filename):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        periods = list(range(1, len(train_rmses) + 1))
        ax.plot(periods, train_rmses, label="training",   color="#2196F3", linewidth=2)
        ax.plot(periods, val_rmses,   label="validation", color="#FF9800", linewidth=2)
        ax.set_xlabel("Periods")
        ax.set_ylabel("RMSE (ms)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend()
        ax.set_ylim(0, max(max(train_rmses), max(val_rmses)) * 1.3)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {filename}")


def plot_test_rmse(test_rmses, title, filename):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        periods = list(range(1, 10))
        ax.plot(periods, [test_rmses] * 9, label="testing",
                color="#2196F3", linewidth=2)
        ax.set_xlabel("Periods")
        ax.set_ylabel("RMSE (ms)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend()
        ax.set_ylim(0, test_rmses * 1.5)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {filename}")


def plot_pred_vs_actual(y_true, y_pred, title, filename):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#5C6BC0")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual CPU Time (ms)")
        ax.set_ylabel("Predicted CPU Time (ms)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {filename}")


def plot_comparison(results: dict, filename: str):
    """Bar chart comparing RMSE across all models."""
    with plt.rc_context(STYLE):
        models = list(results.keys())
        test_rmses = [results[m]["test_rmse"] for m in models]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"][:len(models)]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(models, test_rmses, color=colors, edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%.0f ms", padding=5, fontsize=10)
        ax.set_ylabel("Test RMSE (ms)")
        ax.set_title("Model Comparison — Test RMSE", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(test_rmses) * 1.25)
        plt.xticks(rotation=10)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {filename}")


def plot_feature_importance(model, feature_names, filename):
    """Plot top feature importances from XGBoost."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([feature_names[i] for i in idx[::-1]],
                importances[idx[::-1]], color="#5C6BC0")
        ax.set_xlabel("Importance")
        ax.set_title("XGBoost Feature Importances (Top 20)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {filename}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)

    train, val, test = load_splits()
    print(f"Loaded: train={len(train)}, val={len(val)}, test={len(test)}")

    results = {}

    # ── Model 1: Linear Regression WITHOUT feature crossing ───────────────────
    print("\n[1] Linear Regression — No Feature Crossing")
    X_tr, X_val, X_te, y_tr, y_val, y_te = encode(train, val, test, cross=False)
    tr_rmse, val_rmse, te_rmse, m1 = train_with_periods(
        X_tr, y_tr, X_val, y_val, X_te, y_te)

    print(f"  Train RMSE by period: {[round(r,1) for r in tr_rmse]}")
    print(f"  Val   RMSE by period: {[round(r,1) for r in val_rmse]}")
    print(f"  Test  RMSE (final):   {te_rmse:.1f} ms")
    print(f"  Avg Train RMSE: {np.mean(tr_rmse):.1f} ms")

    results["LinReg (no cross)"] = {"test_rmse": te_rmse, "model": m1,
                                     "X_te": X_te, "y_te": y_te}

    plot_rmse_vs_periods(tr_rmse, val_rmse,
        "RMSE vs Periods — Linear Regression\n(No Feature Crossing, Train+Val)",
        "linreg_no_cross_train_val.png")
    plot_test_rmse(te_rmse,
        "RMSE vs Periods — Linear Regression\n(No Feature Crossing, Test)",
        "linreg_no_cross_test.png")

    # ── Model 2: Linear Regression WITH feature crossing ──────────────────────
    print("\n[2] Linear Regression — WITH Feature Crossing (prefix × type)")
    X_tr2, X_val2, X_te2, _, _, _ = encode(train, val, test, cross=True)
    tr_rmse2, val_rmse2, te_rmse2, m2 = train_with_periods(
        X_tr2, y_tr, X_val2, y_val, X_te2, y_te)

    print(f"  Train RMSE by period: {[round(r,1) for r in tr_rmse2]}")
    print(f"  Val   RMSE by period: {[round(r,1) for r in val_rmse2]}")
    print(f"  Test  RMSE (final):   {te_rmse2:.1f} ms")
    print(f"  Avg Train RMSE: {np.mean(tr_rmse2):.1f} ms")

    results["LinReg (with cross)"] = {"test_rmse": te_rmse2, "model": m2,
                                       "X_te": X_te2, "y_te": y_te}

    plot_rmse_vs_periods(tr_rmse2, val_rmse2,
        "RMSE vs Periods — Linear Regression\n(With Feature Crossing, Train+Val)",
        "linreg_cross_train_val.png")
    plot_test_rmse(te_rmse2,
        "RMSE vs Periods — Linear Regression\n(With Feature Crossing, Test)",
        "linreg_cross_test.png")

    # ── Predicted vs Actual plots ─────────────────────────────────────────────
    plot_pred_vs_actual(y_te, m1.predict(X_te),
        "Predicted vs Actual — No Cross", "pred_actual_no_cross.png")
    plot_pred_vs_actual(y_te, m2.predict(X_te2),
        "Predicted vs Actual — With Cross", "pred_actual_cross.png")

    # ── Model 3: Ridge Regression (regularized) ───────────────────────────────
    print("\n[3] Ridge Regression — WITH Feature Crossing")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr2, y_tr)
    ridge_te_rmse = np.sqrt(mean_squared_error(y_te, ridge.predict(X_te2)))
    print(f"  Test RMSE: {ridge_te_rmse:.1f} ms")
    results["Ridge (with cross)"] = {"test_rmse": ridge_te_rmse}

    # ── Model 4: XGBoost ──────────────────────────────────────────────────────
    if HAS_XGB:
        print("\n[4] XGBoost — WITH Feature Crossing")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
        xgb_model.fit(X_tr2, y_tr,
                      eval_set=[(X_val2, y_val)],
                      verbose=False)
        xgb_te_rmse = np.sqrt(mean_squared_error(y_te, xgb_model.predict(X_te2)))
        print(f"  Test RMSE: {xgb_te_rmse:.1f} ms")
        results["XGBoost"] = {"test_rmse": xgb_te_rmse}

        # Get feature names for importance plot
        all_prefixes = sorted(train["most_common_prefix"].unique())
        all_types    = sorted(train["most_common_type"].unique())
        feat_names = all_prefixes + all_types
        feat_names += [f"{p}×{t}" for p in all_prefixes for t in all_types]
        plot_feature_importance(xgb_model, feat_names, "xgb_feature_importance.png")

        plot_pred_vs_actual(y_te, xgb_model.predict(X_te2),
            "Predicted vs Actual — XGBoost", "pred_actual_xgb.png")

    # ── Summary comparison ────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("FINAL RESULTS — Test RMSE")
    print("="*55)
    for name, info in results.items():
        print(f"  {name:28s}: {info['test_rmse']:>8.1f} ms")
    print("="*55)

    plot_comparison(results, "model_comparison.png")

    # ── Save per-period RMSE table (mimics paper's tables) ───────────────────
    period_df = pd.DataFrame({
        "period": [f"period {i+1:02d}" for i in range(9)],
        "rmse_no_cross":   [round(r, 2) for r in tr_rmse],
        "rmse_with_cross": [round(r, 2) for r in tr_rmse2],
    })
    period_df.to_csv(f"{DATA_DIR}/period_rmse.csv", index=False)
    print(f"\nPer-period RMSE table:")
    print(period_df.to_string(index=False))

    print(f"\nAll plots saved to {PLOTS_DIR}/")
    return results


if __name__ == "__main__":
    main()

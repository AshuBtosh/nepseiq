"""
train.py — Walk-forward model training and comparison for NepseIQ.

WHY NOT train_test_split(shuffle=True)  [ADR-0003]
--------------------------------------------------
A shuffled split places rows from 2024 in the training set and rows from 2018 in
the test set. The model then "predicts" the past having already seen the future.
On financial time series this reliably inflates accuracy into the 70-90% range
and the result is meaningless. Every split in this file is chronological:
train fold strictly precedes test fold, always.

TWO ADDITIONAL GUARDS beyond plain chronological splitting:

1. SPLIT ON DATE, NOT ROW INDEX.
   This is panel data: 11 symbols share a calendar. Splitting on row position
   would interleave symbols in time. Fold boundaries are calendar dates applied
   to all symbols at once.

2. EMBARGO GAP.
   target_5d at time t is computed from close at t+5. The final rows of a
   training fold therefore carry labels derived from prices that fall inside the
   test fold. An embargo of `horizon` trading days is dropped from the end of
   each training fold to break that overlap. Without this, target_5d results are
   optimistically biased.

Feature set: features.get_feature_names(matrix) MINUS
features.NON_STATIONARY_FEATURES  [ADR-0008]

Scaling: inside a sklearn Pipeline, so the scaler is fit on training-fold data
only and never sees the test fold  [ADR-0003 §3].
"""

from __future__ import annotations
import features  # Day 3 module — source of truth for feature names
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

import argparse
import glob
import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED_DIR = os.path.join("data", "processed")
REPORTS_DIR = "reports"

TARGETS = {"target_1d": 1, "target_5d": 5}   # name -> horizon in sessions

# Processed-set majority-class baselines (Day 3, real run). Models must beat these.
BASELINES = {"target_1d": 0.5611, "target_5d": 0.5431}

N_FOLDS = 5
# first fold trains on the earliest 50% of the calendar
INITIAL_TRAIN_FRACTION = 0.50
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def find_processed_file() -> str:
    """Locate the Day 3 processed feature matrix without hard-coding a filename."""
    candidates = sorted(
        glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
        + glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
    )
    if not candidates:
        raise FileNotFoundError(f"No processed file found in {PROCESSED_DIR}/")
    if len(candidates) > 1:
        print(f"[warn] multiple processed files found: {candidates}")
        print(f"[warn] using: {candidates[-1]}")
    return candidates[-1]


def load_matrix(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["date"])
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    """get_feature_names() is authoritative; ADR-0008 subtracts the non-stationary set."""
    all_feats = list(features.get_feature_names(df))
    excluded = set(features.NON_STATIONARY_FEATURES)
    kept = [c for c in all_feats if c not in excluded]

    missing = [c for c in kept if c not in df.columns]
    if missing:
        raise KeyError(f"Features declared but absent from matrix: {missing}")

    print(f"[features] total declared      : {len(all_feats)}")
    print(f"[features] excluded (ADR-0008) : {len(all_feats) - len(kept)}")
    print(f"[features] used for training   : {len(kept)}")
    print(
        f"[features] excluded names      : {sorted(excluded & set(all_feats))}")
    return kept


# --------------------------------------------------------------------------
# Walk-forward splitter
# --------------------------------------------------------------------------

@dataclass
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp     # inclusive, AFTER embargo removal
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


def walk_forward_folds(dates: pd.Series, horizon: int, n_folds: int = N_FOLDS,
                       initial_fraction: float = INITIAL_TRAIN_FRACTION):
    """
    Expanding-window walk-forward split over the unique trading calendar.

    Fold k:   train = [calendar_start, cut_k - embargo]
              test  = (cut_k, cut_{k+1}]

    The training window EXPANDS each fold (it always starts at the beginning of
    history); the test window rolls forward. Embargo = horizon sessions, removed
    from the tail of every training window.
    """
    calendar = np.sort(dates.unique())
    n_dates = len(calendar)

    start_idx = int(n_dates * initial_fraction)
    remaining = n_dates - start_idx
    step = remaining // n_folds

    if step <= horizon:
        raise ValueError(
            f"Fold width ({step} sessions) is too small for horizon {horizon}. "
            f"Reduce n_folds or initial_fraction."
        )

    for k in range(n_folds):
        cut_idx = start_idx + k * step
        test_end_idx = n_dates - 1 if k == n_folds - \
            1 else start_idx + (k + 1) * step - 1

        train_end_idx = cut_idx - horizon          # EMBARGO applied here
        if train_end_idx <= 0:
            continue

        yield Fold(
            index=k + 1,
            train_start=pd.Timestamp(calendar[0]),
            train_end=pd.Timestamp(calendar[train_end_idx]),
            test_start=pd.Timestamp(calendar[cut_idx]),
            test_end=pd.Timestamp(calendar[test_end_idx]),
            n_train=0, n_test=0,
        )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def build_models(seed: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """
    Scaler lives inside each Pipeline -> fit on training fold only.
    Tree models don't require scaling, but keeping the pipeline shape uniform
    means every model sees an identically-prepared matrix.
    """
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=50,
                n_jobs=-1, random_state=seed)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", tree_method="hist",
                n_jobs=-1, random_state=seed)),
        ]),
    }


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def run_target(df: pd.DataFrame, feat_cols: list[str], target: str, horizon: int):
    print("\n" + "=" * 78)
    print(
        f"TARGET: {target}   (horizon = {horizon} sessions, embargo = {horizon})")
    print("=" * 78)

    data = df.dropna(subset=[target] + feat_cols).copy()
    data[target] = data[target].astype(int)
    print(f"[data] rows after dropping NaN target/features: {len(data)}")
    print(
        f"[data] date range: {data['date'].min().date()} -> {data['date'].max().date()}")
    print(f"[data] class balance: {data[target].mean():.4f} up")

    models = build_models()
    fold_rows = []
    oof = {name: {"y": [], "p": []} for name in models}

    for fold in walk_forward_folds(data["date"], horizon=horizon):
        tr = data[(data["date"] >= fold.train_start)
                  & (data["date"] <= fold.train_end)]
        te = data[(data["date"] >= fold.test_start)
                  & (data["date"] <= fold.test_end)]
        if len(tr) < 500 or len(te) < 100:
            print(f"[fold {fold.index}] SKIPPED — too few rows "
                  f"(train={len(tr)}, test={len(te)})")
            continue

        X_tr, y_tr = tr[feat_cols].values, tr[target].values
        X_te, y_te = te[feat_cols].values, te[target].values

        fold_majority = max(y_te.mean(), 1 - y_te.mean())

        print(f"\n[fold {fold.index}] "
              f"train {fold.train_start.date()}..{fold.train_end.date()} (n={len(tr)}) | "
              f"test {fold.test_start.date()}..{fold.test_end.date()} (n={len(te)}) | "
              f"test-fold majority={fold_majority:.4f}")

        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)

            oof[name]["y"].append(y_te)
            oof[name]["p"].append(proba)

            row = {
                "target": target,
                "model": name,
                "fold": fold.index,
                "n_train": len(tr),
                "n_test": len(te),
                "test_start": fold.test_start.date(),
                "test_end": fold.test_end.date(),
                "fold_majority": fold_majority,
                "accuracy": accuracy_score(y_te, pred),
                "precision": precision_score(y_te, pred, zero_division=0),
                "recall": recall_score(y_te, pred, zero_division=0),
                "f1": f1_score(y_te, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else np.nan,
            }
            fold_rows.append(row)
            print(f"    {name:<20} acc={row['accuracy']:.4f}  "
                  f"prec={row['precision']:.4f}  rec={row['recall']:.4f}  "
                  f"f1={row['f1']:.4f}  auc={row['roc_auc']:.4f}")

    folds_df = pd.DataFrame(fold_rows)
    if folds_df.empty:
        print("[error] no folds ran.")
        return folds_df, pd.DataFrame()

    summary = (folds_df.groupby("model")
               [["accuracy", "precision", "recall", "f1", "roc_auc"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary["baseline"] = BASELINES[target]
    summary["acc_minus_baseline"] = (
        summary["accuracy_mean"] - BASELINES[target]).round(4)
    summary["target"] = target

    print(f"\n--- {target} SUMMARY (mean ± std across folds) ---")
    print(f"Processed-set majority-class baseline: {BASELINES[target]:.4f}")
    print(summary.to_string(index=False))

    # Leakage gate — PLAN.md Day 4 DoD
    hot = summary[summary["accuracy_mean"] > 0.70]
    if not hot.empty:
        print("\n" + "!" * 78)
        print("LEAKAGE GATE TRIPPED — mean accuracy above 70%:")
        print(hot[["model", "accuracy_mean"]].to_string(index=False))
        print("Do not proceed. Investigate before recording these numbers.")
        print("!" * 78)

    _export_plots(oof, models, feat_cols, data, target, horizon)
    return folds_df, summary


# --------------------------------------------------------------------------
# Plot exports
# --------------------------------------------------------------------------

def _export_plots(oof, models, feat_cols, data, target, horizon):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Confusion matrices + ROC curves from pooled out-of-fold predictions
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    for name in models:
        y = np.concatenate(oof[name]["y"])
        p = np.concatenate(oof[name]["p"])
        pred = (p >= 0.5).astype(int)

        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
        ax.set_xticks([0, 1], ["pred down", "pred up"])
        ax.set_yticks([0, 1], ["true down", "true up"])
        ax.set_title(f"{name} — {target}\n(pooled out-of-fold)")
        fig.tight_layout()
        fig.savefig(f"{REPORTS_DIR}/day4_cm_{target}_{name}.png", dpi=130)
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="random (0.500)")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(f"ROC — {target} (pooled out-of-fold)")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(f"{REPORTS_DIR}/day4_roc_{target}.png", dpi=130)
    plt.close(fig_roc)

    # Feature importance — refit each model once on the full history for the plot only.
    # These fits are NOT used for any reported metric.
    for name, pipe in models.items():
        X = data[feat_cols].values
        y = data[target].values
        pipe.fit(X, y)
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            label = "importance"
        else:
            imp = np.abs(clf.coef_[0])
            label = "|coefficient|"

        order = np.argsort(imp)[::-1][:20]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh([feat_cols[i] for i in order][::-1], imp[order][::-1])
        ax.set_xlabel(label)
        ax.set_title(f"{name} — top 20 features — {target}")
        fig.tight_layout()
        fig.savefig(
            f"{REPORTS_DIR}/day4_importance_{target}_{name}.png", dpi=130)
        plt.close(fig)

    print(f"[export] figures written to {REPORTS_DIR}/")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="Load data and print shape/columns, then exit without training.")
    args = ap.parse_args()

    path = find_processed_file()
    print(f"[load] {path}")
    df = load_matrix(path)
    print(f"[load] shape={df.shape}")
    print(f"[load] symbols={sorted(df['symbol'].unique())}")

    feat_cols = resolve_feature_columns(df)

    if args.inspect:
        print("\n[inspect] stopping before training as requested.")
        return

    all_folds, all_summaries = [], []
    for target, horizon in TARGETS.items():
        folds_df, summary = run_target(df, feat_cols, target, horizon)
        all_folds.append(folds_df)
        all_summaries.append(summary)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    pd.concat(all_folds).to_csv(
        f"{REPORTS_DIR}/day4_fold_metrics.csv", index=False)
    pd.concat(all_summaries).to_csv(
        f"{REPORTS_DIR}/day4_model_comparison.csv", index=False)
    with open(f"{REPORTS_DIR}/day4_features_used.json", "w") as f:
        json.dump({"n_features": len(feat_cols), "features": feat_cols,
                   "excluded_adr_0008": sorted(features.NON_STATIONARY_FEATURES)}, f, indent=2)

    print("\n[done] metrics -> reports/day4_fold_metrics.csv, "
          "reports/day4_model_comparison.csv")


if __name__ == "__main__":
    main()

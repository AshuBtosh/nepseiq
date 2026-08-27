"""
freeze_model.py — Day 5 model freeze for NepseIQ.

SCOPE (binding, per ADR-0009)
------------------------------
Only target_1d / RandomForest is frozen and served. target_5d did not beat its
baseline in Day 4 walk-forward validation (LogReg 53.72%, RF 54.08%, XGB
53.33% against a 54.31% baseline) and is not deployed -- it is reported as a
negative finding in model_card.md, not pickled here.

WHY A FULL-HISTORY REFIT, AND WHAT IT DOES NOT MEAN
-----------------------------------------------------
PLAN.md's Day 5 instruction is explicit: "select final model, retrain on full
history, freeze." This script does exactly that -- it is a PRODUCTION WEIGHTS
step, not an evaluation step.

This refit has no held-out fold by construction: it trains on every row the
processed dataset contains, because that is what "serve this in production"
means -- the deployed model should see all available history, not withhold
part of it forever for a metric that already exists.

CRITICAL: this refit produces NO new accuracy/AUC number, and none is
computed or reported here. Reporting an in-sample metric from this fit would
be the exact leakage/overfitting failure mode CLAUDE.md SS9 exists to prevent
-- a number computed on data the model was trained on tells you nothing about
generalization. The only honest performance numbers for this model are the
Day 4 walk-forward fold metrics (reports/day4_fold_metrics.csv,
reports/day4_model_comparison.csv) and the Day 5 backtest
(reports/day5_threshold_sensitivity.csv). metadata.json points to those files
by name rather than embedding a number computed here.

HYPERPARAMETERS
----------------
Identical to train.py's build_models()["RandomForest"] -- same n_estimators,
max_depth, min_samples_leaf, random_state. No retuning. Freeze means freeze;
this is the same model Day 4 evaluated, refit on more rows, not a new model.

ARTIFACTS PRODUCED (ml/models/)
---------------------------------
  model.pkl     -- fitted RandomForestClassifier (already scaled internally
                   via the same Pipeline used in training -- see note below)
  scaler.pkl    -- the StandardScaler fit on full history, saved separately
                   in case the serving layer needs to transform raw inputs
                   before calling model.predict without going through the
                   full sklearn Pipeline object
  features.json -- ordered list of the 40 stationary features (ADR-0008),
                   in the exact column order the model expects
  metadata.json -- what this is, what it isn't, where the real numbers live
"""

from __future__ import annotations

import glob
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import features  # Day 3 module — source of truth for feature names

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")
REPORTS_DIR = "reports"

FROZEN_TARGET = "target_1d"
FROZEN_MODEL_NAME = "RandomForest"
RANDOM_STATE = 42

# Must match train.py's build_models() exactly. Freeze means freeze -- this
# is not a place to retune.
RF_PARAMS = dict(
    n_estimators=300, max_depth=8, min_samples_leaf=50,
    n_jobs=-1, random_state=RANDOM_STATE,
)


def find_processed_file() -> str:
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
    all_feats = list(features.get_feature_names(df))
    excluded = set(features.NON_STATIONARY_FEATURES)
    kept = [c for c in all_feats if c not in excluded]
    missing = [c for c in kept if c not in df.columns]
    if missing:
        raise KeyError(f"Features declared but absent from matrix: {missing}")
    return kept


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"[scope] freezing target={FROZEN_TARGET} model={FROZEN_MODEL_NAME} "
          f"(per ADR-0009)")

    path = find_processed_file()
    print(f"[load] {path}")
    df = load_matrix(path)

    feat_cols = resolve_feature_columns(df)
    print(f"[features] {len(feat_cols)} stationary features (ADR-0008)")

    data = df.dropna(subset=[FROZEN_TARGET] + feat_cols).copy()
    data[FROZEN_TARGET] = data[FROZEN_TARGET].astype(int)
    print(f"[data] full-history training rows: {len(data)}")
    print(f"[data] date range: {data['date'].min().date()} -> "
          f"{data['date'].max().date()}")

    X = data[feat_cols].values
    y = data[FROZEN_TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_scaled, y)
    print(f"[fit] RandomForest fit on {len(data)} rows, "
          f"{len(feat_cols)} features. No held-out fold -- this is a "
          f"production refit, not an evaluation. See metadata.json.")

    # Save the pipeline (scaler + classifier together) as model.pkl, since
    # that's what a real serving layer wants to call directly.
    pipeline = Pipeline([("scaler", scaler), ("clf", clf)])
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[save] {model_path}")

    # Also save the scaler alone, in case the serving layer wants to
    # transform inputs without instantiating the full Pipeline object.
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[save] {scaler_path}")

    features_path = os.path.join(MODELS_DIR, "features.json")
    with open(features_path, "w") as f:
        json.dump({
            "target": FROZEN_TARGET,
            "n_features": len(feat_cols),
            "features_ordered": feat_cols,
            "excluded_non_stationary_adr_0008": sorted(
                features.NON_STATIONARY_FEATURES
            ),
        }, f, indent=2)
    print(f"[save] {features_path}")

    # Real performance numbers are NOT computed here -- they're pulled from
    # the Day 4 walk-forward comparison table that already exists on disk,
    # so metadata.json can't drift from what was actually measured.
    day4_comparison_path = os.path.join(
        REPORTS_DIR, "day4_model_comparison.csv")
    rf_1d_metrics = None
    if os.path.exists(day4_comparison_path):
        comp = pd.read_csv(day4_comparison_path)
        row = comp[(comp["target"] == FROZEN_TARGET)
                   & (comp["model"] == FROZEN_MODEL_NAME)]
        if not row.empty:
            rf_1d_metrics = row.iloc[0].to_dict()
    if rf_1d_metrics is None:
        print(f"[error] Could not find {FROZEN_MODEL_NAME}/{FROZEN_TARGET} "
              f"row in {day4_comparison_path}. metadata.json will be written "
              f"WITHOUT walk-forward metrics -- fix this before treating the "
              f"freeze as complete.")

    metadata = {
        "model_name": FROZEN_MODEL_NAME,
        "target": FROZEN_TARGET,
        "horizon_sessions": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_pipeline": ["StandardScaler", "RandomForestClassifier"],
        "hyperparameters": RF_PARAMS,
        "n_features": len(feat_cols),
        "training_rows_full_history": int(len(data)),
        "training_date_range": {
            "start": str(data["date"].min().date()),
            "end": str(data["date"].max().date()),
        },
        "IMPORTANT_full_history_refit_has_no_holdout": (
            "This artifact was refit on ALL available rows per PLAN.md Day 5 "
            "('retrain on full history, freeze'). It has no held-out test "
            "fold and no accuracy/AUC number is computed from this fit. The "
            "only honest performance numbers for this model are the Day 4 "
            "walk-forward metrics below, measured on a DIFFERENT (earlier, "
            "less-fitted) version of this same model configuration."
        ),
        "day4_walk_forward_metrics_source": day4_comparison_path,
        "day4_walk_forward_metrics": rf_1d_metrics,
        "day4_walk_forward_baseline": 0.5611,
        "day5_backtest_source": os.path.join(
            REPORTS_DIR, "day5_threshold_sensitivity.csv"
        ),
        "excluded_data_anomalies": [
            {"symbol": "NLIC", "date": "2018-03-27",
             "reason": ("-30.6% close-to-close on consecutive sessions; "
                        "exceeds any real NEPSE circuit breaker; treated as "
                        "an unadjusted corporate action per the accepted "
                        "PROGRESS.md Parking Lot limitation. Excluded from "
                        "the Day 5 backtest only -- NOT excluded from this "
                        "training fit, since target_1d/features.py were "
                        "already frozen upstream of Day 5 and this script "
                        "does not reopen Day 3/4 decisions.")},
        ],
        "not_served": {
            "target_5d": ("No model beat baseline in Day 4 walk-forward "
                          "validation (best: RandomForest 54.08% vs 54.31% "
                          "baseline). Reported as a negative finding per "
                          "ADR-0009, not frozen or served."),
        },
    }

    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[save] {metadata_path}")

    print("\n[done] Frozen artifacts in ml/models/: "
          "model.pkl, scaler.pkl, features.json, metadata.json")
    print("[reminder] ML work is now closed per PLAN.md Day 5 DoD. "
          "No retraining, no feature changes, in this session or any future "
          "one, without a superseding ADR.")


if __name__ == "__main__":
    main()

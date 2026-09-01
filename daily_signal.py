"""
daily_signal.py — Morning runner for the expiry-pyramid ML signal.

    python daily_signal.py              # today
    python daily_signal.py --date 2025-12-31
    python daily_signal.py --train      # (re)train + save model

Flow: capture today's 09:55 snapshot, load the previous trading day's 15:20
snapshot, build the same 34-feature row used in training, and score it with
the pre-trained LogisticRegression pipeline (models/logreg_model.pkl —
trained once on 2023-2024, loaded every morning, no daily retraining).

Trade card: BULLISH → Bull Put Spread, BEARISH → Bear Call Spread, entry
10:00, target +50 pts, stop −30 pts, hard square-off 15:00.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from comparator import SNAP_A_TIME, SNAP_B_TIME
from ml_direction import (
    TIER_PREFIX,
    _feature_cols,
    _get_snapshot,
    _pct,
    _tier_features,
    build_dataset,
)

MODEL_PATH = Path(__file__).parent / "models" / "logreg_model.pkl"
TRAIN_CUTOFF = "2025-01-01"  # per spec: pre-trained on 2023-2024

ENTRY_TIME = "10:00 AM"
SQUARE_OFF = "3:00 PM"
TARGET_PTS = 50
STOP_PTS = 30

PLAYS = {"BULLISH": "Bull Put Spread (ATM, 50 pts wide)",
         "BEARISH": "Bear Call Spread (ATM, 50 pts wide)"}


# ─── Training ────────────────────────────────────────────────────────────────

def train_and_save() -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = build_dataset()
    train = df[df["date"] < TRAIN_CUTOFF]
    cols = _feature_cols(df)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    )
    model.fit(train[cols].fillna(0.0), train["label"])

    bundle = {
        "model": model,
        "feature_cols": cols,
        "trained_through": train["date"].max(),
        "n_train_rows": len(train),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model trained on {len(train)} rows (2023-01 .. {bundle['trained_through']})")
    print(f"Saved: {MODEL_PATH}")
    return bundle


def load_model() -> dict:
    if not MODEL_PATH.exists():
        print("No saved model found — training now (one-time)...")
        return train_and_save()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ─── Feature row for one morning ─────────────────────────────────────────────

def _prev_snapshot(today: date, index: str):
    """Walk back up to 7 calendar days for the last 15:20 snapshot."""
    for back in range(1, 8):
        d = today - timedelta(days=back)
        snap = _get_snapshot(d, SNAP_A_TIME, index)
        if snap is not None:
            return d, snap
    return None, None


def build_feature_row(today: date, index: str = "NIFTY"):
    snap_b = _get_snapshot(today, SNAP_B_TIME, index)
    if snap_b is None:
        return None, f"No 09:55 cache data for {today} — is the options cache updated?"

    prev_d, snap_a = _prev_snapshot(today, index)
    if snap_a is None:
        return None, f"No 15:20 snapshot found in the 7 days before {today}"

    row = {"overnight_move_pct": _pct(snap_b.get("spot"), snap_a.get("spot"))}
    for tier, prefix in TIER_PREFIX.items():
        row.update(_tier_features(
            prefix, snap_a["expiries"].get(tier), snap_b["expiries"].get(tier)
        ))
    context = {
        "prev_day": prev_d.strftime("%Y-%m-%d"),
        "spot_prev_close": snap_a.get("spot"),
        "spot_morning": snap_b.get("spot"),
    }
    return (row, context), None


# ─── Runner ──────────────────────────────────────────────────────────────────

def run(today: date, index: str = "NIFTY") -> int:
    bundle = load_model()
    built, err = build_feature_row(today, index)
    if err:
        print(f"[ERR] {err}")
        return 2
    row, ctx = built

    X = pd.DataFrame([row]).reindex(columns=bundle["feature_cols"]).fillna(0.0)
    proba_up = float(bundle["model"].predict_proba(X)[0, 1])
    direction = "BULLISH" if proba_up >= 0.5 else "BEARISH"
    confidence = proba_up if direction == "BULLISH" else 1 - proba_up

    w = 58
    print(f"\n{'=' * w}")
    print(f"  EXPIRY PYRAMID DAILY SIGNAL — {today} ({index})")
    print(f"{'=' * w}")
    print(f"  Prev session      : {ctx['prev_day']}  "
          f"(close spot {ctx['spot_prev_close']})")
    print(f"  Morning spot 09:55: {ctx['spot_morning']}   "
          f"overnight {row['overnight_move_pct']:+.2f}%")
    print(f"  Model             : LogisticRegression "
          f"(trained through {bundle['trained_through']}, "
          f"{bundle['n_train_rows']} rows)")
    print(f"{'-' * w}")
    print(f"  DIRECTION         : {direction}   (P = {confidence:.2f})")
    print(f"  RECOMMENDED PLAY  : {PLAYS[direction]}")
    print(f"  ENTRY             : {ENTRY_TIME}")
    print(f"  TARGET            : +{TARGET_PTS} pts from entry")
    print(f"  STOP              : -{STOP_PTS} pts from entry")
    print(f"  SQUARE OFF        : {SQUARE_OFF} hard exit, no exceptions")
    print(f"{'=' * w}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="expiry-pyramid daily signal")
    ap.add_argument("--date", help="run for a specific date (YYYY-MM-DD), default today")
    ap.add_argument("--train", action="store_true", help="(re)train and save the model")
    ap.add_argument("--index", default="NIFTY")
    args = ap.parse_args()

    if args.train:
        train_and_save()
        return 0
    today = date.fromisoformat(args.date) if args.date else date.today()
    return run(today, args.index)


if __name__ == "__main__":
    sys.exit(main())

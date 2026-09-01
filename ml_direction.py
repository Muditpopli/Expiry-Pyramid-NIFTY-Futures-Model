"""
ml_direction.py — ML classifier for NIFTY 10:00→15:00 intraday direction.

Dataset: for every consecutive trading-day pair 2023-2025, the same overnight
shift the comparator reads (15:20 previous day vs 09:55 today) is expanded
into per-expiry features:

    per tier (current/next/far):
        pcr_yday, pcr_today, pcr_change,
        call_oi_chg_pct, put_oi_chg_pct, oi_shift,
        iv_skew_yday, iv_skew_today, iv_skew_change,
        delta_change, vega_change                      (11 features × 3 tiers)
    common: overnight_move_pct                          (= 34 features)

Label: 1 when NIFTY spot at 15:00 > spot at 10:00, 0 when lower, zero-move
rows dropped.

Validation: expanding-window walk-forward, quarterly 2025 folds — each fold
trains on everything before the quarter (starting from 2023) and predicts the
quarter, so all 2025 predictions are out-of-sample with no leakage.

Models tried: XGBoost, RandomForest, LogisticRegression, LightGBM (if
installed).  Best OOS accuracy wins; it is then backtested two ways:

    full session — enter 10:00, hard exit 15:00, no target/stop
                   → data/reports/ml_direction_backtest.csv
    50/30 exits  — +50 target / −30 stop on the spot path, 15:00 square-off
                   → data/reports/ml_direction_50_30_backtest.csv
"""

from __future__ import annotations

import sys
from bisect import bisect_right
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cache_reader import IST, spot_series, trading_dates
from comparator import SNAP_A_TIME, SNAP_B_TIME
from snapshot import capture_snapshot, load_or_capture

ML_DIR = Path(__file__).parent / "data" / "ml"
REPORT_DIR = Path(__file__).parent / "data" / "reports"
FEATURES_CSV = ML_DIR / "direction_features.csv"
FULL_CSV = REPORT_DIR / "ml_direction_backtest.csv"
EXITS_CSV = REPORT_DIR / "ml_direction_50_30_backtest.csv"

YEARS = (2023, 2024, 2025)
TEST_YEAR = 2025
TIER_PREFIX = {"current": "cur", "next": "nxt", "far": "far"}

ENTRY_TIME = (10, 0)
EXIT_TIME = (15, 0)
TARGET_PTS = 50.0
STOP_PTS = 30.0


# ─── Dataset construction ────────────────────────────────────────────────────

def _get_snapshot(d: date, hhmm: str, index: str) -> Optional[dict]:
    """Load a snapshot, recapturing legacy files that predate net_vega."""
    snap = load_or_capture(d, hhmm, index)
    if snap is None:
        return None
    for tier in snap["expiries"].values():
        if "net_vega" not in tier:
            return capture_snapshot(d, hhmm, index)
    return snap


def _pct(new: Optional[float], old: Optional[float]) -> float:
    if new is None or old is None or old == 0:
        return np.nan
    return (new - old) / old * 100


def _tier_features(prefix: str, a: Optional[dict], b: Optional[dict]) -> Dict[str, float]:
    if a is None or b is None:
        keys = ("pcr_yday pcr_today pcr_change call_oi_chg_pct put_oi_chg_pct "
                "oi_shift iv_skew_yday iv_skew_today iv_skew_change "
                "delta_change vega_change").split()
        return {f"{prefix}_{k}": np.nan for k in keys}

    call_chg = _pct(b["total_call_oi"], a["total_call_oi"])
    put_chg = _pct(b["total_put_oi"], a["total_put_oi"])
    return {
        f"{prefix}_pcr_yday": a["pcr"] if a["pcr"] is not None else np.nan,
        f"{prefix}_pcr_today": b["pcr"] if b["pcr"] is not None else np.nan,
        f"{prefix}_pcr_change": (
            b["pcr"] - a["pcr"]
            if a["pcr"] is not None and b["pcr"] is not None else np.nan
        ),
        f"{prefix}_call_oi_chg_pct": call_chg,
        f"{prefix}_put_oi_chg_pct": put_chg,
        f"{prefix}_oi_shift": put_chg - call_chg,
        f"{prefix}_iv_skew_yday": a["iv_skew"] if a["iv_skew"] is not None else np.nan,
        f"{prefix}_iv_skew_today": b["iv_skew"] if b["iv_skew"] is not None else np.nan,
        f"{prefix}_iv_skew_change": (
            b["iv_skew"] - a["iv_skew"]
            if a["iv_skew"] is not None and b["iv_skew"] is not None else np.nan
        ),
        f"{prefix}_delta_change": b["net_delta"] - a["net_delta"],
        f"{prefix}_vega_change": b.get("net_vega", np.nan) - a.get("net_vega", np.nan),
    }


def _spot_at(index: str, d: date, hh: int, mm: int) -> Optional[float]:
    ts, spot = spot_series(index, d)
    if not ts:
        return None
    target = int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=IST).timestamp())
    i = bisect_right(ts, target) - 1
    return spot[i] if i >= 0 else None


def build_dataset(index: str = "NIFTY", rebuild: bool = False) -> pd.DataFrame:
    if FEATURES_CSV.exists() and not rebuild:
        df = pd.read_csv(FEATURES_CSV)
        print(f"Loaded cached dataset: {len(df)} rows ({FEATURES_CSV})")
        return df

    dates: List[date] = []
    for year in YEARS:
        dates.extend(trading_dates(index, year))
    dates = sorted(set(dates))
    print(f"Building dataset over {len(dates)} trading dates "
          f"({dates[0]} to {dates[-1]}) — captures snapshots on first run")

    rows = []
    for n, (prev_d, d) in enumerate(zip(dates, dates[1:]), 1):
        snap_a = _get_snapshot(prev_d, SNAP_A_TIME, index)
        snap_b = _get_snapshot(d, SNAP_B_TIME, index)
        if snap_a is None or snap_b is None:
            continue

        spot_1000 = _spot_at(index, d, *ENTRY_TIME)
        spot_1500 = _spot_at(index, d, *EXIT_TIME)
        if spot_1000 is None or spot_1500 is None:
            continue
        move = spot_1500 - spot_1000
        if move == 0:
            continue

        row: Dict[str, object] = {"date": d.strftime("%Y-%m-%d")}
        row["overnight_move_pct"] = _pct(snap_b.get("spot"), snap_a.get("spot"))
        for tier, prefix in TIER_PREFIX.items():
            row.update(_tier_features(
                prefix, snap_a["expiries"].get(tier), snap_b["expiries"].get(tier)
            ))
        row.update({
            "spot_1000": spot_1000,
            "spot_1500": spot_1500,
            "move_pts": round(move, 2),
            "label": int(move > 0),
        })
        rows.append(row)
        if n % 50 == 0:
            print(f"  ... {n}/{len(dates) - 1} pairs done (latest {d})")

    df = pd.DataFrame(rows)
    ML_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"Dataset built: {len(df)} rows, "
          f"{df['label'].mean() * 100:.1f}% bullish days ({FEATURES_CSV})")
    return df


# ─── Models & walk-forward ───────────────────────────────────────────────────

def _feature_cols(df: pd.DataFrame) -> List[str]:
    meta = {"date", "spot_1000", "spot_1500", "move_pts", "label"}
    return [c for c in df.columns if c not in meta]


def _make_models() -> Dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    models: Dict[str, object] = {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=6, random_state=42, n_jobs=-1,
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42),
        ),
    }
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42,
        )
    except ImportError:
        print("  (LightGBM not installed — skipping)")
    return models


def _quarter_folds(test_year: int) -> List[Tuple[str, str]]:
    q = [f"{test_year}-01-01", f"{test_year}-04-01", f"{test_year}-07-01",
         f"{test_year}-10-01", f"{test_year + 1}-01-01"]
    return list(zip(q, q[1:]))


def walk_forward(df: pd.DataFrame, model_name: str, models: Dict) -> pd.DataFrame:
    """Expanding-window quarterly folds → OOS predictions for TEST_YEAR."""
    cols = _feature_cols(df)
    out = []
    for start, end in _quarter_folds(TEST_YEAR):
        train = df[df["date"] < start]
        test = df[(df["date"] >= start) & (df["date"] < end)]
        if train.empty or test.empty:
            continue
        import copy
        model = copy.deepcopy(models[model_name])
        model.fit(train[cols].fillna(0.0), train["label"])
        proba = model.predict_proba(test[cols].fillna(0.0))[:, 1]
        fold = test[["date", "spot_1000", "spot_1500", "move_pts", "label"]].copy()
        fold["proba_up"] = proba
        fold["pred"] = (proba >= 0.5).astype(int)
        out.append(fold)
    return pd.concat(out, ignore_index=True)


def _score(preds: pd.DataFrame) -> Dict[str, float]:
    signed = preds["move_pts"] * np.where(preds["pred"] == 1, 1, -1)
    wins, losses = signed[signed > 0], signed[signed < 0]
    return {
        "accuracy": (preds["pred"] == preds["label"]).mean() * 100,
        "win_rate": (signed > 0).mean() * 100,
        "total_pts": signed.sum(),
        "avg_win": wins.mean() if not wins.empty else 0.0,
        "avg_loss": losses.mean() if not losses.empty else 0.0,
    }


# ─── Backtests on the best model ─────────────────────────────────────────────

def backtest_full_session(preds: pd.DataFrame) -> pd.DataFrame:
    df = preds.copy()
    df["predicted_direction"] = np.where(df["pred"] == 1, "BULLISH", "BEARISH")
    df["actual_direction"] = np.where(df["label"] == 1, "BULLISH", "BEARISH")
    df["probability"] = np.where(
        df["pred"] == 1, df["proba_up"], 1 - df["proba_up"]
    ).round(4)
    df["actual_move_pts"] = df["move_pts"]
    df["result"] = np.where(df["pred"] == df["label"], "WIN", "LOSS")
    out = df[["date", "predicted_direction", "actual_direction", "probability",
              "spot_1000", "spot_1500", "actual_move_pts", "result"]]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(FULL_CSV, index=False)
    return out


def backtest_50_30(preds: pd.DataFrame, index: str = "NIFTY") -> pd.DataFrame:
    rows = []
    for _, r in preds.iterrows():
        d = date.fromisoformat(r["date"])
        ts, spot = spot_series(index, d)
        if not ts:
            continue
        t0 = int(datetime(d.year, d.month, d.day, *ENTRY_TIME, tzinfo=IST).timestamp())
        t1 = int(datetime(d.year, d.month, d.day, *EXIT_TIME, tzinfo=IST).timestamp())
        path = spot[bisect_right(ts, t0):bisect_right(ts, t1)]
        if len(path) < 2:
            continue
        entry = path[0]
        sign = 1.0 if r["pred"] == 1 else -1.0
        exit_px, reason, pnl = path[-1], "TIME", round((path[-1] - entry) * sign, 2)
        for px in path[1:]:
            m = (px - entry) * sign
            if m >= TARGET_PTS:
                exit_px, reason, pnl = entry + sign * TARGET_PTS, "TARGET", TARGET_PTS
                break
            if m <= -STOP_PTS:
                exit_px, reason, pnl = entry - sign * STOP_PTS, "STOP", -STOP_PTS
                break
        rows.append({
            "date": r["date"],
            "predicted_direction": "BULLISH" if r["pred"] == 1 else "BEARISH",
            "probability": round(r["proba_up"] if r["pred"] == 1 else 1 - r["proba_up"], 4),
            "entry_price": round(entry, 2),
            "exit_price": round(exit_px, 2),
            "exit_reason": reason,
            "pnl_pts": pnl,
        })
    out = pd.DataFrame(rows)
    out.to_csv(EXITS_CSV, index=False)
    return out


# ─── Reporting ───────────────────────────────────────────────────────────────

def _monthly(df: pd.DataFrame, pnl_col: str) -> pd.Series:
    return df.groupby(df["date"].str[:7])[pnl_col].sum()


def _print_importance(df: pd.DataFrame, model_name: str, models: Dict) -> None:
    cols = _feature_cols(df)
    train = df[df["date"] < f"{TEST_YEAR}-01-01"]
    import copy
    model = copy.deepcopy(models[model_name])
    model.fit(train[cols].fillna(0.0), train["label"])

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "named_steps"):  # scaled logistic regression pipeline
        imp = np.abs(model.named_steps["logisticregression"].coef_[0])
    else:
        return
    ranked = sorted(zip(cols, imp), key=lambda x: -x[1])[:10]
    print(f"\n  Top 10 features ({model_name}, trained on 2023-2024):")
    for i, (name, v) in enumerate(ranked, 1):
        print(f"    {i:>2}. {name:<28} {v:.4f}")


def main(rebuild: bool = False) -> None:
    df = build_dataset(rebuild=rebuild)
    models = _make_models()

    print(f"\n{'=' * 62}")
    print(f"  WALK-FORWARD MODEL COMPARISON — OOS {TEST_YEAR}")
    print(f"{'=' * 62}")
    results: Dict[str, Tuple[pd.DataFrame, Dict[str, float]]] = {}
    for name in models:
        preds = walk_forward(df, name, models)
        score = _score(preds)
        results[name] = (preds, score)
        print(f"  {name:<20} acc {score['accuracy']:5.1f}%  "
              f"win {score['win_rate']:5.1f}%  "
              f"pts {score['total_pts']:+8.1f}  "
              f"avg win/loss {score['avg_win']:+6.1f}/{score['avg_loss']:+6.1f}")

    best = max(results, key=lambda k: results[k][1]["accuracy"])
    best_preds, best_score = results[best]
    print(f"\n  BEST MODEL: {best} ({best_score['accuracy']:.1f}% OOS accuracy, "
          f"{len(best_preds)} predictions)")

    full = backtest_full_session(best_preds)
    exits = backtest_50_30(best_preds)
    _print_importance(df, best, models)

    full_signed = full["actual_move_pts"] * np.where(
        full["predicted_direction"] == "BULLISH", 1, -1
    )
    full_monthly = full_signed.groupby(full["date"].str[:7]).sum()
    exit_monthly = _monthly(exits, "pnl_pts")

    print(f"\n{'=' * 62}")
    print(f"  FINAL SUMMARY — {best}")
    print(f"{'=' * 62}")
    print(f"  OOS accuracy         : {best_score['accuracy']:.1f}%")
    print(f"  Full session ({len(full)} trades):")
    print(f"    PnL                : {full_signed.sum():+.1f} pts")
    print(f"    Win rate           : {(full['result'] == 'WIN').mean() * 100:.1f}%")
    print(f"    Best / worst month : {full_monthly.idxmax()} ({full_monthly.max():+.1f})"
          f" / {full_monthly.idxmin()} ({full_monthly.min():+.1f})")
    print(f"  50/30 exits ({len(exits)} trades):")
    print(f"    PnL                : {exits['pnl_pts'].sum():+.1f} pts")
    print(f"    Win rate           : {(exits['pnl_pts'] > 0).mean() * 100:.1f}%")
    print(f"    Best / worst month : {exit_monthly.idxmax()} ({exit_monthly.max():+.1f})"
          f" / {exit_monthly.idxmin()} ({exit_monthly.min():+.1f})")
    print(f"\n  Reports: {FULL_CSV}")
    print(f"           {EXITS_CSV}")


if __name__ == "__main__":
    main(rebuild="--rebuild" in sys.argv)

"""
backtester.py — Full-session spread backtest over the binary pyramid signal.

Per consecutive trading-day pair: 15:20 snapshot of the previous day vs the
09:55 snapshot of the current day → majority vote → BULLISH or BEARISH
(SKIP on a vote tie).

Position logic (credit spreads, so max loss is defined by structure):
    BULLISH → Bull Put Spread  at ATM, 50-pt wide
    BEARISH → Bear Call Spread at ATM, 50-pt wide
    entry 10:00, hard square-off 15:00 — no target, no stop.

A credit spread wins when spot finishes on the right side of entry, so the
result is scored on the NIFTY spot move from 10:00 to 15:00:
    BULLISH + move > 0 → WIN;  BEARISH + move < 0 → WIN;  else LOSS.

Output: data/reports/expiry_pyramid_backtest.csv with
    date, direction, overnight_move, pcr_change, oi_shift, iv_skew_change,
    spot_1000, spot_300, actual_move_pts, result
"""

from __future__ import annotations

import sys
from bisect import bisect_right
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from cache_reader import IST, spot_series, trading_dates
from comparator import SNAP_A_TIME, SNAP_B_TIME, compare
from snapshot import load_or_capture

REPORT_DIR = Path(__file__).parent / "data" / "reports"
REPORT_CSV = REPORT_DIR / "expiry_pyramid_backtest.csv"

ENTRY_TIME = (10, 0)
EXIT_TIME = (15, 0)
SPREAD_WIDTH = 50  # ATM credit spread width in pts (documents the instrument)


def _spot_at(index: str, d: date, hh: int, mm: int) -> Optional[float]:
    """NIFTY spot at the latest minute bar at or before hh:mm IST on `d`."""
    ts, spot = spot_series(index, d)
    if not ts:
        return None
    target = int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=IST).timestamp())
    i = bisect_right(ts, target) - 1
    if i < 0:
        return None
    return spot[i]


def _tier_diagnostics(snap_a: dict, snap_b: dict) -> dict:
    """
    Overnight shift diagnostics from the current-expiry tier (most liquid):
    PCR change, put-vs-call OI build differential (pct points), IV-skew change.
    """
    a = snap_a["expiries"].get("current")
    b = snap_b["expiries"].get("current")
    out = {"pcr_change": None, "oi_shift": None, "iv_skew_change": None}
    if a is None or b is None:
        return out
    if a["pcr"] is not None and b["pcr"] is not None:
        out["pcr_change"] = round(b["pcr"] - a["pcr"], 4)
    if a["total_call_oi"] and a["total_put_oi"]:
        call_build = (b["total_call_oi"] - a["total_call_oi"]) / a["total_call_oi"] * 100
        put_build = (b["total_put_oi"] - a["total_put_oi"]) / a["total_put_oi"] * 100
        out["oi_shift"] = round(put_build - call_build, 2)
    if a["iv_skew"] is not None and b["iv_skew"] is not None:
        out["iv_skew_change"] = round(b["iv_skew"] - a["iv_skew"], 4)
    return out


def run_backtest(year: int = 2025, index: str = "NIFTY") -> Optional[pd.DataFrame]:
    dates = trading_dates(index, year)
    if len(dates) < 2:
        print(f"[ERR] Cache holds {len(dates)} trading dates for {year} — nothing to do")
        return None
    print(f"Backtesting {index} {year}: {len(dates)} trading dates ({dates[0]} to {dates[-1]})")

    rows = []
    skips = 0
    for prev_d, d in zip(dates, dates[1:]):
        snap_a = load_or_capture(prev_d, SNAP_A_TIME, index)
        snap_b = load_or_capture(d, SNAP_B_TIME, index)
        if snap_a is None or snap_b is None:
            skips += 1
            continue

        signal = compare(snap_a, snap_b)
        if signal["signal"] == "SKIP":
            skips += 1
            continue

        spot_1000 = _spot_at(index, d, *ENTRY_TIME)
        spot_300 = _spot_at(index, d, *EXIT_TIME)
        if spot_1000 is None or spot_300 is None:
            skips += 1
            continue

        move = round(spot_300 - spot_1000, 2)
        direction = signal["direction"]
        win = (direction == "BULLISH" and move > 0) or (direction == "BEARISH" and move < 0)

        overnight_move = None
        if snap_a.get("spot") and snap_b.get("spot"):
            overnight_move = round(snap_b["spot"] - snap_a["spot"], 2)

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "direction": direction,
            "overnight_move": overnight_move,
            **_tier_diagnostics(snap_a, snap_b),
            "spot_1000": round(spot_1000, 2),
            "spot_300": round(spot_300, 2),
            "actual_move_pts": move,
            "result": "WIN" if win else "LOSS",
        })
        if len(rows) % 25 == 0:
            print(f"  ... {len(rows)} trades simulated (latest {d})")

    if not rows:
        print("[ERR] No trades produced — check cache coverage")
        return None

    df = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_CSV, index=False)
    _print_summary(df, skips)
    print(f"\nReport: {REPORT_CSV}")
    return df


def _print_summary(df: pd.DataFrame, skips: int) -> None:
    # Directional pts: the move signed by the trade direction — what a fully
    # directional position would have captured; the spread monetises its sign.
    signed = df.apply(
        lambda r: r["actual_move_pts"] * (1 if r["direction"] == "BULLISH" else -1),
        axis=1,
    )
    wins = signed[signed > 0]
    losses = signed[signed < 0]
    monthly = signed.groupby(df["date"].str[:7]).sum()

    print(f"\n{'=' * 56}")
    print("  EXPIRY PYRAMID BACKTEST - 10:00-15:00 ATM SPREADS")
    print(f"{'=' * 56}")
    print(f"  Total trades  : {len(df)}")
    print(f"  Skips         : {skips}")
    print(f"  Win rate      : {(df['result'] == 'WIN').mean() * 100:.1f}%")
    print(f"  Directional pts captured : {signed.sum():+.1f}")
    print(f"  Avg win       : {wins.mean() if not wins.empty else 0:+.1f} pts")
    print(f"  Avg loss      : {losses.mean() if not losses.empty else 0:+.1f} pts")
    print(f"  Best month    : {monthly.idxmax()} ({monthly.max():+.1f} pts)")
    print(f"  Worst month   : {monthly.idxmin()} ({monthly.min():+.1f} pts)")

    print("\n  Monthly directional pts:")
    for month, pnl in monthly.items():
        print(f"    {month}  {pnl:+8.1f}")


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    run_backtest(year)

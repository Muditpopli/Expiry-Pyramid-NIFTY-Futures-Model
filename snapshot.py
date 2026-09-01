"""
snapshot.py — Capture one point-in-time picture of the expiry pyramid.

For a given date and time-of-day (1515 = previous-day close snapshot,
0945 = morning snapshot) this reads the current / next / far weekly expiry
chains from the rolling options cache and reduces each to:

    total call OI, total put OI, PCR, OI-weighted net delta (ATM±10 where
    available), ATM call/put IV, IV skew (call − put).

Delta is not stored in the cache, so it is computed per strike with the
project's Black-Scholes GreeksCalculator from the stored IV.

Snapshots are saved to data/snapshots/ as snapshot_YYYY-MM-DD_<HHMM>.json.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from trading_calendar import prev_trading_day
from greeks import GreeksCalculator

from cache_reader import (
    EXPIRY_LEVELS,
    MAX_STRIKE_OFFSET,
    read_bar,
    strike_label,
)

SNAPSHOT_DIR = Path(__file__).parent / "data" / "snapshots"

# NSE moved the NIFTY weekly expiry from Thursday to Tuesday effective
# September 2025 (one-expiry-per-exchange-per-day mandate). The backtest
# spans the transition, so the weekday is date-dependent here rather than
# taken from a fixed rule.
_TUESDAY_REGIME_START = date(2025, 9, 1)


def _expiry_weekday(d: date) -> int:
    return 1 if d >= _TUESDAY_REGIME_START else 3  # Tue after, Thu before


def _expiry_date(d: date, level_idx: int) -> date:
    """Nominal expiry for pyramid tier `level_idx` (0=current, 1=next, 2=far)."""
    wd = _expiry_weekday(d)
    nominal = d + timedelta(days=(wd - d.weekday()) % 7 + 7 * level_idx)
    return prev_trading_day(nominal)


def snapshot_path(d: date, hhmm: str) -> Path:
    return SNAPSHOT_DIR / f"snapshot_{d.strftime('%Y-%m-%d')}_{hhmm}.json"


def _level_snapshot(index: str, d: date, hhmm: str, tier: str, level: str) -> Optional[dict]:
    """Aggregate one expiry level across ATM±MAX_STRIKE_OFFSET strikes."""
    tier_idx = list(EXPIRY_LEVELS).index(tier)
    expiry = _expiry_date(d, tier_idx)
    dte = max((expiry - d).days, 0)

    total_call_oi = 0.0
    total_put_oi = 0.0
    weighted_delta = 0.0
    weighted_vega = 0.0
    strikes_used = []
    atm_iv_call = None
    atm_iv_put = None
    spot = None

    for offset in range(-MAX_STRIKE_OFFSET, MAX_STRIKE_OFFSET + 1):
        call = read_bar(index, d, hhmm, level, offset, "CALL")
        put = read_bar(index, d, hhmm, level, offset, "PUT")
        if call is None and put is None:
            continue
        strikes_used.append(strike_label(offset))

        if call is not None:
            spot = call.spot
            total_call_oi += call.oi
            g = GreeksCalculator.calculate_greeks(
                call.spot, call.strike, dte, call.iv / 100.0, "CE"
            )
            weighted_delta += g["delta"] * call.oi
            weighted_vega += g["vega"] * call.oi
            if offset == 0:
                atm_iv_call = call.iv
        if put is not None:
            spot = spot or put.spot
            total_put_oi += put.oi
            g = GreeksCalculator.calculate_greeks(
                put.spot, put.strike, dte, put.iv / 100.0, "PE"
            )
            weighted_delta += g["delta"] * put.oi
            weighted_vega += g["vega"] * put.oi
            if offset == 0:
                atm_iv_put = put.iv

    total_oi = total_call_oi + total_put_oi
    if total_oi <= 0:
        return None

    iv_skew = None
    if atm_iv_call is not None and atm_iv_put is not None:
        iv_skew = round(atm_iv_call - atm_iv_put, 4)

    return {
        "level": level,
        "expiry": expiry.strftime("%Y-%m-%d"),
        "dte": dte,
        "spot": spot,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "pcr": round(total_put_oi / total_call_oi, 4) if total_call_oi > 0 else None,
        "net_delta": round(weighted_delta / total_oi, 5),
        "net_vega": round(weighted_vega / total_oi, 5),
        "atm_iv_call": atm_iv_call,
        "atm_iv_put": atm_iv_put,
        "iv_skew": iv_skew,
        "strikes_used": strikes_used,
    }


def capture_snapshot(
    d: date, hhmm: str, index: str = "NIFTY", save: bool = True
) -> Optional[dict]:
    """
    Build the full pyramid snapshot for `d` at `hhmm` ("1515" or "0945").
    Returns None when the cache has no data for the date at all.
    """
    expiries = {}
    for tier, level in EXPIRY_LEVELS.items():
        level_snap = _level_snapshot(index, d, hhmm, tier, level)
        if level_snap is not None:
            expiries[tier] = level_snap
    if not expiries:
        return None

    snap = {
        "date": d.strftime("%Y-%m-%d"),
        "time": f"{hhmm[:2]}:{hhmm[2:]}",
        "index": index,
        "spot": next(iter(expiries.values()))["spot"],
        "expiries": expiries,
    }

    if save:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path(d, hhmm), "w") as f:
            json.dump(snap, f, indent=2)
    return snap


def load_or_capture(d: date, hhmm: str, index: str = "NIFTY") -> Optional[dict]:
    """Reuse a saved snapshot when present, otherwise capture and save it."""
    fp = snapshot_path(d, hhmm)
    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    return capture_snapshot(d, hhmm, index)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python snapshot.py YYYY-MM-DD 1515|0945")
        sys.exit(1)
    d = date.fromisoformat(sys.argv[1])
    result = capture_snapshot(d, sys.argv[2])
    if result is None:
        print(f"[ERR] No cache data for {d}")
        sys.exit(2)
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {snapshot_path(d, sys.argv[2])}")

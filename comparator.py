"""
comparator.py — Binary bull/bear vote over the overnight pyramid shift.

Compares snapshot A (15:20 previous day) with snapshot B (09:55 current day).
Per expiry tier (current / next / far), three factors each cast one vote:

    PCR shift   — B > A (PCR rising)            = bearish, falling = bullish
    Delta shift — B > A (net delta rising)      = bullish, falling = bearish
    OI build    — put OI building faster than call OI = bearish, else bullish

9 votes total.  Majority bullish → BULLISH, majority bearish → BEARISH.
A tie (possible only when a factor or tier has no data) → SKIP.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from typing import Optional

from snapshot import load_or_capture

SNAP_A_TIME = "1520"   # previous-day close snapshot
SNAP_B_TIME = "0955"   # morning snapshot (signal ready before 10:00 entry)

BULL, BEAR = 1, -1


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / old * 100, 2)


def _tier_votes(a: dict, b: dict) -> dict:
    """Three binary votes for one expiry tier. 0 = factor had no data."""
    votes = {"pcr": 0, "delta": 0, "oi": 0}

    if a["pcr"] is not None and b["pcr"] is not None:
        votes["pcr"] = BEAR if b["pcr"] > a["pcr"] else BULL

    delta_shift = b["net_delta"] - a["net_delta"]
    votes["delta"] = BULL if delta_shift > 0 else BEAR

    call_build = _pct_change(b["total_call_oi"], a["total_call_oi"])
    put_build = _pct_change(b["total_put_oi"], a["total_put_oi"])
    if call_build is not None and put_build is not None:
        votes["oi"] = BEAR if put_build > call_build else BULL

    return votes


def compare(snap_a: dict, snap_b: dict) -> dict:
    """
    Reduce the A→B shift to a single binary call.

    Returns direction (BULLISH/BEARISH), votes (majority count out of 9),
    and signal (TRADE, or SKIP on a tie).
    """
    tiers = {}
    bull_votes = 0
    bear_votes = 0
    for tier in ("current", "next", "far"):
        a = snap_a["expiries"].get(tier)
        b = snap_b["expiries"].get(tier)
        if a is None or b is None:
            continue
        votes = _tier_votes(a, b)
        tiers[tier] = votes
        bull_votes += sum(1 for v in votes.values() if v == BULL)
        bear_votes += sum(1 for v in votes.values() if v == BEAR)

    if bull_votes > bear_votes:
        direction, majority = "BULLISH", bull_votes
    elif bear_votes > bull_votes:
        direction, majority = "BEARISH", bear_votes
    else:
        direction, majority = None, bull_votes

    return {
        "evening_date": snap_a["date"],
        "morning_date": snap_b["date"],
        "direction": direction,
        "votes": f"{majority}/9",
        "bull_votes": bull_votes,
        "bear_votes": bear_votes,
        "signal": "TRADE" if direction else "SKIP",
        "tiers": tiers,
    }


def compare_dates(evening_date: date, morning_date: date, index: str = "NIFTY") -> Optional[dict]:
    """Load/capture the 15:20 and 09:55 snapshots for the pair and compare."""
    a = load_or_capture(evening_date, SNAP_A_TIME, index)
    b = load_or_capture(morning_date, SNAP_B_TIME, index)
    if a is None or b is None:
        return None
    return compare(a, b)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python comparator.py EVENING_DATE MORNING_DATE")
        sys.exit(1)
    result = compare_dates(date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]))
    if result is None:
        print("[ERR] Missing snapshot data for one or both dates")
        sys.exit(2)
    print(json.dumps(result, indent=2))

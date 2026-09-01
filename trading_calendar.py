"""
trading_calendar.py — minimal holiday-aware trading-day helper.

Vendored from datashots' btst_engine/expiry_manager.py at fork time
(2026-08-31), trimmed to the one function this project actually calls, so
there is no runtime dependency on that repo or its btst_engine package.

Holiday awareness is driven entirely by config.MARKET_HOLIDAYS (populated
from the MARKET_HOLIDAYS_JSON env var); if that list is empty, only weekends
are treated as non-trading days.
"""

from __future__ import annotations

from datetime import date, timedelta

import config


def is_market_holiday(d: date) -> bool:
    if d.weekday() >= 5:
        return True
    return d.strftime("%Y-%m-%d") in config.MARKET_HOLIDAYS


def prev_trading_day(d: date) -> date:
    while is_market_holiday(d):
        d -= timedelta(days=1)
    return d

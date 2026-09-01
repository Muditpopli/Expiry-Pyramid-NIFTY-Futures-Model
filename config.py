"""
config.py — minimal local configuration.

Vendored from the datashots monorepo's root config.py at fork time
(2026-08-31): only the constants greeks.py and trading_calendar.py actually
use, so this project has no runtime dependency on that repo.
"""

import json
import os

# Env format:
# MARKET_HOLIDAYS_JSON=["2026-03-30","2026-08-15"]
_MARKET_HOLIDAYS_RAW = os.getenv("MARKET_HOLIDAYS_JSON", "").strip()
try:
    _parsed_holidays = json.loads(_MARKET_HOLIDAYS_RAW) if _MARKET_HOLIDAYS_RAW else []
except Exception:
    _parsed_holidays = []

MARKET_HOLIDAYS = set()
if isinstance(_parsed_holidays, list):
    for v in _parsed_holidays:
        try:
            MARKET_HOLIDAYS.add(str(v))
        except Exception:
            continue

# --- Black-Scholes / Greeks ---
RISK_FREE_RATE = 0.07  # 7% India
DIVIDEND_YIELD = 0.02  # 2% typical for indices

# --- Risk-buffer constants (used by greeks.py risk-sizing helpers) ---
DELTA_RISK_MOVE = 0.02      # 2% adverse move for delta risk
GAMMA_RISK_MOVE = 0.02      # 2% move for gamma risk
VEGA_RISK_IV_CHANGE = 0.01  # 1% IV increase for vega risk

DELTA_RISK_BUFFER = 1.5
GAMMA_RISK_BUFFER = 2.0
VEGA_RISK_BUFFER = 1.5

MAX_DELTA_NEUTRAL = 0.15  # Position delta must be < 0.15 for "neutral"

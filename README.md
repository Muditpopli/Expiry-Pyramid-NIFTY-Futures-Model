# expiry-pyramid

A multi-expiry options-positioning signal for NIFTY: reads the current, next,
and far weekly option chains, scores the overnight shift in open interest,
put-call ratio, and net delta into a directional call, and (in a second,
ML-based variant) predicts the 10:00→15:00 intraday direction.

This is a live/future strategy under active development, not a closed,
historical backtest-only project.

## How it works

1. **Snapshot** (`snapshot.py`) — captures a point-in-time picture of the
   current/next/far weekly option chains: total call/put OI, PCR, OI-weighted
   net delta, and IV skew. Taken at 15:20 (previous day close) and 09:55
   (current morning).
2. **Compare** (`comparator.py`) — reduces the overnight shift between those
   two snapshots to a binary directional vote: 3 factors (PCR shift, delta
   shift, OI build) across 3 expiry tiers = 9 votes, majority wins, a tie
   skips the day.
3. **ML variant** (`ml_direction.py`, `daily_signal.py`) — expands the same
   overnight shift into a 34-feature row and trains a classifier
   (LogisticRegression in production; XGBoost/RandomForest/LightGBM compared
   in research) to predict whether NIFTY's 10:00→15:00 session will be
   bullish or bearish, validated by expanding-window quarterly walk-forward.
4. **Backtesting** (`backtester.py`) — the credit-spread backtest over the
   binary signal, reading from the same historical options-chain cache.

## Validated results

From the 2.5-year expanding-window walk-forward evaluation (10 quarterly
folds, out-of-sample July 2023 – December 2025, 592 out-of-sample trades):

- **Directional accuracy: 53.5%** — about **1.7 standard deviations above**
  the 50% baseline given the sample size. That is a real but modest edge,
  not a strong statistical margin; it should be read as "better than a coin
  flip, not yet proven decisively so."
- **Win rate: 43.6%**, against a breakeven win rate of roughly 40.4% implied
  by the strategy's own win/loss payoff asymmetry — about 3 percentage
  points of margin above breakeven.
- Applying the signal to a futures overlay position and accounting for
  realistic trading costs, backtested **annualized return was approximately
  19%**, ranging from roughly 3-4% in the weakest half-year to roughly 26%
  in the strongest year. All three calendar years in the window were
  net positive.
- At its worst point, the futures overlay's peak-to-trough drawdown reached
  roughly 21% of the capital allocated to the position, with a longest
  losing streak of 10 consecutive trades.

No rupee amounts, absolute point totals, or margin-size figures are
reported here or in any other document in this repository — see
"Risk and compliance" below for why.

## Setup

```bash
pip install -r requirements.txt
```

### Data

This project does not fetch or generate its own historical options-chain
data. `cache_reader.py` expects a local cache at `data/cache/rolling_options/<INDEX>/`,
one JSON file per (period, expiry-level, strike-offset, side) with
minute-level `iv`, `oi`, `strike`, `spot`, `close`, `timestamp` arrays. That
cache is not included in this repository — populate it yourself (or point
`cache_reader.CACHE_DIR` at wherever you keep it) before running anything
that reads historical data.

`data/snapshots/`, `data/reports/`, `data/ml/`, and `models/` hold this
project's own generated snapshots, backtest reports, feature tables, and
trained model file. All are gitignored — this is a live strategy, and none
of that output is meant to be published.

A small, seeded random sample of the engineered feature table lives in
`data/sample/` (see `data/sample/NOTE.md`) purely so the notebook below can
run end-to-end for anyone reading the code — it is not the dataset the
production model is trained or evaluated on.

## Exploratory Analysis

See [`notebooks/01_eda_and_strategy_overview.ipynb`](notebooks/01_eda_and_strategy_overview.ipynb)
for a plain-language walkthrough of the objective, the three-tier
snapshot-differencing approach, why logistic regression was chosen over
tree ensembles, and the validated results reported honestly (confidence
interval, not just the point estimate) — run against the sample in
`data/sample/`.

## Usage

```bash
python snapshot.py 2025-01-15 1520        # capture one snapshot
python comparator.py 2025-01-15 2025-01-16  # compare two dates
python backtester.py 2025                 # credit-spread backtest for a year
python daily_signal.py                    # today's ML signal
python daily_signal.py --train            # retrain the production model
```

## Repository structure

Flat by design — this is a small, single-purpose project:

- `snapshot.py`, `comparator.py`, `cache_reader.py` — core pyramid signal
- `ml_direction.py`, `daily_signal.py` — ML variant and morning runner
- `backtester.py` — credit-spread backtest over the binary signal
- `greeks.py`, `trading_calendar.py`, `config.py` — small vendored
  dependencies (Black-Scholes Greeks, a holiday-aware trading-day helper,
  and the handful of constants they use) so this project has no runtime
  dependency on any other repository
- `notebooks/` — the EDA/strategy-overview notebook, runnable end-to-end on
  `data/sample/`
- `assets/` — figures exported from the notebook (PNG), referenced from
  the README

## Risk and compliance

- This is a live/future strategy, not investment advice, and not a closed
  backtest whose numbers are locked in. Backtested performance does not
  guarantee future results.
- Only rate- and percentage-based figures are published for this project —
  directional accuracy, win rate, and annualized return — never absolute
  PnL, point totals, or position/margin sizing. Those depend on account
  size and leverage choices that are specific to how it's actually traded,
  and are not meaningful (or appropriate to disclose) out of that context.
- Derivatives trading carries significant risk of loss.

# Backtesting Framework

A signal-generation and backtesting harness for crypto spot data, built around
six technical indicators with **walk-forward optimization** rather than a single
in-sample fit. Streamlit front end, Binance OHLC data.

*Originally my Year 13 A-Level Computer Science coursework submission (2024).
Left largely as submitted — the limitations section below is honest about what
that means.*

---

## Why walk-forward

The easy way to build this is to grid-search an indicator's parameters over your
whole dataset, report the best Sharpe, and be pleased with yourself. That number
is almost always a lie — you have fitted the parameters to the same data you are
measuring on.

Four of the indicators here do grid search *inside* a walk-forward loop instead:

```
train window   →  grid search for best params
                        ↓
                  apply to the NEXT unseen window  →  record returns
                        ↓
                  roll both windows forward, repeat
```

Every return recorded that way comes from parameters chosen without seeing the
period they are measured on. The reported cumulative return is therefore out-of-sample throughout,
which makes it a far more honest number — usually a much worse one than the
in-sample fit, which is the point.

## Indicators

Five of the six are self-contained modules exposing the same interface — fetch
data, compute the indicator, identify signals, grid search, walk forward:

| module | indicator |
|---|---|
| `RSI2.py` | Relative Strength Index — window, overbought/oversold thresholds |
| `MACD2.py` | Moving Average Convergence Divergence |
| `BB2.py` | Bollinger Bands |
| `ATR2.py` | Average True Range |
| `ROC2.py` | Rate of Change |
| `VTS.py` | Volume Trend Signal — used both as a signal and as a volume filter |

`VTS.py` is the exception: it has neither grid search nor walk-forward, so its
output is not comparable to the other five and should not be read as if it were.
`MACD2.py` walks forward but over a fixed parameter set rather than a searched
one.

Signals are written to CSV, which decouples generation from evaluation: the
backtester consumes a signals file and does not care which indicator produced it.
Multiple indicators can be run and combined.

## The backtester

`backtest.py` takes a signals file and a symbol, aligns signals against OHLC
bars, and reports:

- **Price changes over configurable horizons** — how a signal performed after
  *n* bars, rather than assuming one exit rule
- **Sharpe ratio** (configurable risk-free rate)
- **Maximum drawdown**
- **Cumulative returns**
- Signal plots overlaid on price

An optional volume filter (`use_vts` with `min_volume`) drops signals that fired
on thin volume, on the theory that a signal nobody traded into is not a signal.

Duplicate signals are stripped on load — the same timestamp/direction pair firing
twice would otherwise double-count a single event into the returns.

## Running it

```bash
pip install -r requirements.txt
streamlit run appb.py
```

Pick a symbol, a timeframe, and one or more indicators; the app runs the
optimization for each and then backtests the combined signal set.

`cache_assets.py` refreshes the tradeable-symbol list into `binance_assets.csv`
so the picker is not hitting the API on every load.

## Known limitations

Stated plainly, because they bound what the output means:

- **No transaction costs or slippage.** Returns are gross. On short timeframes
  with frequent signals this is the difference between a strategy and a
  spreadsheet.
- **Long-only, single position.** No shorting, no sizing, no concurrent
  positions.
- **Signal alignment is to bar close**, so intra-bar execution is assumed away.
- **VTS is not optimized at all** — no grid search, no walk-forward — so it sits
  outside the out-of-sample guarantee that applies to the rest. Its interaction
  with the volume filter is the roughest part of the codebase.
- **Cumulative return calculation is duplicated** across indicator modules rather
  than living solely in the backtester — it should be computed in one place.

## What I would do differently

- **Move return calculation entirely into `backtest.py`.** Having each indicator
  compute its own cumulative returns means six implementations that can silently
  disagree. The signals file should be the only contract between the two halves.
- **Add a cost model** before trusting any of the numbers.
- **Benchmark against buy-and-hold.** Without it, a positive return says nothing
  about whether the signal beat doing nothing — which for crypto over most
  sample windows is a high bar.
- **Unify the indicator modules behind an explicit base class.** They already
  share a shape by convention; making it a real interface would remove the
  per-indicator branching in `appb.py` — and would have made VTS's missing
  optimization impossible to overlook, since it could not have satisfied the
  interface.

## Related

The prediction-market study in
[polyphemus-findings](https://github.com/kktt667/polyphemus-findings) applies the
same discipline — out-of-sample honesty, explicit caveats about costs and fills —
to a different market structure.

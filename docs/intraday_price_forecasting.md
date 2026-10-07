# Intraday continuous price forecast for MPC

This model supplies the remaining pointwise forecast block for the deterministic
MPC: an hourly DE-LU intraday-price forecast after the day-ahead auction has
cleared.

## Target and market simplification

The target is Energy-Charts' **Intraday Continuous Average Price (DE-LU)**:
the realised volume-weighted continuous-market average for each hourly delivery
period. The source also supplies ID1 and ID3 as supplementary audit columns.

This is intentionally a settlement proxy. A delivery-period average is not an
executable quote observed at an MPC decision instant, and the public source
does not provide trade timestamps or order-book snapshots. A production trader
would require a time-stamped EPEX market-data feed. V1 instead assumes that a
completed delivery hour becomes observable at the start of the next hour, then
forecasts the average indices of the remaining delivery hours.

The decision table contains rows `(as_of, target hour)` where:

```text
as_of = start of an hour after the six preceding delivery hours have completed
target hour > as_of and belongs to the same Europe/Berlin delivery day
```

The model predicts a spread rather than the raw price:

```text
spread[h] = intraday_continuous_average[h] - day_ahead_price[h]
intraday_hat[h] = known_day_ahead_price[h] + spread_hat[h]
```

At intraday decision time, the auction-cleared full day-ahead curve is known.
Predicting the residual spread lets the model treat it as the strong baseline
instead of relearning the daily price shape.

## Features

The compact V1 feature set deliberately avoids unverified live weather or
market-order inputs:

- known target-hour day-ahead price and its full-day mean, spread and range;
- six completed continuous intraday price and spread lags;
- mean, variation, range and momentum across those six observations;
- same target hour's intraday price and spread one day, one week and two weeks
  earlier;
- difference between the latest six-hour mean and the same local-clock window
  one week earlier;
- target calendar fields, decision hour and forecast lead time.

The baseline V1 feature set excludes ratios. DE-LU prices and spreads can be
zero or negative, making a price ratio unstable and difficult to interpret.
Differences in EUR/MWh express the same correction signal without a singular
denominator.

## Parallel enriched V2 feature experiment

The V2 table is intentionally separate from V1, so an expanded feature trial
cannot overwrite the working hybrid. It adds:

- D-7 and D-14 same-clock six-hour price means and standard deviations;
- ratios of the current six-hour mean to the D-7 and D-14 means, with `-99`
  only for an exactly zero denominator and separate validity flags;
- rolling median, IQR, mean absolute price change, local linear slope and
  last-value-minus-mean for price and spread;
- a local AR(1) forecast for price and spread. This is a compact
  ARIMA(1,0,0) calculation on six completed observations. For the next
  delivery hour it makes two AR steps: the current delivery period is still
  incomplete and is not observable at the decision time;
- the known change in the day-ahead curve from decision hour to target hour.

Reproduce the isolated trial with:

```bash
energy build-intraday-price-data --project-root . --feature-version v2
energy run-intraday-price-experiment --project-root . \
  --data-path data/features/intraday_price/intraday_price_mpc_v2_2024_2025.parquet \
  --feature-version v2 --artifact-name intraday_price_v2_features
```

The resulting V2 CatBoost selection is Huber/depth 4. On the Q4 2024
next-hour validation it reaches MAE **14.51 EUR/MWh**, versus **15.52** for the
day-ahead curve and **14.89** for the V1 CatBoost. The January–September 2025
hybrid result is MAE **12.64** and RMSE **21.51**, compared with V1's **12.67**
and **21.56** on its nearly identical evaluation set. The AR feature and mean
ratios have low feature importance; the main value remains the day-ahead curve,
calendar and most recent spread information.

The public Energy-Charts payload contains neither traded volumes nor order-book
information. It therefore cannot represent *no-trade effects*: those mean that
no transaction occurred for the same product in a small trading interval, not
that the electricity price was zero. Adding that signal requires timestamped
contract-level trades or market-data feed from EPEX.

### Blend calibration

A small grid search evaluates a shrinkage version of the V2 next-hour rule:

```text
intraday_hat = day_ahead_price + alpha * CatBoost(spread_hat),  lead_hours = 1
intraday_hat = day_ahead_price,                                 lead_hours >= 2
```

The grid is selected only on Q4 2024 next-hour MAE. It chooses
`alpha = 0.70`: 70% of the CatBoost correction and 30% of the day-ahead
baseline. On Q4 2024, this reduces next-hour MAE from **15.52** to **14.21
EUR/MWh**, while RMSE increases from 49.26 to 52.67. On the later 2025 holdout,
the fixed blend reduces MAE from **11.60** to **9.96** and RMSE from 19.21 to
**17.86**. This is an optional calibrated policy for the economic MPC
backtest, not a replacement selected from 2025 results. The full grid and its
configuration are saved under `artifacts/experiments/intraday_price_v2_blending/`.

## Run

```bash
energy collect-intraday-price-data --project-root . --years 2024 2025
energy build-intraday-price-data --project-root .
energy run-intraday-price-experiment --project-root .
```

Raw yearly Energy-Charts payloads are retained under `data/raw`; the audited
hourly source is written to:

```text
data/processed/prices_intraday_continuous_de_lu_hourly.csv.gz
```

The feature table is saved to:

```text
data/features/intraday_price/intraday_price_mpc_2024_2025.parquet
```

The experiment uses two chronological stages. It first develops candidate
CatBoost spread corrections from January through September 2024 and selects
them, if at all, on the untouched Q4 2024 window. It then refits the chosen
challenger on all eligible 2024 rows and reports January–September 2025 once.
The final quarter of 2025 is excluded because the matching hourly day-ahead
anchor becomes incomplete after the market's switch to 15-minute products; V1
does not average quarter-hour products into an artificial hourly curve.

The experiment compares:

1. the known day-ahead price;
2. latest observed intraday-price persistence;
3. a regularised CatBoost forecast of the intraday-day-ahead spread.

The first is the economically meaningful baseline. The second is a deliberately
naïve short-term time-series check, not a recommended market strategy. A
challenger can correct **only the next delivery hour** when it improves MAE at
`lead_hours = 1` on the 2024 validation window. All later hours retain the
known day-ahead curve.

## Result and decision

V1 uses a hybrid point forecast:

```text
lead_hours = 1:  intraday_hat = day_ahead_price + CatBoost(spread_hat)
lead_hours >= 2: intraday_hat = day_ahead_price
```

The regularised CatBoost configuration was selected only from Q4 2024, where
it reduced next-hour MAE from **15.52** to **14.89 EUR/MWh**. Across all
remaining-hour rows in that validation period, the hybrid reduces MAE from
**17.69** to **17.63 EUR/MWh**, although its RMSE is slightly higher (56.45 vs
56.07). This means the correction gives more small improvements but still has
some large misses; the later economic MPC backtest must therefore remain the
final decision criterion.

The untouched January–September 2025 holdout supports the same rule: hybrid
MAE is **12.67 EUR/MWh** and RMSE **21.56 EUR/MWh**, compared with **12.79** and
**21.60** for the day-ahead curve. Applying the raw CatBoost correction to all
future hours is worse (MAE **12.95**, RMSE **24.20**). The result is saved
alongside predictions, feature importances and per-lead metrics under
`artifacts/experiments/intraday_price_v1/`.

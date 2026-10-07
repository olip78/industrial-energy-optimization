# Economic replay V2: dynamic daytime battery charging

This experiment reruns the four deterministic economic strategies with an
explicit battery state of charge. V1 artifacts remain unchanged.

## Setting

- Replay period: 261 complete delivery days from January through September
  2025.
- Operating window: local hours 06:00--22:00, represented by the 16 hourly
  intervals 6 through 21.
- Production demand: 200 kWh per operating day.
- Battery capacity and initial state of charge: 44.16 kWh.
- Maximum charge and discharge: 5 kWh per hour.
- Charge and discharge efficiencies: 0.98 each.
- Daytime charge is permitted only when the price available to that policy at
  decision time is at most zero.
- A binary operating mode prevents simultaneous charging and discharging.
- The previous completed night's mean day-ahead price over the nine local
  labels 22:00, 23:00 and 00:00--06:00 is floored at zero before entering the
  conservative discharge reference cost. Nine equally weighted 5 kWh blocks
  represent a 45 kWh night-charge purchase.
- There is no terminal state-of-charge constraint.

The battery reference cost remains a deliberately conservative dispatch
penalty. Energy charged during the day is purchased through the market balance,
but its later discharge also receives the common night-reference penalty. This
is the agreed approximation rather than exact inventory-cost tracing.

## Strategies

`Rule-based` retains its original definition: it has no daily price forecast,
PV forecast or MPC and therefore does not add daytime charging. The other four
strategies use the new dynamic-battery MILP:

1. `Day-ahead only` decides charge, discharge and load from D-1 point forecasts.
2. `Deterministic` submits the same day-ahead position and replans the remaining
   physical schedule every hour with updated PV and intraday-price forecasts.
3. `Day-ahead Oracle` uses factual PV and factual day-ahead prices to choose
   one physical load and battery schedule. Its nomination equals that
   schedule's net position, intraday prices do not enter the decision, and its
   realised imbalance is zero.
4. `CQR + empirical copula, CVaR λ=0.10` draws 500 joint conditional PV and
   price trajectories and minimizes expected cost plus 0.10 times CVaR95. The
   submitted position still uses the frozen point-PV centre. Its charge mask is
   fixed from the point day-ahead price forecast, so scenario outcomes cannot
   reveal in advance whether charging is feasible.

## Result

| Strategy | Total cost, EUR | Mean daily cost, EUR | P95 daily cost, EUR | Maximum day, EUR | Charge, kWh | Discharge, kWh | Mean final SoC, kWh |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based | 3,258.85 | 12.486 | 26.954 | 51.946 | 0.00 | 1,299.32 | 39.08 |
| Day-ahead only | 3,039.94 | 11.647 | 26.104 | 46.323 | 71.24 | 3,445.39 | 30.96 |
| CQR + empirical copula, CVaR λ=0.10 | 3,050.18 | 11.687 | 26.205 | 48.364 | 40.62 | 4,019.40 | 28.60 |
| Deterministic + MPC | **2,982.61** | **11.428** | **25.559** | **43.544** | 152.06 | 2,532.54 | 34.83 |
| Day-ahead Oracle | 2,961.04 | 11.345 | 25.606 | 41.608 | 187.27 | 2,955.64 | 33.31 |

Relative to rule-based, day-ahead control saves 218.92 EUR and deterministic
MPC saves 276.24 EUR. MPC adds 57.32 EUR beyond the frozen day-ahead schedule.
The remaining deterministic-to-day-ahead-Oracle gap is only 21.58 EUR.
The Oracle does not use intraday information and cannot profit from deliberate
market imbalance. The stochastic day-ahead policy remains 89.14 EUR above this
strictly physical perfect-information benchmark.

The stochastic policy costs 10.24 EUR more than deterministic day-ahead over
the full period. In return, its realised CVaR95 falls from 31.861 to 31.580
EUR/day, a reduction of 0.281 EUR per tail day. This remains an insurance
trade-off rather than an average-cost improvement. Deterministic MPC is still
cheaper because it receives updated information during the delivery day.

![Seven-day mean economic result by strategy](../artifacts/experiments/economic_backtest_v2_dynamic_charge/economic_strategy_7d_average.png)

Daytime charging is sparse because the battery starts every operating window
full, must first create headroom through discharge and may charge only under a
non-positive decision price. Day-ahead charges on 10 days and 18 hours;
deterministic MPC charges on 18 days and 37 hours; the day-ahead Oracle charges
on 18 days and 43 hours. Updated intraday information therefore finds roughly
twice as many eligible hours as the D-1 point forecast, while the Oracle finds
the factual non-positive day-ahead hours without using intraday prices.

## Physical and information audit

- State of charge stays within 0--44.16 kWh.
- No hour contains simultaneous charge and discharge.
- No optimized charge is executed with a positive decision-time price.
- The day-ahead Oracle has zero physical deviation in every hour and exactly
  zero intraday settlement cost over the replay.
- Oracle nomination and execution are the same physical schedule; cross-market
  arbitrage is excluded by construction.
- The hourly artifact stores both the decision-time charge price and the later
  factual prices. A factual positive price after a negative forecast is thus a
  forecast error, not look-ahead leakage.
- One replay day, 2025-09-17, has a negative previous-night mean reference
  price. It is correctly floored from -0.4533 to 0 EUR/MWh.

## Forecast plan versus fact

The economic replay consumes the same frozen 2025 holdout forecasts evaluated
in the temporal backtest. The diagnostic figure below compares those forecasts
with facts over the optimizer's local 06:00--22:00 operating window. To keep
the nine-month trajectories readable, both panels show trailing seven-day
means. The price panel is the daily mean of the 16 hourly prices; the PV panel
is daily energy obtained by summing the 16 hourly mean-power values and applying
the scenario's factor of 20 from the measured approximately 0.5 kW reference
profile to the modelled 10 kWp installation.

![Day-ahead price and PV plan versus fact](../artifacts/experiments/economic_backtest_v2_dynamic_charge/day_ahead_forecast_plan_fact_7d_average.png)

The MAE and RMSE annotations remain hour-level metrics calculated before daily
aggregation or smoothing. Thus, the plot shows medium-term calibration and
seasonality while the annotations retain the operational forecast error.

### Conditional price ranges

The selected stochastic strategy uses conformalized conditional quantile
regression for its marginal forecasts and an empirical copula to preserve the
joint hourly and PV/price dependence. The figure below shows the P05--P95 and
P10--P90 marginal price ranges over the 14-day window around the largest
absolute deviation from the conditional median. The empirical copula does not
change these one-hour ranges; it determines which hourly quantiles occur
together inside a sampled daily trajectory.

![CQR price intervals around the largest surprise](../artifacts/experiments/economic_backtest_v2_dynamic_charge/cqr_price_intervals_largest_surprise.png)

## Reproduction and artifacts

```bash
energy run-dynamic-charge-economic-backtest \
  --project-root . \
  --artifact-name economic_backtest_v2_dynamic_charge

python scripts/plot_dynamic_charge_weekly_average.py --project-root .
python scripts/plot_cqr_price_intervals.py --project-root .
python scripts/analyze_oracle_regret.py --project-root .
python scripts/analyze_price_spike_losses.py --project-root .
```

Outputs are stored under
`artifacts/experiments/economic_backtest_v2_dynamic_charge/`:

- `daily_results.csv` contains the economic ledger, charge, discharge and SoC
  summaries;
- `hourly_decisions.parquet` contains the physical schedules, SoC trajectory,
  factual prices and decision-time charge price;
- `coverage.csv` records included and excluded dates;
- `summary.json` contains aggregate results;
- `experiment_config.json` freezes the physical and information contract.
- `economic_strategy_7d_average.png` compares weekly-smoothed realised costs;
- `day_ahead_forecast_plan_fact_7d_average.png` compares price and PV forecasts
  with their realised values;
- `forecast_plan_fact_daily.csv` contains the unsmoothed daily series used by
  the forecast figure.
- `stochastic_scenario_objectives.csv` contains the predicted mean, CVaR and
  combined objective for the selected CQR-copula policy;
- `cqr_price_intervals_largest_surprise.png` shows calibrated conditional price
  ranges around the largest 2025 median forecast error;
- `cqr_price_interval_stress_window.csv` contains the hourly quantiles and facts
  used in that figure;
- `oracle_regret_daily_error_analysis.csv` and
  `oracle_regret_error_analysis.md` decompose the stochastic policy's cost gap
  against the non-speculative day-ahead Oracle;
- `oracle_regret_decomposition_top_days.png` and
  `oracle_regret_case_studies.png` visualize the largest remaining errors.

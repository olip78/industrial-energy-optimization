# Economic replay V1: rule-based policy, day-ahead control, deterministic MPC and Oracle

This is the first end-to-end economic comparison in the project. It replays
the frozen 2025 point forecasts through the common V1 settlement ledger; it
does not train, tune or select any model during the replay.

## Reference site and common accounting

All strategies use the same fixed object from
[the design document](design_document_draft.md#35-зафиксированная-учебная-установка-v1):

- a 200 kWh daily production requirement in the local 06:00–22:00 working
  window, with hourly load bounds of 5–20 kWh;
- a 10 kWp scaled MPVBench `1a` PV shape;
- two reference HVM 22.1 battery units: 44.16 kWh available at the start of a
  day and a shared AC delivery cap of 5 kWh per hour;
- a 0.98 charge and discharge efficiency and 0.05 EUR/kWh delivered-energy
  degradation cost;
- the mean day-ahead price over the nine local labels 22:00, 23:00 and
  00:00--06:00 on the preceding completed night as the known accounting
  reference; nine equally weighted 5 kWh blocks represent a 45 kWh purchase.

Every policy is charged for its fixed day-ahead position at the realised
day-ahead price, its factual deviation at Energy-Charts' realised hourly
`Intraday Continuous Average Price (DE-LU)`, and its delivered battery energy
at the same full marginal cost. The public intraday index is a settlement proxy,
not an executable quote.

## Strategies

### Rule-based baseline

The policy is fitted only to 2024 historical day-ahead prices. It freezes a
median price score by local hour, workday/weekend and meteorological season.
Within the operating window, a capped inverse-rank rule places production at
20 kWh in the cheapest typical hours, 5 kWh in the most expensive ones, and
allocates the intermediate hours monotonically in between. It fills the battery
in the same historical price order when the score exceeds its full marginal
cost.

It buys this fixed load profile day-ahead:

```text
q_DA_rule[h] = load_rule[h]
```

It has no PV forecast, price forecast or MPC. Actual PV and scheduled battery
discharge are recorded as negative intraday deviations.

### Deterministic point-forecast policy

At the D-1 decision it uses the frozen 2024-trained forecasts:

- `spatial_weather_catboost` for the 24-hour day-ahead price curve;
- `day_ahead_catboost` for the scaled PV profile.

It solves the day-ahead LP and fixes `q_DA`. During the delivery day it
re-solves the remaining LP every hour. The PV forecast defaults to the frozen
day-ahead profile and uses `direct_residual_catboost` only where the matching
point-in-time MPC prediction exists. The intraday forecast uses the known
auction-cleared day-ahead curve for the remaining hours, with the preselected
0.70 CatBoost spread correction only for the immediate next hour.

The MPC never changes `q_DA`; it changes only the unexecuted flexible load and
remaining battery discharge. Its realised difference from `q_DA` is settled at
the actual intraday proxy price.

### Day-ahead-only attribution control

This is not an additional candidate policy. It is the deterministic policy
with exactly the same D-1 forecasts, day-ahead LP and fixed position `q_DA`,
but it executes that original load and battery schedule unchanged through the
delivery day. It cannot use the short-term PV correction or intraday-price
correction and never re-solves the LP.

The difference between this control and deterministic MPC is therefore the
clean incremental value of intraday replanning in the V1 simulator. The
difference from the rule-based baseline also includes the value of the D-1
forecasts and replacing the historical-clock rule with the day-ahead LP.

### Day-ahead Oracle: perfect DA information

Oracle is an intentionally unattainable day-ahead lower bound under the same
V1 physical model. Before the D-1 decision it sees the factual PV profile and
realised day-ahead price curve. It chooses one physical production and battery
schedule, and its day-ahead nomination is exactly that schedule's net position.

Intraday prices do not enter the Oracle decision. Nomination and physical
execution are identical, so the Oracle has zero imbalance and zero intraday
settlement cost. It measures the remaining value of perfect PV and day-ahead
price information without cross-market speculation.

## Temporal and data contract

Run:

```bash
energy run-economic-backtest --project-root .
```

The default replay is 1 January–30 September 2025. Forecast models were fitted
on 2024, and the intraday blend weight of 0.70 was selected on Q4 2024; no 2025
economic outcome affected those choices.

The simulator uses only regular 24-hour local delivery days. Twelve dates are
excluded from the requested window: ten lack the matching frozen day-ahead
price forecast, one is the daylight-saving 23-hour day, and one lacks the
matching day-ahead PV prediction. The final sample contains **261 days** and
52,200 kWh of required production.

## Outputs

The command writes `artifacts/experiments/economic_backtest_v1/`:

- `daily_results.csv` — cost components, positions, deviations, PV and battery
  use for all four evaluated trajectories;
- `hourly_decisions.parquet` — executable hourly load/discharge decisions and
  realised settlement values;
- `coverage.csv` — every requested date and explicit exclusion reason;
- `summary.json` and `experiment_config.json` — aggregate results and all
  frozen choices.

## Initial frozen result

| Metric, January–September 2025 | Rule-based | Day-ahead only | Deterministic MPC | Oracle |
|---|---:|---:|---:|---:|
| Total cost, EUR | 3,258.09 | 3,039.83 | 2,983.61 | **2,964.77** |
| Day-ahead component, EUR | 3,691.28 | 2,587.30 | 2,587.30 | 2,602.85 |
| Intraday-deviation component, EUR | -561.93 | -1.15 | 98.83 | **0.00** |
| Battery component, EUR | 128.74 | 453.68 | 297.48 | 361.93 |
| Battery discharge, kWh | 1,309.92 | 3,423.12 | 2,439.88 | 2,867.36 |
| Equivalent full cycles | 29.66 | 77.52 | 55.25 | 64.93 |

The full deterministic policy saves **274.48 EUR (8.42%)** over the whole
sample relative to rule-based. The attribution control separates this total
without changing the D-1 decision:

| Increment | Saving, EUR | Interpretation |
|---|---:|---|
| Rule-based → day-ahead only | 218.26 | D-1 price/PV forecasts and day-ahead LP instead of the historical-clock rule |
| Day-ahead only → deterministic MPC | 56.22 | Replanning the remaining day with the same fixed `q_DA` |
| Rule-based → deterministic MPC | 274.48 | Full V1 system effect |

Thus MPC accounts for **20.5%** of the measured full-system saving, but only
**1.85%** of the cost of the day-ahead-only control: 0.22 EUR per replayed day
on average and 0.04 EUR on the median day. It is cheaper on 145 of 261 days
(55.6%), more expensive on 53, and identical on 63; a small number of winter
days produces most of its aggregate effect.

The ledger shows what MPC is doing. It does **not** generate a standalone
intraday trading gain in this first setup: its intraday-deviation component is
99.98 EUR worse than day-ahead-only. Instead it avoids 983.24 kWh of battery
discharge (22.27 equivalent full cycles), reducing the full battery cost by
156.20 EUR and leaving the net 56.22 EUR saving. This is the combined effect of
the updated PV and short-term intraday forecasts; this experiment does not
separately identify their individual contributions. The non-speculative
day-ahead Oracle is another **18.84 EUR** below deterministic MPC. This small
remaining gap measures perfect PV and day-ahead price information rather than
perfect intraday arbitrage.

This validates the deterministic workflow and its accounting contract. It does
not yet test whether a stochastic policy adds value; SAA and CVaR are the next
separate strategies under the same replay interface.

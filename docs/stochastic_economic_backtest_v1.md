# Stochastic day-ahead economic replay V1

This experiment connects the joint residual-bootstrap scenarios to the common
2025 economic ledger. Its purpose is to test whether scenario-based decisions
improve realised cost or reduce the expensive tail. It does not compare the
point accuracy of scenario means.

## Decision and uncertainty contract

The replay uses the same 261 complete delivery days, 200 kWh daily load,
10 kWp PV proxy, 44.16 kWh daytime battery resource and factual settlement
ledger as the deterministic experiment. Each day uses 500 joint PV/price
scenarios sampled from 210 rolling-origin residual days in 2024. Residuals are
centred by hour so that the experiment primarily tests uncertainty shape rather
than a historical bias correction.

One load schedule `L[h]` and one battery-discharge schedule `d[h]` are shared by
all scenarios. The optimizer therefore cannot select an action after learning
which scenario will occur. The submitted position is

```text
q_DA[h] = L[h] - G_hat[h] - d[h]
```

where `G_hat[h]` is the point PV forecast. In the first stochastic market
approximation, the sampled day-ahead price is also the scenario's intraday
settlement proxy. Scenario cost is therefore

```text
C[s](L, d) = sum over h {
    P[s, h] * (L[h] - G[s, h] - d[h])
    + c_bat * d[h]
}
```

The risk-neutral policy minimizes the scenario mean. The risk-aware policies
solve

```text
minimize over L, d:  E[C] + λ * CVaR_0.95(C)
```

for `λ` in `{0.05, 0.10, 0.25, 0.50, 1.0}`. CVaR is represented by the
standard linear auxiliary-variable formulation, so every problem remains an
LP. Factual replay then prices `q_DA` at the realised day-ahead price and the
actual deviation at the realised intraday continuous average proxy.

## 2025 result

The realised CVaR95 below is the mean daily cost among the 14 most expensive
days in each strategy's own 2025 distribution.

| Strategy | Total cost, EUR | Change vs deterministic DA, EUR | Realised CVaR95, EUR/day | CVaR95 reduction, EUR/day | Maximum day, EUR | Battery discharge, kWh |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic day-ahead | 3,039.83 | 0.00 | 31.862 | 0.000 | 46.323 | 3,423.12 |
| Bootstrap SAA | 3,037.30 | -2.53 | 31.813 | 0.049 | 46.323 | 3,404.80 |
| CVaR, λ = 0.05 | 3,040.04 | +0.21 | 31.603 | 0.259 | 45.533 | 3,673.76 |
| CVaR, λ = 0.10 | 3,046.35 | +6.52 | 31.431 | 0.432 | 45.533 | 3,918.22 |
| CVaR, λ = 0.25 | 3,081.45 | +41.62 | 31.397 | 0.465 | 45.533 | 4,669.56 |
| CVaR, λ = 0.50 | 3,141.00 | +101.17 | 31.327 | 0.535 | 45.388 | 5,724.91 |
| CVaR, λ = 1.00 | 3,262.20 | +222.36 | 31.255 | 0.607 | 44.892 | 7,292.76 |

Here a positive change is an extra cost. Bootstrap SAA is economically
indistinguishable from deterministic day-ahead: its apparent 2.53 EUR saving
has a paired seven-day block-bootstrap 95% interval from 0.68 EUR extra cost
to 6.07 EUR saving. That is the expected result for a linear risk-neutral
problem whose centred scenario mean is close to the point forecast.

CVaR behaves in the intended qualitative direction. It spends more battery
energy to protect expensive joint PV/price outcomes, lowers the average of the
worst realised days and reduces the maximum daily cost. It does not improve
the ordinary p95 quantile monotonically because the objective targets the mean
beyond the quantile rather than the quantile itself.

The useful part of the frontier is around λ = 0.05--0.10. At 0.05 the
total extra cost over nine months is only 0.21 EUR while the observed tail mean
falls by 0.259 EUR/day. At 0.10 the extra cost is 6.52 EUR (0.21%) and the tail
mean falls by 0.432 EUR/day (1.36%). Larger weights buy little additional tail
reduction and sharply increase battery cycling and total cost.

All day-ahead-only variants have the same realised intraday-deviation
component, -1.15 EUR over the replay. Algebraically, substituting
`q_DA[h] = L[h] - G_hat[h] - d[h]` into the factual deviation gives
`G_hat[h] - G_fact[h]`, so changing load or discharge does not change that
component. The frontier is therefore easy to audit: CVaR lowers the day-ahead
purchase component, but increasingly pays the overnight energy and degradation
cost of extra battery discharge.

The tail evidence is still exploratory. The paired weekly block-bootstrap 95%
interval for the CVaR95 reduction crosses zero for every tested weight; for
λ = 0.10 it is -0.055 to 1.394 EUR/day. Only 210 unconditional residual
days support the scenario distribution, and the realised 5% tail contains 14
days. The result demonstrates the mechanism and a plausible risk/cost tradeoff,
but it is not strong evidence of a stable production benefit.

For context, deterministic MPC costs 2,983.61 EUR and has realised CVaR95 of
31.047 EUR/day, while the non-speculative day-ahead Oracle costs 2,964.77 EUR and has
CVaR95 of 30.420 EUR/day. MPC sees updated information during the day, so these are context
lines rather than like-for-like day-ahead competitors. A later experiment can
hold each stochastic day-ahead position fixed and add the same MPC policy on
top of it.

## Reproduction and outputs

```bash
energy run-stochastic-economic-backtest --project-root .
```

The command writes `artifacts/experiments/stochastic_economic_backtest_v1/`:

- `daily_results.csv` contains realised daily ledger components for every
  strategy;
- `hourly_decisions.parquet` contains common executable hourly schedules and
  positions;
- `scenario_objectives.csv` records predicted mean, CVaR and objective values;
- `coverage.csv` records used and excluded dates;
- `summary.json` contains the aggregate frontier and paired weekly
  block-bootstrap intervals;
- `experiment_config.json` freezes the market approximation, scenario count,
  risk weights and random seed.

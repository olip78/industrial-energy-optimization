# Economic replay V4: final deterministic formulation

## Status

The 2025 economic replay has completed for 261 delivery days from January
through September. Its outputs are stored in
`artifacts/experiments/economic_backtest_v4_final_formulation`; the archived
V3.1 result remains unchanged.

## Result

| Strategy | Total cost, EUR | Mean/day, EUR | Charge, kWh | Discharge, kWh | Mean final SoC, kWh | Curtailment, kWh |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based | 7,992.53 | 30.623 | 473.85 | 2,179.96 | 37.42 | 1,280.29 |
| Point day-ahead | 7,547.38 | 28.917 | 1,860.18 | 6,286.64 | 26.57 | 0.00 |
| Point day-ahead + MPC | **7,484.07** | **28.675** | 2,243.96 | 6,039.56 | 28.97 | 23.07 |
| Day-ahead Oracle | 7,431.97 | 28.475 | 2,237.67 | 6,368.76 | 27.66 | 21.47 |

The full Rule-based-to-Oracle opportunity is EUR 560.56. Point day-ahead
captures EUR 445.15, or 79.41% of that opportunity, and remains EUR 115.41
above Oracle. MPC then saves another EUR 63.31 and closes 54.86% of the
remaining point-day-ahead gap. The complete deterministic system captures
90.71% of the Rule-based-to-Oracle opportunity and finishes EUR 52.10 above
Oracle.

Battery discharge corresponds to 49.37 equivalent full cycles for Rule-based,
142.36 for point day-ahead, 136.77 for point day-ahead + MPC and 144.22 for
Oracle over the replay. These are throughput indicators based on delivered
discharge energy divided by 44.16 kWh; efficiency and degradation remain
accounted for separately in the ledger.

The physical audit passes: every strategy covers the same 261 days and exactly
200 kWh/day of production; SoC stays inside 0–44.16 kWh; charge and discharge
never overlap; curtailment stays inside factual PV; Rule-based charge never
exceeds factual PV; and Oracle intraday deviation is exactly zero.

## Compared strategies

The default run contains four strategies:

1. `rule_based` — a training-history price shape fixes load and discharge
   priorities; the day-ahead purchase equals the load profile; factual PV can
   refill headroom created by earlier discharge; residual PV is curtailed only
   when the known day-ahead clearing price is negative.
2. `day_ahead_only` — point PV and price forecasts determine load, battery,
   curtailment and the day-ahead position; the physical plan is then executed
   without intraday replanning.
3. `deterministic` — the same day-ahead position is fixed, while hourly MPC
   reoptimizes all remaining load, battery actions and curtailment.
4. `oracle` — factual PV and factual day-ahead prices are known before the
   decision; the nomination equals the physical schedule and has zero
   intraday deviation.

The legacy CQR-copula/CVaR strategy is excluded by default because the current
review covered the deterministic formulations. It can still be requested with
`--include-stochastic`; its formulation should be audited separately before it
is used in the final comparison.

## Common physical and financial rules

The working window is 06:00–22:00, with delivery hours 6 through 21. For each
hour,

$$
g_h=L_h+C_h-G_h+U_h-D_h,
\qquad
\Delta_h=g_h-q_h^{DA},
$$

where $U_h$ is curtailed PV. Battery state follows

$$
E_{h+1}=E_h+\eta_cC_h-\frac{D_h}{\eta_d},
$$

with capacity, power and mutually exclusive charge/discharge constraints. The
full ledger is

$$
\begin{aligned}
C_{day}={}&
\sum_h\frac{P_h^{DA}}{1000}q_h^{DA}
+\sum_h\frac{P_h^{ID}}{1000}\Delta_h
+\sum_h t_h^{imp}I_h
+\sum_h t_h^{export}X_h\\
&+c_{deg}\sum_hD_h
+v_T(E_{max}-E_T).
\end{aligned}
$$

The terminal value uses the rolling mean of the latest 14 completed nights.
Each nightly mean covers local labels 22, 23 and 0 through 6 and is floored at
zero before aggregation. A fixed daily grid charge is not applied because it
is common to every strategy and cannot affect dispatch or pairwise savings.

## Causal execution rules

- The first MPC decision is taken immediately before 06:00. Row `h` of each
  point-in-time forecast matrix contains only information available before
  delivery hour `h`.
- Only $q_h^{DA}$ is fixed after the auction. MPC can change load, charge,
  discharge and curtailment; the change is settled through $\Delta_h$.
- Optimized curtailment is executed as the planned fraction of factual PV. If
  forecast PV is zero, the fraction is zero.
- Grid tariffs are calculated from physical import and export, never from the
  financial deviation.
- The daily ledger is calculated once from executed actions and realised
  prices. It is not accumulated from overlapping MPC horizons.

## Run command

```bash
energy run-dynamic-charge-economic-backtest \
  --project-root . \
  --start 2025-01-01 \
  --end 2025-09-30 \
  --artifact-name economic_backtest_v4_final_formulation \
  --grid-tariff-preset pforzheim-slp-2025
```

The output includes daily costs, hourly decisions, coverage diagnostics and an
experiment configuration. Daily and hourly artifacts include curtailed and
used PV, battery charge/discharge and SoC, the fixed day-ahead position,
intraday deviation, and physical grid import/export.

## Required post-run checks

- all four strategies cover exactly the same delivery days;
- every strategy completes 200 kWh of production inside the working window;
- SoC remains within 0–44.16 kWh and charge/discharge never overlap;
- $0\le U_h\le G_h$ for every executed hour;
- the Oracle has zero intraday deviation;
- ledger components reconcile to total cost;
- rule-based charge never exceeds factual PV and never creates grid charging;
- the default output contains no fixed daily grid-cost component.

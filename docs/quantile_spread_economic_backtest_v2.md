# Quantile scenarios and stochastic day-ahead optimization V2

## Decision contract

This experiment evaluates probabilistic day-ahead optimization without MPC.
For every delivery day, one load, battery and curtailment plan is selected
before the scenario is known. No physical control is scenario-dependent and
the chosen plan is not reoptimized during factual settlement.

For operating hour $h$ and scenario $s$, define the common decisions
$L_h$, $C_h$, $D_h$ and curtailment fraction $u_h$. The day-ahead nomination
is tied to the frozen point PV forecast:

$$
q_h^{DA}
=L_h+C_h-D_h-(1-u_h)\widehat G_h.
$$

This prevents deliberate day-ahead/intraday speculation. Scenario physical
exchange and deviation are

$$
g_{s,h}=L_h+C_h-D_h-(1-u_h)G_{s,h},
$$

$$
\Delta_{s,h}=g_{s,h}-q_h^{DA},
\qquad
P_{s,h}^{ID}=P_{s,h}^{DA}+S_{s,h},
$$

where $S_{s,h}$ is the sampled intraday-minus-day-ahead spread. The complete
scenario cost uses the final V4 physical ledger:

$$
\begin{aligned}
J_s={}&\sum_h \frac{P_{s,h}^{DA}}{1000}q_h^{DA}
+\sum_h \frac{P_{s,h}^{ID}}{1000}\Delta_{s,h}\\
&+\sum_h t_h^{imp}I_{s,h}
+\sum_h t_h^{export}X_{s,h}
+c_{deg}\sum_h D_h
+v_T(E_{max}-E_T).
\end{aligned}
$$

The optimizer solves either risk-neutral SAA,

$$
\min_x \frac{1}{N}\sum_{s=1}^{N}J_s(x),
$$

or the risk-aware problem

$$
\min_x
\left[
\frac{1}{N}\sum_{s=1}^{N}J_s(x)
+\lambda\operatorname{CVaR}_{0.95}(J_s(x))
\right].
$$

The operating window is 06:00--21:00. Battery state, efficiency, mutually
exclusive charge/discharge mode, dynamic charging, terminal value, PV
curtailment and the Pforzheim 2025 variable grid tariff are identical to the
deterministic V4 replay.

## Experiment

- calibration information: 210 rolling-origin days from 2024;
- factual holdout: 261 days from 1 January through 30 September 2025;
- scenarios: 500 per day;
- generators: raw quantile regression and hourly CQR, each with one empirical
  48-dimensional copula path per scenario;
- policies: SAA and CVaR weights 0.05, 0.10, 0.25 and 0.50;
- context strategies: Rule-based, deterministic point day-ahead and day-ahead
  Oracle;
- MPC: excluded.

## Economic result

| Strategy | Total cost, EUR | Change vs point DA, EUR | Realized CVaR95, EUR/day | Maximum day, EUR |
|---|---:|---:|---:|---:|
| Day-ahead Oracle | 7,431.97 | -115.41 | 48.821 | 59.295 |
| CQR SAA | **7,538.92** | **-8.46** | 49.766 | 66.014 |
| Raw SAA | 7,539.50 | -7.88 | 49.766 | 66.014 |
| CQR CVaR, $\lambda=0.05$ | 7,540.37 | -7.01 | 49.771 | 66.014 |
| Raw CVaR, $\lambda=0.05$ | 7,541.65 | -5.73 | 49.766 | 66.014 |
| Raw CVaR, $\lambda=0.10$ | 7,543.14 | -4.24 | 49.765 | 66.047 |
| CQR CVaR, $\lambda=0.10$ | 7,544.95 | -2.43 | 49.777 | 66.032 |
| Raw CVaR, $\lambda=0.25$ | 7,546.01 | -1.37 | 49.799 | 66.047 |
| Point day-ahead | 7,547.38 | 0.00 | **49.490** | **63.195** |
| Raw CVaR, $\lambda=0.50$ | 7,550.41 | +3.03 | 49.848 | 66.005 |
| CQR CVaR, $\lambda=0.25$ | 7,552.19 | +4.81 | 49.828 | 66.043 |
| CQR CVaR, $\lambda=0.50$ | 7,564.57 | +17.19 | 49.888 | 66.044 |
| Rule-based | 7,992.53 | +445.15 | 53.087 | 71.494 |

CQR SAA is the lowest-cost probabilistic policy, but its EUR 8.46 saving has a
paired seven-day block-bootstrap 95% interval of EUR -2.70 to EUR 21.08. The
effect is therefore small and not statistically stable. Raw SAA is only EUR
0.58 more expensive than CQR SAA.

No probabilistic variant reduces the realised CVaR95. CQR SAA increases it by
EUR 0.276/day, and stronger CVaR weights make the realised tail progressively
worse. At $\lambda=0.50$, CQR costs EUR 17.19 more than point day-ahead and
increases realised CVaR95 by EUR 0.398/day. Its block-bootstrap interval for
the CVaR reduction is entirely negative: EUR -0.905 to EUR -0.076/day.

The economic conclusion is that conditional quantiles and conformal coverage
are technically valid but do not identify the realised joint tail well enough
for CVaR control. Better marginal coverage alone does not imply better
decisions.

## Tail failure example

On 15 January 2025, factual day-ahead price exceeded the CQR P90 in 12 of 16
operating hours. The negative intraday spread was below its CQR P10 for eight
consecutive hours. This joint regime was absent from the sampled distribution.
CQR SAA moved 15 kWh of production from factual hour 21, priced at
150.87 EUR/MWh, to hour 9, priced at 315.02 EUR/MWh, and lost EUR 2.82 against
point day-ahead on that day. CVaR cannot protect against a regime missing from
the scenario distribution.

## Physical audit

All checks pass:

- every strategy contains the same 261 delivery days;
- production load is 200 kWh/day to numerical tolerance;
- SoC remains inside 0--44.16 kWh;
- charge and discharge never overlap;
- factual curtailment lies between zero and factual PV;
- Oracle deviation is exactly zero;
- every ledger reconciles to total cost within $2.2\times10^{-14}$ EUR;
- Rule-based, point day-ahead and Oracle reproduce the deterministic V4 totals
  within EUR 0.01.

## Reproduction

```bash
energy run-quantile-spread-economic-backtest \
  --project-root . \
  --quantile-artifact-name quantile_spread_copula_v2 \
  --generators raw cqr \
  --n-scenarios 500 \
  --risk-weights 0.05 0.10 0.25 0.50
```

Outputs are stored in
`artifacts/experiments/quantile_spread_economic_backtest_v2/`:

- `daily_results.csv`;
- `hourly_decisions.parquet`;
- `scenario_objectives.csv`;
- `coverage.csv`;
- `summary.json`;
- `experiment_config.json`.

The next experiment should keep this optimizer and factual ledger unchanged,
replace only the scenario generator with a centred joint residual bootstrap,
and compare it against the same point day-ahead baseline.

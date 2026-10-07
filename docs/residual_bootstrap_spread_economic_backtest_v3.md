# Three-target residual bootstrap and economic backtest

## Purpose

This experiment is the second probabilistic day-ahead approach. It keeps the
same point forecasts and the same V4 economic formulation as the quantile plus
empirical-copula experiment, but generates uncertainty from joint historical
forecast errors. MPC is intentionally excluded: every physical decision is
made once day-ahead and is fixed when the 2025 facts are settled.

The three uncertain hourly quantities are:

- photovoltaic generation (G_h);
- day-ahead price (P_h^{DA});
- intraday spread (S_h=P_h^{ID}-P_h^{DA}).

## Residual construction

The 2024 residual library is built with rolling-origin predictions. For each
eligible historical day (d) and operating hour (h\in\{6,\ldots,21\}),

$$
\varepsilon_{d,h}^{G}=G_{d,h}-\widehat G_{d,h},
$$

$$
\varepsilon_{d,h}^{DA}=P_{d,h}^{DA}-\widehat P_{d,h}^{DA},
$$

$$
\varepsilon_{d,h}^{S}=S_{d,h}-\widehat S_{d,h}.
$$

One sampled library day supplies all 48 residual coordinates: three targets
times sixteen hours. Thus a scenario retains cross-hour dependence and the
dependence between PV, day-ahead price and spread.

For a sampled historical day (D_s), scenario (s) is

$$
G_{s,h}=\operatorname{clip}\!\left(
\widehat G_h+\varepsilon_{D_s,h}^{G},0,G^{\max}
\right),
$$

$$
P_{s,h}^{DA}=\widehat P_h^{DA}+\varepsilon_{D_s,h}^{DA},
\qquad
S_{s,h}=\widehat S_h+\varepsilon_{D_s,h}^{S},
$$

$$
P_{s,h}^{ID}=P_{s,h}^{DA}+S_{s,h}.
$$

The `raw` generator uses the historical errors directly. The `centered`
generator first subtracts the eligible library mean separately for every
target and hour. Centering makes the scenario mean approximately equal to the
point forecast; the raw variant also retains any persistent rolling-origin
bias.

The point PV and day-ahead price forecasts are checked for exact equality with
the frozen deterministic replay inputs. The bootstrap code trains the price
model on all 24 daily hours, exactly as the packaged point model does, and only
then selects hours 06:00--21:00 for the joint scenario. This matters: training
the same CatBoost only on the operating window produces a different model.

## Forecast audit

The untouched test period is 1 January through 30 September 2025: 261 complete
days and 4,176 target hours. The table reports an 80% central interval and
ensemble CRPS.

| Generator | Target | Point MAE | Point RMSE | P10--P90 coverage | Mean width | CRPS |
|---|---|---:|---:|---:|---:|---:|
| raw | PV, kWh | 0.548 | 1.019 | 81.35% | 1.642 kWh | 0.405 |
| centered | PV, kWh | 0.548 | 1.019 | 84.10% | 1.695 kWh | 0.406 |
| raw | DA price, EUR/MWh | 20.822 | 31.905 | 78.88% | 60.934 | 15.817 |
| centered | DA price, EUR/MWh | 20.822 | 31.905 | 79.41% | 60.934 | 15.559 |
| raw | ID spread, EUR/MWh | 15.204 | 24.382 | 79.50% | 46.316 | 11.541 |
| centered | ID spread, EUR/MWh | 15.204 | 24.382 | 78.98% | 46.316 | 12.028 |

The residual intervals are much closer to their nominal 80% coverage than the
conformally widened quantile intervals. Quantile CQR has better CRPS for PV and
day-ahead price; raw residual bootstrap has the better spread CRPS.

## Optimization contract

All scenarios share one load, charge, discharge and curtailment-fraction
schedule. The submitted day-ahead position is tied to the frozen point PV:

$$
q_h^{DA}=L_h+C_h-D_h-(1-u_h)\widehat G_h.
$$

This prevents deliberate day-ahead/intraday speculation. Scenario uncertainty
only changes the physical deviation and its intraday settlement. The objective
is either sample-average approximation,

$$
\min_x \frac{1}{N}\sum_{s=1}^{N} C_s(x),
$$

or the risk-augmented objective

$$
\min_x \left[
\frac{1}{N}\sum_{s=1}^{N} C_s(x)
+\lambda\operatorname{CVaR}_{0.95}(C_s(x))
\right].
$$

The physical and economic assumptions are identical to the final V4 replay:
200 kWh daily flexible load, 44.16 kWh usable battery, 5 kWh hourly charge and
discharge limits, efficiency, degradation, terminal recharge liability,
Pforzheim variable grid charges and PV curtailment.

## 2025 economic results

Each stochastic strategy uses 500 scenarios per delivery day.

| Strategy | Total cost | Saving vs point DA | 95% block-bootstrap CI | Realised CVaR95 per day | Maximum day |
|---|---:|---:|---:|---:|---:|
| Oracle | EUR 7,431.97 | EUR 115.41 | -- | EUR 48.82 | EUR 59.30 |
| residual raw SAA | **EUR 7,535.12** | **EUR 12.26** | **[EUR 6.12, EUR 19.47]** | EUR 49.44 | EUR 63.54 |
| residual centered SAA | EUR 7,536.67 | EUR 10.71 | [EUR 7.06, EUR 14.66] | EUR 49.51 | EUR 63.60 |
| centered CVaR, lambda 0.05 | EUR 7,540.35 | EUR 7.03 | [EUR 1.32, EUR 13.38] | EUR 49.52 | EUR 63.53 |
| centered CVaR, lambda 0.10 | EUR 7,547.08 | EUR 0.31 | [EUR -7.18, EUR 8.62] | EUR 49.52 | EUR 63.51 |
| point day-ahead | EUR 7,547.38 | -- | -- | EUR 49.49 | EUR 63.20 |
| centered CVaR, lambda 0.25 | EUR 7,578.67 | EUR -31.29 | [EUR -43.03, EUR -16.85] | EUR 49.53 | EUR 63.44 |
| centered CVaR, lambda 0.50 | EUR 7,637.96 | EUR -90.58 | [EUR -113.28, EUR -66.37] | EUR 49.49 | EUR 63.36 |
| rule-based | EUR 7,992.53 | EUR -445.15 | -- | EUR 53.09 | EUR 71.49 |

Raw residual SAA closes 10.62% of the remaining point-to-Oracle gap. Centered
SAA closes 9.28%. For comparison, quantile CQR SAA costs EUR 7,538.92, saves
EUR 8.46 and closes 7.33% of the same gap.

Raw SAA reduces realised CVaR95 by EUR 0.053 per day relative to point
day-ahead, but its 95% interval is [-EUR 0.051, EUR 0.211], so the tail benefit
is not statistically established. Increasing the CVaR weight does not produce
a reliable out-of-sample tail improvement. At lambda 0.25 and 0.50 it clearly
damages total cost. The practical champion of this experiment is therefore raw
SAA, with centered SAA as the cleaner zero-bias sensitivity check.

## Physical audit

All nine strategies have 261 delivery days. The maximum daily load error is
(2.02\times10^{-11}) kWh; the maximum SoC dynamics error is
(6.30\times10^{-11}) kWh. There are no SoC-bound violations, no simultaneous
charge/discharge rows and no curtailment violations. The maximum factual
energy-balance error is (3.55\times10^{-15}) kWh and the maximum ledger
reconciliation error is (1.42\times10^{-14}) EUR. Rule-based, point
day-ahead and Oracle daily costs exactly reproduce the quantile backtest.

## Reproduction

First rebuild the current point-aligned rolling residual library:

```bash
energy run-residual-bootstrap-experiment \
  --project-root . \
  --artifact-name joint_residual_bootstrap_current_v3
```

Join the spread residual and audit raw and centered scenarios:

```bash
energy run-residual-bootstrap-spread-experiment \
  --project-root . \
  --residual-artifact-name joint_residual_bootstrap_current_v3 \
  --artifact-name joint_residual_bootstrap_spread_current_v3
```

Run the full economic replay:

```bash
energy run-residual-bootstrap-spread-economic-backtest \
  --project-root . \
  --residual-artifact-name joint_residual_bootstrap_spread_current_v3 \
  --n-scenarios 500 \
  --artifact-name residual_bootstrap_spread_economic_backtest_v3
```

The forecast artifacts are under
`artifacts/experiments/joint_residual_bootstrap_spread_current_v3`. The
economic artifacts, comparison table and physical audit are under
`artifacts/experiments/residual_bootstrap_spread_economic_backtest_v3`.

## Seasonal conditional extension

A follow-up experiment replaces uniform residual-day sampling with a circular
day-of-year kernel. For a target delivery date \(d\), historical day \(j\)
receives the weight

$$
w_j(d)=(1-\rho)
\frac{\exp\left[-\Delta_{\mathrm{circ}}(d,j)^2/(2b^2)\right]}
{\sum_k \exp\left[-\Delta_{\mathrm{circ}}(d,k)^2/(2b^2)\right]}
+\rho\frac{1}{N},
$$

where \(b=25\) days, \(\rho=0.15\), and
\(\Delta_{\mathrm{circ}}\) is circular calendar distance. The global
mixture preserves a small probability for unusual residual days. A sampled
day still supplies the complete paired PV, day-ahead-price and intraday-spread
trajectory. PV errors are sampled after local scale standardisation and then
mapped to the target day's point-forecast scale. Price and spread residuals
remain in their physical units.

The parameters above were fixed before the 2025 economic replay. They were
not selected on realised 2025 cost. A radiation-similarity kernel remains a
separate future experiment.

### Scenario audit

The seasonal generator has mean effective sample size 64.81 days (median
54.73, minimum 25.00) versus 210 days for uniform sampling. On 4,176 test
hours it produces the following central 80% intervals:

| Target | Coverage | Mean width | CRPS |
|---|---:|---:|---:|
| PV, kWh | 78.26% | 1.717 kWh | 0.4064 |
| DA price, EUR/MWh | 80.36% | 64.861 | 15.4604 |
| ID spread, EUR/MWh | 78.83% | 49.992 | 12.6573 |

Seasonal sampling improves day-ahead-price calibration and slightly improves
its CRPS relative to the centred uniform generator. It does not improve PV
CRPS and materially worsens spread CRPS. The result is therefore mixed rather
than a general forecasting improvement.

### Economic replay

| Strategy | Total cost | Saving vs point DA | 95% block-bootstrap CI | Realised CVaR95 per day |
|---|---:|---:|---:|---:|
| seasonal SAA | EUR 7,538.57 | **EUR 8.81** | **[EUR 5.31, EUR 12.87]** | EUR 49.49 |
| seasonal CVaR, lambda 0.05 | EUR 7,543.61 | EUR 3.77 | [EUR -0.96, EUR 10.54] | EUR 49.44 |
| point day-ahead | EUR 7,547.38 | -- | -- | EUR 49.49 |
| seasonal CVaR, lambda 0.10 | EUR 7,551.84 | EUR -4.46 | [EUR -11.86, EUR 5.77] | EUR 49.45 |

Seasonal SAA closes 7.63% of the remaining point-to-Oracle gap and is cheaper
than point day-ahead on 59.77% of days. Its mean-cost improvement is
statistically supported, but it does not reduce realised CVaR95. The small
CVaR95 reduction for \(\lambda=0.05\) is not statistically established, and
\(\lambda=0.10\) raises total cost.

The rolling-origin residual library contains 210 complete days from 24 May
through 31 December 2024. Consequently, January through early May 2025 lack
nearby observations on both sides of the circular seasonal window. On the
well-supported 24 May--30 September 2025 subset, seasonal SAA saves EUR 3.76
and centred uniform SAA saves EUR 3.74 over 129 days: they are effectively
equal. This supports retaining raw uniform SAA as the overall benchmark and
using the seasonal generator as a robustness check until a full-year residual
library is available.

Reproduce the scenario and economic audits with:

```bash
energy run-residual-bootstrap-spread-experiment \
  --project-root . \
  --artifact-name joint_residual_bootstrap_spread_seasonal_v4

energy run-residual-bootstrap-spread-economic-backtest \
  --project-root . \
  --residual-artifact-name joint_residual_bootstrap_spread_seasonal_v4 \
  --generators seasonal \
  --risk-generators seasonal \
  --risk-weights 0.05 0.10 \
  --n-scenarios 500 \
  --artifact-name residual_bootstrap_spread_seasonal_economic_v4
```

The saved outputs are under
`artifacts/experiments/joint_residual_bootstrap_spread_seasonal_v4` and
`artifacts/experiments/residual_bootstrap_spread_seasonal_economic_v4`.

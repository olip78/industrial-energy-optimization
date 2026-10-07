# Economic replay V3.1: grid tariff and terminal battery value

> **Archived result.** The numerical tables and figures in this document
> describe the previous V3.1 replay. They have not been recomputed after the
> final formulation review. The current V4 code changes four material rules:
> MPC reoptimizes load, charge, discharge and curtailment; PV curtailment is an
> explicit physical decision; the rule baseline can recharge only from factual
> PV headroom and curtails residual PV at a negative known day-ahead price; and
> the fixed daily grid charge is excluded. The next approved simulation writes
> to `economic_backtest_v4_final_formulation`, so these archived artifacts are
> preserved.

The current mathematical contract is defined in
[`design_document_draft.md`](design_document_draft.md), sections 6 and 7.

## Archived V3.1 specification and result

V3.1 keeps the physical grid tariff and moves every battery action into the
day-ahead decision. It also values the terminal state of charge through the
expected cost of restoring the battery during the following night. The market
ledger and the grid ledger remain deliberately
separate: day-ahead and intraday positions are financial settlements, while
network charges apply to positive **physical withdrawal at the meter**.

## Reference customer and tariff

The modelled site consumes 200 kWh per operating day, has a maximum flexible
load of 20 kWh per hour, a 10 kWp PV installation and a 44.16 kWh battery with
a 5 kWh/h charge/discharge limit. Its annual consumption is below 100 MWh and
its maximum modelled withdrawal cannot reach 30 kW. V3 therefore uses the 2025
SWP low-voltage standard-load-profile tariff without quarter-hour demand
metering rather than an RLM demand tariff.

All values below exclude VAT. VAT is excluded because the fictional industrial
customer is assumed able to deduct input VAT.

| Component | Day 06:00--22:00 | Night 22:00--06:00 | Source/assumption |
|---|---:|---:|---|
| SWP network use | 5.490 ct/kWh | 2.750 ct/kWh | SWP 2025 SLP low-voltage tariff |
| KWKG levy | 0.277 ct/kWh | 0.277 ct/kWh | nationwide 2025 rate |
| special grid-use levy | 1.558 ct/kWh | 1.558 ct/kWh | nationwide 2025 rate |
| offshore levy | 0.816 ct/kWh | 0.816 ct/kWh | nationwide 2025 rate |
| Pforzheim concession fee | 1.990 ct/kWh | 1.990 ct/kWh | tariff-customer rate for cities up to 500,000 inhabitants |
| electricity tax after producing-industry relief | 0.050 ct/kWh | 0.050 ct/kWh | section 9b StromStG assumption |
| **Total variable import charge** | **10.181 ct/kWh** | **7.441 ct/kWh** | sum of the rows above |

The fixed charge is EUR 80.00/year for network use plus EUR 29.21/year for a
bidirectional meter, or EUR 0.2992 per calendar day. It is included in the
realised ledger, but it cannot change dispatch because it is constant across
all feasible schedules.

Official references:

- [SWP 2025 electricity network price sheet](https://www.stadtwerke-pforzheim.de/fileadmin/user_upload/Downloads/Netze/Netznutzung_Strom/SWP_Strom_endgueltiges_Preisblatt_2025_neu.pdf)
- [KWKG levy 2025](https://www.netztransparenz.de/de-de/Erneuerbare-Energien-und-Umlagen/KWKG/KWKG-Umlage/KWKG-Umlagen-%C3%9Cbersicht/KWKG-Umlage-2025)
- [Special grid-use levy 2025](https://www.netztransparenz.de/de-de/Erneuerbare-Energien-und-Umlagen/Sonstige-Umlagen/Aufschlag-f%C3%BCr-besondere-Netznutzung-19-StromNEV-Umlage/%C3%9Cbersicht-Aufschlag-f%C3%BCr-besondere-Netznutzung/Aufschlag-f%C3%BCr-besondere-Netznutzung-2025)
- [Offshore levy 2025](https://www.netztransparenz.de/de-de/Erneuerbare-Energien-und-Umlagen/Sonstige-Umlagen/Offshore-Netzumlage/Offshore-Netzumlagen-%C3%9Cbersicht/Offshore-Netzumlage-2025)
- [Concession Fee Ordinance, KAV](https://www.gesetze-im-internet.de/kav/BJNR000120992.html)
- [Producing-industry electricity-tax relief, section 9b StromStG](https://www.gesetze-im-internet.de/stromstg/__9b.html)
- [Storage network-charge exemption, section 118(6) EnWG](https://www.gesetze-im-internet.de/enwg_2005/__118.html)

The producing-industry tax relief is an explicit scenario assumption and
requires a successful application. Without it, the electricity-tax component
would be 2.05 rather than 0.05 ct/kWh. This is exposed as a tariff parameter so
the sensitivity can be run without changing the optimizer.

The storage exemption in section 118(6) EnWG is not applied. It concerns energy
withdrawn for storage and later returned to the same grid. The project battery
primarily supplies the site's own load and therefore does not meet that simple
charge-and-reinject pattern.

## Physical and financial balances

For hour $h$, let:

- $L_h$ be production load;
- $C_h$ and $D_h$ be battery charge and discharge at the site bus;
- $G_h$ be PV production;
- $q_h^{DA}$ be the fixed day-ahead position;
- $P_h^{DA}$ and $P_h^{ID}$ be realised day-ahead and intraday prices.

The physical exchange at the meter is

$$
g_h = L_h + C_h - G_h - D_h.
$$

Import and export are separate non-negative quantities:

$$
I_h = \max(g_h,0),
\qquad
X_h = \max(-g_h,0).
$$

The intraday deviation is a financial quantity:

$$
\Delta_h = g_h-q_h^{DA}.
$$

This distinction matters in MPC. A large day-ahead purchase can make
$\Delta_h$ negative while the site still physically imports energy. Grid
charges therefore use $I_h$, never $\Delta_h$, and export in another hour does
not cancel a previous hour's charged withdrawal.

Let the daytime variable import tariff be

$$
t_h^{\mathrm{imp}}
=t_h^{\mathrm{net}}+t^{\mathrm{KWKG}}+t^{\mathrm{special}}
+t^{\mathrm{offshore}}+t^{\mathrm{concession}}+t^{\mathrm{tax}}.
$$

The realised operating cost is

$$
\begin{aligned}
C_{\mathrm{day}}
={}&\sum_h \frac{P_h^{DA}}{1000}q_h^{DA}
+\sum_h \frac{P_h^{ID}}{1000}\Delta_h
+\sum_h t_h^{\mathrm{imp}} I_h
+\sum_h t_h^{\mathrm{export}}X_h\\
&+c_{\mathrm{deg}}\sum_hD_h
+v_T(E_{\max}-E_T)
+C^{\mathrm{fixed}}.
\end{aligned}
$$

V3 sets
$t_h^{\mathrm{export}}=0$: physical export earns the wholesale market price but
does not receive a refund of import network charges. Direct-marketing fees,
bid-ask spread and balancing-group fees remain outside the current data model.

## Terminal battery value

For each completed night, the raw recharge proxy is the mean day-ahead price
over nine local labels: 22:00, 23:00 and 00:00--06:00. Equal weighting
represents nine 5 kWh blocks, or 45 kWh in total, which matches the 5 kWh/h
charge limit and approximately fills the 44.16 kWh battery. Every raw nightly
mean is floored at zero, then the latest 14 completed nights are averaged:

$$
\bar P_{\mathrm{night},14}(D)
=\frac{1}{14}\sum_{j=1}^{14}
\max(P_{D-j}^{\mathrm{night}},0).
$$

This value is causal: the newest included night ends before the day-ahead
decision. The value of one stored kWh at the end of the operating window is

$$
v_T(D)=\frac{\bar P_{\mathrm{night},14}(D)/1000
+t_{\mathrm{night}}^{\mathrm{imp}}}{\eta_{\mathrm{charge}}}.
$$

The ledger charges the expected cost of filling the terminal deficit:

$$
C_T=v_T(D)(E_{\max}-E_T).
$$

The optimizer therefore has no hard terminal-SoC constraint. It may finish
below full, but pays the causal overnight replacement liability. Battery
degradation is charged per delivered discharge kWh. The old per-discharge
night-energy term is removed from the objective and ledger, because keeping it
alongside $C_T$ would count overnight replacement twice.

Charging is feasible in every operating hour. Grid charging pays the current
day-ahead price plus the daytime physical-import tariff. Charging from PV
surplus pays no import tariff while the meter remains at zero or export; its
cost is the forgone export revenue. If charge exceeds the surplus, the import
auxiliary variable automatically charges the tariff only on the imported part.
The model can also perform an additional profitable day-ahead cycle when the
within-day price spread covers import charges, losses and degradation.

## Optimizer implementation

The deterministic day-ahead, stochastic CQR-copula/CVaR and day-ahead Oracle
objectives all contain the same physical-import and terminal-value terms. The
piecewise maximum is represented with non-negative auxiliary variables and
linear constraints, so the deterministic problem remains a MILP and the
scenario optimizer remains a two-stage scenario MILP with a common physical
schedule.

The day-ahead optimizer fixes the complete charge and discharge trajectory.
Hourly MPC then reallocates only the remaining production load using refreshed
PV and intraday-price forecasts. It cannot revise charge or discharge. This
keeps all battery market manipulation in day-ahead and prevents intraday
battery speculation.

The rule-based strategy does not know the daily PV trajectory. Its battery
threshold therefore treats the daytime tariff as an optimistic avoided cost;
the common realised ledger subsequently charges only actual physical import.
This keeps the baseline executable and prevents it from being weakened by the
new accounting convention.

## Reproduction and artifacts

```bash
energy run-dynamic-charge-economic-backtest \
  --project-root . \
  --artifact-name economic_backtest_v3_grid_tariff \
  --grid-tariff-preset pforzheim-slp-2025

python scripts/plot_dynamic_charge_weekly_average.py \
  --project-root . \
  --artifact-name economic_backtest_v3_grid_tariff
```

Use `--grid-tariff-preset none` to reproduce the market-only counterfactual.
The experiment configuration stores every tariff component. Daily and hourly
artifacts additionally store physical import/export and the separated network,
levy, concession, tax, metering and battery-cost components.

## 2025 result

The frozen replay contains 261 complete delivery days from January through
September 2025.

| Strategy | Total cost, EUR | Mean/day, EUR | Realised CVaR95, EUR/day | Physical import, kWh | Charge, kWh | Discharge, kWh |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based | 7,947.00 | 30.448 | 53.386 | 43,091.9 | 0.0 | 2,174.7 |
| Day-ahead only | 7,625.47 | 29.216 | 49.789 | 41,059.1 | 1,860.2 | 6,286.6 |
| CQR-copula + CVaR $\lambda=0.10$ | 7,623.34 | 29.208 | 50.082 | **40,682.5** | 1,553.5 | 6,181.3 |
| Deterministic + load-only MPC | **7,597.39** | **29.109** | **49.580** | 40,995.2 | 1,860.2 | 6,286.6 |
| Day-ahead Oracle | 7,511.63 | 28.780 | 49.120 | 41,008.3 | 2,237.7 | 6,368.8 |

The Rule-based--Oracle opportunity is EUR 435.37. Day-ahead optimization
captures 73.9% of it; CQR-copula/CVaR captures 74.3%; deterministic planning
with load-only MPC captures 80.3%. MPC saves another EUR 28.09 after the
day-ahead plan and finishes EUR 85.76 above the non-speculative day-ahead
Oracle.

The selected stochastic policy is EUR 2.14 cheaper than deterministic
day-ahead, but its realised CVaR95 is EUR 0.293/day worse. It remains a small
mean-cost improvement rather than a demonstrated tail-risk improvement.

For deterministic MPC, the EUR 4,600.47 regulated-cost subtotal consists of
EUR 348.66 of night grid charges in the terminal recharge liability, EUR
2,250.64 of daytime network use, EUR 1,086.78 of nationwide levies, EUR 815.80
of concession fees, EUR 20.50 of electricity tax and EUR 78.09 of fixed
charges. The complete battery subtotal is EUR 1,098.26: EUR 783.93 of terminal
recharge liability and EUR 314.33 of degradation.

Day-ahead-only and deterministic MPC both charge 1,860.18 kWh and discharge
6,286.64 kWh over the replay. Their hourly battery schedules are identical to
solver tolerance; the EUR 28.09 MPC increment comes only from reallocation of
production load and its intraday settlement. This is the intended consequence
of fixing all battery actions in day-ahead.

The reference installation has no realised hour with PV above the production
load during a scheduled charge, so the 2025 replay contains no observable
PV-surplus charging. All optimised charge is classified as grid-supplied under
the project's load-first physical convention. The optimizer nevertheless
supports PV-surplus charging, and the synthetic physical tests verify that it
does not incur an import tariff until charge pushes the meter into net import.

The physical audit passes: ledger components reconcile with total cost to
machine precision; terminal recharge plus degradation equals the battery
subtotal; import and export equal the positive and negative parts of meter
flow; SoC remains within numerical tolerance of 0--44.16 kWh; no row charges
and discharges simultaneously; deterministic MPC preserves every day-ahead
battery action; and the day-ahead Oracle has exactly zero intraday deviation.

![Seven-day mean economic result by strategy](../artifacts/experiments/economic_backtest_v3_grid_tariff/economic_strategy_7d_average.png)

![Cost component breakdown](../artifacts/experiments/economic_backtest_v3_grid_tariff/economic_cost_component_breakdown.png)

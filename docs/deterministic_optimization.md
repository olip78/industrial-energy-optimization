# Deterministic economic decision core

The first optimisation component implements the agreed hourly linear programs:

1. `solve_day_ahead` builds the initial 24-hour flexible-load, battery-discharge
   and net market-position plan;
2. `solve_mpc` replans the remaining horizon against that fixed day-ahead
   position; only its first hourly action is executed;
3. `evaluate_actual_day` calculates the ex-post V1 ledger from realised PV,
   load, battery discharge and intraday settlement price.

All energies are kWh per one-hour interval. Input market prices are EUR/MWh and
are converted to EUR/kWh inside the optimiser and ledger.

## Physical and economic contracts

The V1 battery starts each daytime window with an externally set energy budget.
It cannot charge during the day, and its discharge is bounded by both a total
energy budget and an hourly maximum. The full delivered-energy cost is:

```text
night reference price / (charge efficiency × discharge efficiency)
+ degradation cost
```

The optimiser will therefore leave the battery unused when its full marginal
cost exceeds the hourly energy value. The day-ahead position is derived after
the physical decision:

```text
q_DA = load - PV forecast - battery discharge
```

MPC does not change `q_DA`. It forms expected and later actual deviations from
that fixed position, while reallocating only the remaining production load and
available battery energy.

## Economic replay adapter

`energy run-economic-backtest --project-root .` connects the array-level core
to frozen historical forecast artifacts. It first builds the day-ahead position,
then re-solves the full remaining MPC horizon before every delivery hour and
executes only the first action. Both the rule-based policy and the deterministic
policy are settled by the same `evaluate_actual_day` ledger.

The fixed reference site, temporal contract, output files and the initial
January–September 2025 result are documented in
[economic_backtest_v1.md](economic_backtest_v1.md).

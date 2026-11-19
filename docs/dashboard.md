# Energy operations dashboard

## Purpose

The dashboard is the read-only operational surface for the historical replay.
It separates the operator's three questions instead of mixing today and
tomorrow in one chart:

1. **Live Operations** — what was nominated day-ahead, what MPC committed, and
   what has actually happened today;
2. **Day-Ahead Plan** — what the system proposes for tomorrow and which
   forecasts, constraints and costs explain that plan;
3. **Monitoring** — whether data, models, decisions and retraining remain
   healthy in recent operational history.

All UI copy is English. The first implementation is a static frontend backed
by a versioned replay JSON bundle. The future service will implement the same
data contract over HTTP; chart and interaction code must not depend on local
Parquet paths.

## Shared replay controls

Every view shows the virtual delivery date, `cycle_id`, active model release,
data freshness and a clear `Historical replay` badge. Demo-only controls play
the replay by whole days, advance one day, change playback speed and reset the
July--September loop. Each day is displayed at a fixed 13:00 operational
snapshot. At the end of 30 September the clock returns to 1 July and
increments the cycle.

Actual observations are revealed only through the selected virtual hour.
Future values remain hidden even though the browser bundle contains the frozen
historical replay.

## Live Operations

The main charts are:

- **Flexible Load Schedule:** original day-ahead plan, past executed MPC
  setpoints and the current MPC outlook for the remaining hours;
- **Market & Grid Position:** day-ahead nomination, past executed physical
  exchange and the current grid outlook for the remaining hours;
- **Battery:** charge/discharge power and SoC on one aligned chart;
- **PV Forecast vs Actual:** calibrated P10--P90 band, P50 forecast and revealed
  actual generation.

The screen also exposes the current action, remaining production, SoC, cost so
far, expected end-of-day cost and the next MPC run. `Main Changes Since
Day-Ahead` reports observable input and schedule changes; it does not claim
formal causal attribution.

The economy card compares cost with a rolling median of prior comparable
weekdays. A normalized production-cost row is intentionally absent because the
demo uses the same 200 kWh target every day, making it only a rescaling of the
daily cost. Rule-based results are intentionally absent. Oracle cost is unavailable during
the operating day and therefore appears only as a retrospective diagnostic in
Monitoring.

## Day-Ahead Plan

The tomorrow view includes:

- grouped hourly blocks for flexible load, forecast PV and battery net action,
  plus the cumulative 200 kWh target;
- day-ahead market nomination aligned with the price forecast;
- PV and price uncertainty bands;
- battery charge/discharge and SoC schedule;
- expected cost and cost-component breakdown;
- plan timestamp, data cutoff, release ID, solver status, assumptions and
  operational risks.

Price is not presented as an isolated headline chart: it shares the time axis
with the nominated position so the economic reason for moving consumption is
visible.

## Monitoring

Monitoring has three internal modes without adding more top-level tabs:

- **Data:** freshness, health score, missingness/incidents and drift;
- **Models:** selectable rolling metrics for all four deployable forecasts,
  latest-versus-previous release validation metrics and release transitions;
- **Decisions:** a rolling day-ahead cost gap to the retrospective Oracle,
  plus a separate rolling incremental MPC-cost effect measured against the
  day-ahead-only policy.

The retraining timeline records pipeline completion, candidate rejection,
promotion and rollback reasons. Cycle resets are explicit. Multiple repeated
July--September cycles are never silently concatenated on one calendar axis.

The two summary cards expose current data quality and all four model metrics
for the latest promoted release against the previous promoted release. A
release validation metric is the rolling 14-day MAE ending on the promotion
date; the release-over-release delta is `latest - previous`, so negative is an
improvement. Decision economics is measured in EUR per delivery day. The first
chart shows `day-ahead-only cost - day-ahead Oracle cost`; positive values are
the opportunity left by the direct plan. The second shows
`deterministic MPC cost - day-ahead-only cost`; negative values mean MPC was
cheaper. Both charts show the unsmoothed daily result as a light line and a
trailing 14-delivery-day mean as the primary line over the complete
January--September test period. They use a common symmetric scale, with zero
emphasized; the positive half of the MPC panel is lightly shaded as the loss
zone. The MPC-effect chart is shorter because it is an incremental-effect
diagnostic.

The complete March--September diagnostic history is displayed as one continuous
operating record. The bundle can still use earlier rows for model training,
reference construction and pseudo-release validation, but the operator UI does
not expose an engineering phase boundary or a synthetic `loop start`. The demo
opens on 29 September so the current point has nearly seven months of visible
history while the next day remains available to the Day-Ahead view.

## Data contract and honesty

`scripts/build_dashboard_demo.py` materializes `dashboard/data/demo.json` from
the frozen backtest artifacts. The bundle contains real Q3 decisions, PV,
prices, battery trajectories, economics and rolling model errors. The project
does not have equipment telemetry. The dashboard therefore treats historical
committed MPC setpoints as executed controls and does not fabricate an `Actual
Load` or meter series. Data-health history is replay-derived; the 23 July
common all-zero PV incident comes from the audited source metadata.

Operational UI code reads the bundle through a provider boundary. The service
phase will replace the file provider with endpoints such as:

```text
GET /v1/dashboard/state?as_of=...
GET /v1/dashboard/day-ahead?delivery_date=...
GET /v1/dashboard/monitoring?start=...&end=...&cycle_id=...
```

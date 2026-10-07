# ECMWF IFS weather snapshots for intraday PV control

This V1.1 data layer adds weather forecasts that were available during a
delivery day. It is separate from the fixed-lead DWD ICON tables used by the
day-ahead model, so the original V1 results remain reproducible.

## Why a separate archive is needed

The D-1 PV model needs one weather forecast available before the auction. Its
fixed 24/48-hour ICON features are appropriate for that purpose. The MPC PV
residual model has a different information contract: at decision time `tau`,
it may use a forecast issued after the D-1 decision but must not use weather
observed after `tau` or a model run still being computed.

Open-Meteo's ECMWF IFS HRES Single Runs archive preserves individual forecast
runs. For every `(decision hour, future target hour)` the snapshot builder
selects the latest run satisfying:

```text
weather_snapshot_available_at_utc <= as_of_utc < valid_time_utc
```

The provider documents that global ECMWF output is normally distributed 4–6
hours after initialisation. The pipeline uses a conservative fixed six-hour
publication delay. Thus a 00:00 UTC run first becomes eligible at 06:00 UTC.

## Scope and storage

- Model: ECMWF IFS HRES, queried at the existing provisional Pforzheim proxy
  coordinate (48.89, 8.70).
- Variables: 2 m temperature, GHI, direct and diffuse radiation, 10 m wind,
  and cloud cover.
- Training begins on 14 March 2024, the first date of the available individual
  IFS-run archive. This deliberately removes January and February 2024 from
  the V1.1 residual-model experiment.
- Raw run responses are saved atomically under
  `data/raw/weather_forecasts_ecmwf_ifs_single_runs/runs/`; a retry reuses
  completed files.
- The normalised run table is
  `data/processed/weather_forecast_hourly_ecmwf_ifs_single_runs.parquet`.
- The model-ready decision/target tables are
  `data/features/mpc_pv_weather_ifs_snapshots/`.

The historic archive has an empirical coverage difference: its 2024 data is
available as the 00:00 UTC run, while 2025 exposes all four 00/06/12/18 UTC
cycles. The output table records the actual selected run and availability time,
so this difference is visible rather than silently imputed.

## Commands

```bash
# 2024 archive: the historical HRES archive exposes the daily 00 UTC run.
energy collect-ecmwf-ifs-weather \
  --project-root . \
  --start 2024-03-14 --end 2024-12-31 \
  --run-hours 0

# 2025 archive: retain all updates for a forward evaluation.
energy collect-ecmwf-ifs-weather \
  --project-root . \
  --start 2025-01-01 --end 2025-09-30 \
  --run-hours 0 6 12 18

energy build-mpc-weather-snapshots --project-root . --year 2024 \
  --start 2024-03-14 --end 2024-12-31
energy build-mpc-weather-snapshots --project-root . --year 2025 \
  --start 2025-01-01 --end 2025-09-30

energy train-pv-mpc-residual \
  --project-root . --year 2024 \
  --train-start-date 2024-03-14 \
  --weather-snapshot-path \
    data/features/mpc_pv_weather_ifs_snapshots/mpc_pv_weather_ifs_snapshots_2024.parquet

# Reproduce the controlled forward comparison on 2025.
energy run-pv-mpc-ifs-weather-experiment --project-root . \
  --train-start-date 2024-03-14 \
  --test-start-date 2025-01-01 --test-end-date 2025-09-30
```

## Model contract

The residual model retains the D-1 ICON weather features used by the frozen
base forecast. It receives the fresh IFS level **and** the IFS-minus-ICON
difference for every weather variable, plus IFS lead time and published-snapshot
age. This matters because the two providers can have different systematic
biases: the model may keep the original forecast when an update is unhelpful
and learn where the update carries new information. Actual PV residual lags
remain strictly limited to completed hours before the decision.

This separates the two questions in the evaluation: whether day-ahead planning
is sound, and whether a fresh weather update improves correction of its
remaining intra-day PV trajectory.

## Frozen 2025 forward result

The first controlled comparison trains on 14 March–31 December 2024 and tests
on 1 January–30 September 2025. All three variants use the same 14,582 MPC
decision/target pairs, the same D-1 prediction and the same factual residual
lags.

| Variant | MAE, W | RMSE, W |
|---|---:|---:|
| Frozen day-ahead base | 17.97 | 33.89 |
| Direct residual, D-1 ICON only | **16.77** | 31.80 |
| Direct residual, ICON + IFS level/delta | 17.44 | **31.00** |

The IFS layer reduces the 95th percentile absolute error from 69.37 W to
67.44 W, but worsens median and mean absolute error. It is therefore retained
as a documented V1.1 challenger and a possible tail-risk signal; it does not
replace the MAE-champion direct residual model in the deterministic MPC or
economic replay. A future selection experiment can choose an ensemble weight
on a 2024 validation segment, then retest it once on 2025.

The reproducible predictions and metrics are in
`artifacts/experiments/pv_mpc_ifs_weather_update_2024_to_2025/`.

## 2026-10 rolling re-evaluation

The original frozen-2024 comparison above is retained for provenance. A later
audit found that 2024 contains only 00 UTC IFS runs, whereas 2025 contains the
00/06/12/18 UTC cycles. The frozen experiment therefore could not learn the
behavior of the fresh daytime updates. It also supplied the old level, new
level and exact level difference simultaneously.

The replacement evaluation uses expanding monthly fits, selects the weather
representation on 2025 Q2 and holds 2025 Q3 back. Replacing the stale weather
fields with the latest on-time IFS fields improves Q3 MAE from 16.98 to
16.46 W and RMSE from 32.78 to 31.31 W against the rolling original-weather
model. Its Q3 economic benefit is positive but small: EUR 0.23 over 91 days.

See [pv_mpc_weather_refresh_evaluation.md](pv_mpc_weather_refresh_evaluation.md)
for the audit, feature contract, validation design and economic replay.

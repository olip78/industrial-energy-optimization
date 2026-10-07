# Energy optimization project

## Current data layer

The project stores original source responses under `data/raw` and analysis-ready tables under `data/processed`.

- `pv_power_15min.csv.gz`: measured KIT MPVBench active power for five PV profiles. Profile `1a` is the preliminary working profile. Its source timestamps have an **unknown timezone**.
- `weather_actual_hourly_era5.csv.gz`: ERA5 reanalysis for an approximate Pforzheim location (48.89, 8.70), used as a historical weather estimate.
- `weather_forecast_hourly_icon_lead_24_48.csv.gz`: archived ICON weather forecasts at fixed 24- and 48-hour lead times. This supports a fixed-lead day-ahead experiment but is not a complete forecast trajectory from one historical decision time.
- `weather_forecast_hourly_icon_spatial_lead_24_48.csv.gz`: fixed 24- and 48-hour-lead ICON forecasts for ten representative DE-LU weather-regime locations. It is separate from the PV proxy and supports spatial weather features for day-ahead price experiments.
- `prices_day_ahead_de_lu.csv.gz`: DE-LU day-ahead auction outcomes in EUR/MWh. The series is hourly before 1 October 2025 and 15-minute afterwards. These are settled prices, not prices known before auction clearing or intraday prices.
- `coverage_by_month.csv`: counts for each monthly data layer.

The approximate Pforzheim coordinate is a modelling choice because the KIT metadata gives district/city rather than confirmed panel coordinates. It is not a claim about the actual location of PV profile 1a.

Read `docs/data_quality_report.md` before joining or modelling the data. In particular, do not join source PV timestamps to UTC weather or price timestamps until the source timezone has been confirmed or an explicit modelling assumption has been made.

## Reproduction

`scripts/collect_energy_data.py` refreshes the raw and processed data. Its forecast collection is resumable by month. `scripts/audit_energy_data.py` recreates the coverage and quality reports.

## Training data pipeline

The Python package under `src/energy` builds point-in-time datasets from the processed source tables. Create a Python 3.11+ environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Rebuild the 2024 datasets:

```bash
energy build-training-data --project-root . --year 2024
```

The outputs are stored under `data/curated`, `data/features` and `data/metadata`. The generated `energy_training.duckdb` contains materialized tables for exploration, while the Parquet files are the model-training interface. See `docs/training_datasets_2024.md` for targets, leakage rules and known limitations.

Run the data-contract checks with:

```bash
pytest
```

## PV model training and Model Registry

The deployable day-ahead PV model has a small training application with MLflow
tracking and optional Model Registry registration. Its setup, commands and
feature-availability contract are described in
[docs/pv_training_and_mlflow.md](docs/pv_training_and_mlflow.md).

The direct PV residual correction model for MPC is documented separately in
[docs/pv_mpc_residual_training.md](docs/pv_mpc_residual_training.md).

The joint multi-horizon MIMO neural residual experiments are documented in
[docs/pv_mimo_mlp_experiment.md](docs/pv_mimo_mlp_experiment.md).

A separate pooled-profile experiment, which leaves the `1a` data path intact, is described in
[docs/pv_multisite_experiment.md](docs/pv_multisite_experiment.md).

Spatial weather collection and its point-in-time contract are described in
[docs/spatial_weather_data.md](docs/spatial_weather_data.md).

## Day-ahead price experiment

The compact price-only, local-weather and spatial-weather comparison is documented in
[docs/day_ahead_price_experiment.md](docs/day_ahead_price_experiment.md).

The final frozen 2024 → 2025 evaluation of the PV, MPC and price forecasters is described in
[docs/temporal_backtest_2025.md](docs/temporal_backtest_2025.md).

The hourly intraday continuous price data contract and compact MPC forecaster are described in
[docs/intraday_price_forecasting.md](docs/intraday_price_forecasting.md).

The intraday experiment includes a 2024-only selection window and a frozen 2025 holdout. Its V1 champion uses a CatBoost spread correction for the next delivery hour only, then retains the known day-ahead curve for the rest of the MPC horizon.

The packaged fourth forecaster and its MLflow Model Registry flow are described in [docs/intraday_price_training.md](docs/intraday_price_training.md).

The reusable deterministic day-ahead, MPC and settlement-ledger core is described in [docs/deterministic_optimization.md](docs/deterministic_optimization.md).

The first probabilistic component, a joint rolling-origin residual bootstrap
for PV and day-ahead prices, is documented in
[docs/residual_bootstrap_scenarios.md](docs/residual_bootstrap_scenarios.md).

The corresponding 2025 SAA/CVaR economic replay and its cost-versus-tail-risk
frontier are documented in
[docs/stochastic_economic_backtest_v1.md](docs/stochastic_economic_backtest_v1.md).

The conditional MultiQuantile plus empirical-copula experiment, including its
calibration diagnosis and economic replay, is documented in
[docs/quantile_copula_experiment.md](docs/quantile_copula_experiment.md).

The conformally calibrated extension and its final comparison with residual
bootstrap are documented in
[docs/cqr_copula_experiment.md](docs/cqr_copula_experiment.md).

The V2 three-target quantile generator for PV, day-ahead price and the
intraday-minus-day-ahead spread is documented in
[docs/quantile_spread_copula_experiment.md](docs/quantile_spread_copula_experiment.md).

Its day-ahead-only SAA/CVaR economic replay under the final V4 battery and
grid-tariff ledger is documented in
[docs/quantile_spread_economic_backtest_v2.md](docs/quantile_spread_economic_backtest_v2.md).

The matching three-target whole-day residual bootstrap and its 2025 economic
comparison with the quantile approach are documented in
[docs/residual_bootstrap_spread_economic_backtest_v3.md](docs/residual_bootstrap_spread_economic_backtest_v3.md).
The same report includes the circular seasonal-kernel extension, its scenario
calibration audit and the frozen 2025 SAA/CVaR replay.

The first frozen 2025 economic replay, including the fixed site, rule-based
baseline, command and result artifacts, is documented in
[docs/economic_backtest_v1.md](docs/economic_backtest_v1.md).

The final deterministic replay formulation with dynamic MPC battery control,
PV curtailment, the strengthened rule baseline and the 2025 Pforzheim variable
grid tariff is documented in
[docs/economic_backtest_v4_final_formulation.md](docs/economic_backtest_v4_final_formulation.md).
The previous numerical V3.1 result remains archived in
[docs/economic_backtest_v3_grid_tariff.md](docs/economic_backtest_v3_grid_tariff.md).

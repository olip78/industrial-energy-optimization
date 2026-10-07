"""Command line entry point for reproducible energy data builds."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from energy.data import TrainingDatasetBuilder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="energy")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-training-data")
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path)
    build.add_argument("--year", type=int, default=2024)

    collect_ifs_weather = commands.add_parser(
        "collect-ecmwf-ifs-weather",
        help="download resumable archived ECMWF IFS runs for intraday PV forecasts",
    )
    collect_ifs_weather.add_argument("--project-root", type=Path, required=True)
    collect_ifs_weather.add_argument("--start", type=date.fromisoformat, default=date(2024, 3, 14))
    collect_ifs_weather.add_argument("--end", type=date.fromisoformat, default=date(2025, 9, 30))
    collect_ifs_weather.add_argument("--forecast-hours", type=int, default=48)
    collect_ifs_weather.add_argument(
        "--run-hours",
        type=int,
        nargs="+",
        default=[0, 6, 12, 18],
        help="ECMWF initialisation hours to retrieve; use 0 for the daily-only 2024 archive layer",
    )

    materialize_ifs_weather = commands.add_parser(
        "materialize-ecmwf-ifs-weather",
        help="combine locally cached ECMWF IFS runs into one processed table",
    )
    materialize_ifs_weather.add_argument("--project-root", type=Path, required=True)

    collect_spatial_ifs_weather = commands.add_parser(
        "collect-ecmwf-ifs-spatial-weather",
        help="download archived ECMWF IFS runs for the ten DE-LU price-weather coordinates",
    )
    collect_spatial_ifs_weather.add_argument("--project-root", type=Path, required=True)
    collect_spatial_ifs_weather.add_argument(
        "--start", type=date.fromisoformat, default=date(2024, 3, 14)
    )
    collect_spatial_ifs_weather.add_argument(
        "--end", type=date.fromisoformat, default=date(2025, 9, 30)
    )
    collect_spatial_ifs_weather.add_argument("--forecast-hours", type=int, default=48)
    collect_spatial_ifs_weather.add_argument(
        "--workers", type=int, default=1, help="parallel requests, capped at two"
    )
    collect_spatial_ifs_weather.add_argument(
        "--run-hours",
        type=int,
        nargs="+",
        default=[0, 6, 12, 18],
        help="ECMWF initialisation hours; use 0 for the daily-only 2024 archive layer",
    )

    materialize_spatial_ifs_weather = commands.add_parser(
        "materialize-ecmwf-ifs-spatial-weather",
        help="combine cached spatial ECMWF IFS runs into the price-weather table",
    )
    materialize_spatial_ifs_weather.add_argument("--project-root", type=Path, required=True)

    build_ifs_snapshots = commands.add_parser(
        "build-mpc-weather-snapshots",
        help="select the latest published ECMWF run for every MPC decision/target pair",
    )
    build_ifs_snapshots.add_argument("--project-root", type=Path, required=True)
    build_ifs_snapshots.add_argument("--year", type=int, required=True)
    build_ifs_snapshots.add_argument("--source-path", type=Path)
    build_ifs_snapshots.add_argument("--start", type=date.fromisoformat)
    build_ifs_snapshots.add_argument("--end", type=date.fromisoformat)

    build_multisite = commands.add_parser(
        "build-multisite-pv-data",
        help="build the isolated multi-site PV training table from eligible MPVBench profiles",
    )
    build_multisite.add_argument("--project-root", type=Path, required=True)
    build_multisite.add_argument("--year", type=int, default=2024)

    run_multisite = commands.add_parser(
        "run-pv-multisite-experiment",
        help="compare direct CatBoost, MIMO MLP and MIMO LSTM on the parallel multi-site data",
    )
    run_multisite.add_argument("--project-root", type=Path, required=True)
    run_multisite.add_argument("--data-path", type=Path)
    run_multisite.add_argument("--year", type=int, default=2024)
    run_multisite.add_argument("--n-lags", type=int, default=4)
    run_multisite.add_argument("--cv-splits", type=int, default=3)
    run_multisite.add_argument("--catboost-iterations", type=int, default=500)
    run_multisite.add_argument("--random-state", type=int, default=42)

    build_price_spatial = commands.add_parser(
        "build-day-ahead-price-spatial-data",
        help="build leakage-safe day-ahead price features with spatial ICON weather",
    )
    build_price_spatial.add_argument("--project-root", type=Path, required=True)
    build_price_spatial.add_argument("--start", type=date.fromisoformat, default=date(2024, 2, 18))
    build_price_spatial.add_argument("--end", type=date.fromisoformat, default=date(2025, 9, 30))

    run_price = commands.add_parser(
        "run-day-ahead-price-experiment",
        help="compare persistence, price-only, local-weather and spatial-weather price models",
    )
    run_price.add_argument("--project-root", type=Path, required=True)
    run_price.add_argument("--data-path", type=Path)
    run_price.add_argument("--train-end", default="2024-12-31")
    run_price.add_argument("--test-start", default="2025-01-01")
    run_price.add_argument("--test-end", default="2025-09-30")
    run_price.add_argument("--iterations", type=int, default=500)
    run_price.add_argument("--random-state", type=int, default=42)

    collect_intraday = commands.add_parser(
        "collect-intraday-price-data",
        help="download and audit public DE-LU hourly intraday continuous indices",
    )
    collect_intraday.add_argument("--project-root", type=Path, required=True)
    collect_intraday.add_argument("--years", type=int, nargs="+", default=[2024, 2025])

    build_intraday = commands.add_parser(
        "build-intraday-price-data",
        help="build a leakage-safe intraday price MPC decision/target table",
    )
    build_intraday.add_argument("--project-root", type=Path, required=True)
    build_intraday.add_argument("--source-path", type=Path)
    build_intraday.add_argument("--feature-version", choices=["v1", "v2"], default="v1")

    run_intraday = commands.add_parser(
        "run-intraday-price-experiment",
        help="compare compact intraday price forecasts on a chronological future holdout",
    )
    run_intraday.add_argument("--project-root", type=Path, required=True)
    run_intraday.add_argument("--data-path", type=Path)
    run_intraday.add_argument("--train-year", type=int, default=2024)
    run_intraday.add_argument("--test-year", type=int, default=2025)
    run_intraday.add_argument("--test-end", default="2025-09-30")
    run_intraday.add_argument(
        "--validation-start",
        default="2024-10-01",
        help="first local delivery date reserved for the 2024 model-selection window",
    )
    run_intraday.add_argument("--feature-version", choices=["v1", "v2"], default="v1")
    run_intraday.add_argument(
        "--artifact-name",
        default="intraday_price_v1",
        help="subdirectory below artifacts/experiments for this feature experiment",
    )
    run_intraday.add_argument("--iterations", type=int, default=500)
    run_intraday.add_argument("--random-state", type=int, default=42)

    run_intraday_spatial_weather = commands.add_parser(
        "run-intraday-price-spatial-weather-experiment",
        help="compare price-only and spatial-IFS next-hour intraday price corrections on 2024 Q4",
    )
    run_intraday_spatial_weather.add_argument("--project-root", type=Path, required=True)
    run_intraday_spatial_weather.add_argument("--data-path", type=Path)
    run_intraday_spatial_weather.add_argument("--weather-path", type=Path)
    run_intraday_spatial_weather.add_argument("--train-start", default="2024-03-15")
    run_intraday_spatial_weather.add_argument("--validation-start", default="2024-10-01")
    run_intraday_spatial_weather.add_argument("--end", default="2024-12-31")
    run_intraday_spatial_weather.add_argument("--correction-weight", type=float, default=0.70)
    run_intraday_spatial_weather.add_argument("--iterations", type=int, default=500)
    run_intraday_spatial_weather.add_argument("--random-state", type=int, default=42)
    run_intraday_spatial_weather.add_argument(
        "--artifact-name", default="intraday_price_spatial_ifs_2024"
    )

    run_intraday_spatial_weather_week_cv = commands.add_parser(
        "run-intraday-price-spatial-weather-week-cv",
        help="compare price-only and spatial-IFS corrections with randomized whole-week CV",
    )
    run_intraday_spatial_weather_week_cv.add_argument("--project-root", type=Path, required=True)
    run_intraday_spatial_weather_week_cv.add_argument("--data-path", type=Path)
    run_intraday_spatial_weather_week_cv.add_argument("--weather-path", type=Path)
    run_intraday_spatial_weather_week_cv.add_argument("--train-start", default="2024-03-15")
    run_intraday_spatial_weather_week_cv.add_argument("--end", default="2024-12-31")
    run_intraday_spatial_weather_week_cv.add_argument("--correction-weight", type=float, default=0.70)
    run_intraday_spatial_weather_week_cv.add_argument("--iterations", type=int, default=500)
    run_intraday_spatial_weather_week_cv.add_argument("--random-state", type=int, default=42)
    run_intraday_spatial_weather_week_cv.add_argument("--n-splits", type=int, default=3)
    run_intraday_spatial_weather_week_cv.add_argument(
        "--artifact-name", default="intraday_price_spatial_ifs_week_cv_2024"
    )

    residual_bootstrap = commands.add_parser(
        "run-residual-bootstrap-experiment",
        help="build rolling-origin paired PV/price errors and evaluate joint bootstrap scenarios",
    )
    residual_bootstrap.add_argument("--project-root", type=Path, required=True)
    residual_bootstrap.add_argument("--residual-year", type=int, default=2024)
    residual_bootstrap.add_argument("--test-start", default="2025-01-01")
    residual_bootstrap.add_argument("--test-end", default="2025-09-30")
    residual_bootstrap.add_argument("--initial-train-days", type=int, default=84)
    residual_bootstrap.add_argument("--block-days", type=int, default=28)
    residual_bootstrap.add_argument("--iterations", type=int, default=500)
    residual_bootstrap.add_argument("--n-scenarios", type=int, default=500)
    residual_bootstrap.add_argument("--random-state", type=int, default=42)
    residual_bootstrap.add_argument("--center-residuals", action="store_true")
    residual_bootstrap.add_argument("--pv-capacity-kwh-per-hour", type=float, default=10.0)
    residual_bootstrap.add_argument(
        "--artifact-name", default="joint_residual_bootstrap_v1"
    )

    residual_spread = commands.add_parser(
        "run-residual-bootstrap-spread-experiment",
        help=(
            "join rolling-origin PV, day-ahead-price and intraday-spread "
            "errors and audit complete-day bootstrap scenarios"
        ),
    )
    residual_spread.add_argument("--project-root", type=Path, required=True)
    residual_spread.add_argument(
        "--residual-artifact-name", default="joint_residual_bootstrap_current_v3"
    )
    residual_spread.add_argument(
        "--quantile-artifact-name", default="quantile_spread_copula_v2"
    )
    residual_spread.add_argument("--n-scenarios", type=int, default=500)
    residual_spread.add_argument("--random-state", type=int, default=42)
    residual_spread.add_argument(
        "--pv-capacity-kwh-per-hour", type=float, default=10.0
    )
    residual_spread.add_argument(
        "--seasonal-bandwidth-days", type=float, default=25.0
    )
    residual_spread.add_argument(
        "--global-mixture-weight", type=float, default=0.15
    )
    residual_spread.add_argument(
        "--artifact-name", default="joint_residual_bootstrap_spread_seasonal_v4"
    )

    quantile_copula = commands.add_parser(
        "run-quantile-copula-experiment",
        help="fit conditional PV/price quantiles and evaluate empirical-copula scenarios",
    )
    quantile_copula.add_argument("--project-root", type=Path, required=True)
    quantile_copula.add_argument("--residual-year", type=int, default=2024)
    quantile_copula.add_argument("--test-start", default="2025-01-01")
    quantile_copula.add_argument("--test-end", default="2025-09-30")
    quantile_copula.add_argument("--initial-train-days", type=int, default=84)
    quantile_copula.add_argument("--block-days", type=int, default=28)
    quantile_copula.add_argument("--iterations", type=int, default=500)
    quantile_copula.add_argument("--n-scenarios", type=int, default=500)
    quantile_copula.add_argument("--random-state", type=int, default=42)
    quantile_copula.add_argument("--pv-capacity-kwh-per-hour", type=float, default=10.0)
    quantile_copula.add_argument("--artifact-name", default="quantile_copula_v1")

    quantile_spread_copula = commands.add_parser(
        "run-quantile-spread-copula-experiment",
        help=(
            "fit the intraday-spread quantiles and build calibrated joint "
            "PV/day-ahead-price/spread scenarios"
        ),
    )
    quantile_spread_copula.add_argument("--project-root", type=Path, required=True)
    quantile_spread_copula.add_argument(
        "--source-artifact-name", default="quantile_copula_v1"
    )
    quantile_spread_copula.add_argument("--residual-year", type=int, default=2024)
    quantile_spread_copula.add_argument("--test-start", default="2025-01-01")
    quantile_spread_copula.add_argument("--test-end", default="2025-09-30")
    quantile_spread_copula.add_argument("--iterations", type=int, default=500)
    quantile_spread_copula.add_argument("--n-scenarios", type=int, default=500)
    quantile_spread_copula.add_argument("--random-state", type=int, default=42)
    quantile_spread_copula.add_argument(
        "--pv-capacity-kwh-per-hour", type=float, default=10.0
    )
    quantile_spread_copula.add_argument(
        "--artifact-name", default="quantile_spread_copula_v2"
    )

    quantile_spread_economic = commands.add_parser(
        "run-quantile-spread-economic-backtest",
        help=(
            "replay raw/CQR SAA and CVaR day-ahead plans with explicit "
            "intraday-spread settlement"
        ),
    )
    quantile_spread_economic.add_argument("--project-root", type=Path, required=True)
    quantile_spread_economic.add_argument("--start", default="2025-01-01")
    quantile_spread_economic.add_argument("--end", default="2025-09-30")
    quantile_spread_economic.add_argument(
        "--quantile-artifact-name", default="quantile_spread_copula_v2"
    )
    quantile_spread_economic.add_argument(
        "--generators", nargs="+", choices=("raw", "cqr"), default=("raw", "cqr")
    )
    quantile_spread_economic.add_argument("--n-scenarios", type=int, default=500)
    quantile_spread_economic.add_argument("--cvar-alpha", type=float, default=0.95)
    quantile_spread_economic.add_argument(
        "--risk-weights",
        nargs="+",
        type=float,
        default=(0.05, 0.10, 0.25, 0.50),
    )
    quantile_spread_economic.add_argument("--random-state", type=int, default=42)
    quantile_spread_economic.add_argument(
        "--economic-bootstrap-resamples", type=int, default=5000
    )
    quantile_spread_economic.add_argument(
        "--economic-bootstrap-block-days", type=int, default=7
    )
    quantile_spread_economic.add_argument(
        "--artifact-name", default="quantile_spread_economic_backtest_v2"
    )

    residual_spread_economic = commands.add_parser(
        "run-residual-bootstrap-spread-economic-backtest",
        help=(
            "replay raw/centered residual-bootstrap SAA and centered CVaR "
            "day-ahead plans with explicit intraday-spread settlement"
        ),
    )
    residual_spread_economic.add_argument(
        "--project-root", type=Path, required=True
    )
    residual_spread_economic.add_argument("--start", default="2025-01-01")
    residual_spread_economic.add_argument("--end", default="2025-09-30")
    residual_spread_economic.add_argument(
        "--residual-artifact-name",
        default="joint_residual_bootstrap_spread_seasonal_v4",
    )
    residual_spread_economic.add_argument(
        "--generators",
        nargs="+",
        choices=("raw", "centered", "seasonal"),
        default=("raw", "centered", "seasonal"),
    )
    residual_spread_economic.add_argument(
        "--risk-generators",
        nargs="+",
        choices=("raw", "centered", "seasonal"),
        default=("centered", "seasonal"),
    )
    residual_spread_economic.add_argument(
        "--seasonal-bandwidth-days", type=float, default=25.0
    )
    residual_spread_economic.add_argument(
        "--global-mixture-weight", type=float, default=0.15
    )
    residual_spread_economic.add_argument("--n-scenarios", type=int, default=500)
    residual_spread_economic.add_argument("--cvar-alpha", type=float, default=0.95)
    residual_spread_economic.add_argument(
        "--risk-weights",
        nargs="+",
        type=float,
        default=(0.05, 0.10, 0.25, 0.50),
    )
    residual_spread_economic.add_argument("--random-state", type=int, default=42)
    residual_spread_economic.add_argument(
        "--economic-bootstrap-resamples", type=int, default=5000
    )
    residual_spread_economic.add_argument(
        "--economic-bootstrap-block-days", type=int, default=7
    )
    residual_spread_economic.add_argument(
        "--artifact-name",
        default="residual_bootstrap_spread_seasonal_economic_v4",
    )

    cqr_copula = commands.add_parser(
        "run-cqr-copula-experiment",
        help="conformally calibrate quantile marginals and evaluate copula scenarios",
    )
    cqr_copula.add_argument("--project-root", type=Path, required=True)
    cqr_copula.add_argument("--source-artifact-name", default="quantile_copula_v1")
    cqr_copula.add_argument("--n-scenarios", type=int, default=500)
    cqr_copula.add_argument("--random-state", type=int, default=42)
    cqr_copula.add_argument("--pv-capacity-kwh-per-hour", type=float, default=10.0)
    cqr_copula.add_argument("--artifact-name", default="cqr_copula_v1")

    temporal_backtest = commands.add_parser(
        "run-temporal-backtest-2025",
        help="fit all V1 forecasters on 2024 and evaluate the future 2025 period",
    )
    temporal_backtest.add_argument("--project-root", type=Path, required=True)
    temporal_backtest.add_argument("--train-year", type=int, default=2024)
    temporal_backtest.add_argument("--test-year", type=int, default=2025)
    temporal_backtest.add_argument("--n-lags", type=int, default=4)
    temporal_backtest.add_argument("--catboost-iterations", type=int, default=500)
    temporal_backtest.add_argument("--neural-epochs", type=int, default=300)
    temporal_backtest.add_argument("--random-state", type=int, default=42)

    economic_backtest = commands.add_parser(
        "run-economic-backtest",
        help="replay the frozen V1 forecasts through the rule-based and deterministic policies",
    )
    economic_backtest.add_argument("--project-root", type=Path, required=True)
    economic_backtest.add_argument("--start", default="2025-01-01")
    economic_backtest.add_argument("--end", default="2025-09-30")
    economic_backtest.add_argument("--artifact-name", default="economic_backtest_v1")
    economic_backtest.add_argument("--correction-weight", type=float, default=0.70)

    dynamic_charge_economic = commands.add_parser(
        "run-dynamic-charge-economic-backtest",
        help=(
            "replay rule-based, day-ahead, deterministic MPC and Oracle policies "
            "with explicit battery charge and state of charge"
        ),
    )
    dynamic_charge_economic.add_argument("--project-root", type=Path, required=True)
    dynamic_charge_economic.add_argument("--start", default="2025-01-01")
    dynamic_charge_economic.add_argument("--end", default="2025-09-30")
    dynamic_charge_economic.add_argument(
        "--artifact-name", default="economic_backtest_v4_final_formulation"
    )
    dynamic_charge_economic.add_argument(
        "--correction-weight", type=float, default=0.70
    )
    dynamic_charge_economic.add_argument(
        "--pv-mpc-prediction-path",
        type=Path,
        help="optional long-form PV MPC prediction artifact",
    )
    dynamic_charge_economic.add_argument(
        "--include-stochastic",
        action="store_true",
        help="also replay the legacy CQR-copula/CVaR day-ahead strategy",
    )
    dynamic_charge_economic.add_argument(
        "--grid-tariff-preset",
        choices=("pforzheim-slp-2025", "none"),
        default="pforzheim-slp-2025",
    )

    stochastic_economic = commands.add_parser(
        "run-stochastic-economic-backtest",
        help="replay residual-bootstrap SAA and CVaR day-ahead policies on 2025 facts",
    )
    stochastic_economic.add_argument("--project-root", type=Path, required=True)
    stochastic_economic.add_argument("--start", default="2025-01-01")
    stochastic_economic.add_argument("--end", default="2025-09-30")
    stochastic_economic.add_argument("--residual-library-path", type=Path)
    stochastic_economic.add_argument("--n-scenarios", type=int, default=500)
    stochastic_economic.add_argument("--cvar-alpha", type=float, default=0.95)
    stochastic_economic.add_argument(
        "--risk-weights", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50, 1.0]
    )
    stochastic_economic.add_argument(
        "--center-residuals",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    stochastic_economic.add_argument("--pv-capacity-kwh-per-hour", type=float, default=10.0)
    stochastic_economic.add_argument("--random-state", type=int, default=42)
    stochastic_economic.add_argument("--correction-weight", type=float, default=0.70)
    stochastic_economic.add_argument("--economic-bootstrap-resamples", type=int, default=5000)
    stochastic_economic.add_argument("--economic-bootstrap-block-days", type=int, default=7)
    stochastic_economic.add_argument(
        "--artifact-name", default="stochastic_economic_backtest_v1"
    )

    quantile_copula_economic = commands.add_parser(
        "run-quantile-copula-economic-backtest",
        help="replay quantile-copula SAA and CVaR policies on the 2025 ledger",
    )
    quantile_copula_economic.add_argument("--project-root", type=Path, required=True)
    quantile_copula_economic.add_argument(
        "--quantile-artifact-name", default="quantile_copula_v1"
    )
    quantile_copula_economic.add_argument("--start", default="2025-01-01")
    quantile_copula_economic.add_argument("--end", default="2025-09-30")
    quantile_copula_economic.add_argument("--n-scenarios", type=int, default=500)
    quantile_copula_economic.add_argument("--cvar-alpha", type=float, default=0.95)
    quantile_copula_economic.add_argument(
        "--risk-weights", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50, 1.0]
    )
    quantile_copula_economic.add_argument("--random-state", type=int, default=42)
    quantile_copula_economic.add_argument("--economic-bootstrap-resamples", type=int, default=5000)
    quantile_copula_economic.add_argument("--economic-bootstrap-block-days", type=int, default=7)
    quantile_copula_economic.add_argument(
        "--artifact-name", default="quantile_copula_economic_backtest_v1"
    )

    cqr_copula_economic = commands.add_parser(
        "run-cqr-copula-economic-backtest",
        help="replay conformally calibrated quantile-copula SAA/CVaR policies",
    )
    cqr_copula_economic.add_argument("--project-root", type=Path, required=True)
    cqr_copula_economic.add_argument("--quantile-artifact-name", default="cqr_copula_v1")
    cqr_copula_economic.add_argument("--start", default="2025-01-01")
    cqr_copula_economic.add_argument("--end", default="2025-09-30")
    cqr_copula_economic.add_argument("--n-scenarios", type=int, default=500)
    cqr_copula_economic.add_argument("--cvar-alpha", type=float, default=0.95)
    cqr_copula_economic.add_argument(
        "--risk-weights", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50, 1.0]
    )
    cqr_copula_economic.add_argument("--random-state", type=int, default=42)
    cqr_copula_economic.add_argument("--economic-bootstrap-resamples", type=int, default=5000)
    cqr_copula_economic.add_argument("--economic-bootstrap-block-days", type=int, default=7)
    cqr_copula_economic.add_argument(
        "--artifact-name", default="cqr_copula_economic_backtest_v1"
    )

    train_pv = commands.add_parser(
        "train-pv-day-ahead",
        help="train the deployable day-ahead PV model and log an MLflow run",
    )
    train_pv.add_argument("--project-root", type=Path, required=True)
    train_pv.add_argument("--data-path", type=Path)
    train_pv.add_argument("--year", type=int, default=2024)
    train_pv.add_argument("--tracking-uri")
    train_pv.add_argument("--experiment-name")
    train_pv.add_argument("--hour-start", type=int, default=6)
    train_pv.add_argument("--hour-end", type=int, default=21)
    train_pv.add_argument("--min-clear-sky-ghi", type=float, default=25.0)
    train_pv.add_argument("--random-state", type=int, default=42)
    train_pv.add_argument("--register", action="store_true")
    train_pv.add_argument("--registered-model-name")

    train_residual = commands.add_parser(
        "train-pv-mpc-residual",
        help="train the direct MPC PV residual model and log an MLflow run",
    )
    train_residual.add_argument("--project-root", type=Path, required=True)
    train_residual.add_argument("--data-path", type=Path)
    train_residual.add_argument(
        "--weather-snapshot-path",
        type=Path,
        help="MPC decision/target weather snapshots built from published ECMWF IFS runs",
    )
    train_residual.add_argument("--year", type=int, default=2024)
    train_residual.add_argument(
        "--train-start-date",
        help="optional ISO local date cutoff, e.g. 2024-03-14 for the IFS archive",
    )
    train_residual.add_argument("--tracking-uri")
    train_residual.add_argument("--experiment-name")
    train_residual.add_argument("--n-lags", type=int, default=4)
    train_residual.add_argument("--min-clear-sky-ghi", type=float, default=25.0)
    train_residual.add_argument("--cv-splits", type=int, default=3)
    train_residual.add_argument("--iterations", type=int, default=500)
    train_residual.add_argument("--random-state", type=int, default=42)
    train_residual.add_argument(
        "--updated-weather-feature-mode",
        choices=("replace", "revision", "legacy"),
        default="replace",
        help=(
            "how the residual model uses an intraday IFS snapshot; 'replace' "
            "is the validated default"
        ),
    )
    train_residual.add_argument("--register", action="store_true")
    train_residual.add_argument("--registered-model-name")

    ifs_weather_experiment = commands.add_parser(
        "run-pv-mpc-ifs-weather-experiment",
        help="compare original and updated IFS weather for direct MPC PV residual correction",
    )
    ifs_weather_experiment.add_argument("--project-root", type=Path, required=True)
    ifs_weather_experiment.add_argument("--train-start-date", default="2024-03-14")
    ifs_weather_experiment.add_argument("--test-start-date", default="2025-01-01")
    ifs_weather_experiment.add_argument("--test-end-date", default="2025-09-30")
    ifs_weather_experiment.add_argument("--n-lags", type=int, default=4)
    ifs_weather_experiment.add_argument("--iterations", type=int, default=500)
    ifs_weather_experiment.add_argument("--random-state", type=int, default=42)
    ifs_weather_experiment.add_argument(
        "--artifact-name", default="pv_mpc_ifs_weather_update_2024_to_2025"
    )

    weather_refresh_evaluation = commands.add_parser(
        "run-pv-mpc-weather-refresh-evaluation",
        help=(
            "select intraday weather features with rolling 2025 fits and "
            "evaluate them on a locked Q3 holdout"
        ),
    )
    weather_refresh_evaluation.add_argument("--project-root", type=Path, required=True)
    weather_refresh_evaluation.add_argument("--train-start-date", default="2024-03-14")
    weather_refresh_evaluation.add_argument("--rolling-start-date", default="2025-04-01")
    weather_refresh_evaluation.add_argument("--selection-end-date", default="2025-06-30")
    weather_refresh_evaluation.add_argument("--holdout-start-date", default="2025-07-01")
    weather_refresh_evaluation.add_argument("--test-end-date", default="2025-09-30")
    weather_refresh_evaluation.add_argument("--n-lags", type=int, default=4)
    weather_refresh_evaluation.add_argument("--iterations", type=int, default=500)
    weather_refresh_evaluation.add_argument("--random-state", type=int, default=42)
    weather_refresh_evaluation.add_argument(
        "--artifact-name", default="pv_mpc_weather_refresh_rolling_v1"
    )

    train_intraday = commands.add_parser(
        "train-intraday-price",
        help="train the calibrated one-hour intraday spread model and log an MLflow run",
    )
    train_intraday.add_argument("--project-root", type=Path, required=True)
    train_intraday.add_argument("--data-path", type=Path)
    train_intraday.add_argument("--year", type=int, default=2024)
    train_intraday.add_argument("--validation-start", default="2024-10-01")
    train_intraday.add_argument("--correction-weight", type=float, default=0.70)
    train_intraday.add_argument("--iterations", type=int, default=500)
    train_intraday.add_argument("--random-state", type=int, default=42)
    train_intraday.add_argument("--tracking-uri")
    train_intraday.add_argument("--experiment-name")
    train_intraday.add_argument("--register", action="store_true")
    train_intraday.add_argument("--registered-model-name")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "build-training-data":
        result = TrainingDatasetBuilder(
            project_root=arguments.project_root,
            output_root=arguments.output_root,
        ).build(arguments.year)
        print(
            json.dumps(
                {
                    "canonical": str(result.canonical_path),
                    "datasets": {key: str(value) for key, value in result.dataset_paths.items()},
                    "manifest": str(result.manifest_path),
                    "duckdb": str(result.duckdb_path),
                },
                indent=2,
            )
        )
    elif arguments.command == "collect-ecmwf-ifs-weather":
        from energy.data import EcmwfIfsArchiveConfig, collect_ecmwf_ifs_single_runs

        result = collect_ecmwf_ifs_single_runs(
            EcmwfIfsArchiveConfig(
                project_root=arguments.project_root,
                start=arguments.start,
                end=arguments.end,
                forecast_hours=arguments.forecast_hours,
                run_hours_utc=tuple(arguments.run_hours),
                max_workers=arguments.workers,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "materialize-ecmwf-ifs-weather":
        from energy.data import materialize_ecmwf_ifs_archive

        result = materialize_ecmwf_ifs_archive(arguments.project_root)
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "collect-ecmwf-ifs-spatial-weather":
        from energy.data import (
            SpatialEcmwfIfsArchiveConfig,
            collect_spatial_ecmwf_ifs_single_runs,
        )

        result = collect_spatial_ecmwf_ifs_single_runs(
            SpatialEcmwfIfsArchiveConfig(
                project_root=arguments.project_root,
                start=arguments.start,
                end=arguments.end,
                forecast_hours=arguments.forecast_hours,
                run_hours_utc=tuple(arguments.run_hours),
                max_workers=arguments.workers,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "materialize-ecmwf-ifs-spatial-weather":
        from energy.data import materialize_spatial_ecmwf_ifs_archive

        result = materialize_spatial_ecmwf_ifs_archive(arguments.project_root)
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "build-mpc-weather-snapshots":
        from energy.data import build_mpc_weather_snapshots

        result = build_mpc_weather_snapshots(
            arguments.project_root,
            year=arguments.year,
            source_path=arguments.source_path,
            start=arguments.start,
            end=arguments.end,
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "build-multisite-pv-data":
        from energy.data.multisite import build_multisite_day_ahead_pv

        result = build_multisite_day_ahead_pv(
            arguments.project_root,
            year=arguments.year,
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-pv-multisite-experiment":
        from energy.training.pv_multisite_experiment import (
            MultiSitePVExperimentConfig,
            run_multisite_pv_experiment,
        )

        result = run_multisite_pv_experiment(
            MultiSitePVExperimentConfig(
                project_root=arguments.project_root,
                data_path=arguments.data_path,
                year=arguments.year,
                n_lags=arguments.n_lags,
                cv_splits=arguments.cv_splits,
                catboost_iterations=arguments.catboost_iterations,
                random_state=arguments.random_state,
            )
        )
        print(
            json.dumps(
                {
                    "artifact_dir": str(result.artifact_dir),
                    "metrics": result.metrics.to_dict(orient="records"),
                },
                indent=2,
            )
        )
    elif arguments.command == "build-day-ahead-price-spatial-data":
        from energy.data.day_ahead_price_spatial import build_day_ahead_price_spatial_dataset

        result = build_day_ahead_price_spatial_dataset(
            arguments.project_root,
            start=arguments.start,
            end=arguments.end,
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-day-ahead-price-experiment":
        from energy.training.day_ahead_price_experiment import (
            DayAheadPriceExperimentConfig,
            run_day_ahead_price_experiment,
        )

        result = run_day_ahead_price_experiment(
            DayAheadPriceExperimentConfig(
                project_root=arguments.project_root,
                data_path=arguments.data_path,
                train_end=arguments.train_end,
                test_start=arguments.test_start,
                test_end=arguments.test_end,
                iterations=arguments.iterations,
                random_state=arguments.random_state,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "collect-intraday-price-data":
        from energy.data.intraday_price import collect_intraday_price_data

        result = collect_intraday_price_data(
            arguments.project_root, years=tuple(arguments.years)
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "build-intraday-price-data":
        from energy.data.intraday_price import build_intraday_price_mpc_dataset

        result = build_intraday_price_mpc_dataset(
            arguments.project_root,
            source_path=arguments.source_path,
            feature_version=arguments.feature_version,
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-intraday-price-experiment":
        from energy.training.intraday_price_experiment import (
            IntradayPriceExperimentConfig,
            run_intraday_price_experiment,
        )

        result = run_intraday_price_experiment(
            IntradayPriceExperimentConfig(
                project_root=arguments.project_root,
                data_path=arguments.data_path,
                train_year=arguments.train_year,
                test_year=arguments.test_year,
                test_end=arguments.test_end,
                validation_start=arguments.validation_start,
                feature_version=arguments.feature_version,
                artifact_name=arguments.artifact_name,
                iterations=arguments.iterations,
                random_state=arguments.random_state,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-intraday-price-spatial-weather-experiment":
        from energy.training.intraday_price_spatial_weather_experiment import (
            IntradayPriceSpatialWeatherExperimentConfig,
            run_intraday_price_spatial_weather_experiment,
        )

        result = run_intraday_price_spatial_weather_experiment(
            IntradayPriceSpatialWeatherExperimentConfig(
                project_root=arguments.project_root,
                data_path=arguments.data_path,
                weather_path=arguments.weather_path,
                train_start=arguments.train_start,
                validation_start=arguments.validation_start,
                end=arguments.end,
                correction_weight=arguments.correction_weight,
                iterations=arguments.iterations,
                random_state=arguments.random_state,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-intraday-price-spatial-weather-week-cv":
        from energy.training.intraday_price_spatial_weather_experiment import (
            IntradayPriceSpatialWeatherExperimentConfig,
            run_intraday_price_spatial_weather_week_cv,
        )

        result = run_intraday_price_spatial_weather_week_cv(
            IntradayPriceSpatialWeatherExperimentConfig(
                project_root=arguments.project_root,
                data_path=arguments.data_path,
                weather_path=arguments.weather_path,
                train_start=arguments.train_start,
                end=arguments.end,
                correction_weight=arguments.correction_weight,
                iterations=arguments.iterations,
                random_state=arguments.random_state,
                n_splits=arguments.n_splits,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-residual-bootstrap-experiment":
        from energy.training.residual_bootstrap_experiment import (
            ResidualBootstrapExperimentConfig,
            run_residual_bootstrap_experiment,
        )

        result = run_residual_bootstrap_experiment(
            ResidualBootstrapExperimentConfig(
                project_root=arguments.project_root,
                residual_year=arguments.residual_year,
                test_start=arguments.test_start,
                test_end=arguments.test_end,
                initial_train_days=arguments.initial_train_days,
                block_days=arguments.block_days,
                iterations=arguments.iterations,
                n_scenarios=arguments.n_scenarios,
                random_state=arguments.random_state,
                center_residuals=arguments.center_residuals,
                pv_capacity_kwh_per_hour=arguments.pv_capacity_kwh_per_hour,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-residual-bootstrap-spread-experiment":
        from energy.training.residual_bootstrap_spread_experiment import (
            ResidualBootstrapSpreadExperimentConfig,
            run_residual_bootstrap_spread_experiment,
        )

        result = run_residual_bootstrap_spread_experiment(
            ResidualBootstrapSpreadExperimentConfig(
                project_root=arguments.project_root,
                residual_artifact_name=arguments.residual_artifact_name,
                quantile_artifact_name=arguments.quantile_artifact_name,
                n_scenarios=arguments.n_scenarios,
                random_state=arguments.random_state,
                pv_capacity_kwh_per_hour=arguments.pv_capacity_kwh_per_hour,
                seasonal_bandwidth_days=arguments.seasonal_bandwidth_days,
                global_mixture_weight=arguments.global_mixture_weight,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-quantile-copula-experiment":
        from energy.training.quantile_copula_experiment import (
            QuantileCopulaExperimentConfig,
            run_quantile_copula_experiment,
        )

        result = run_quantile_copula_experiment(
            QuantileCopulaExperimentConfig(
                project_root=arguments.project_root,
                residual_year=arguments.residual_year,
                test_start=arguments.test_start,
                test_end=arguments.test_end,
                initial_train_days=arguments.initial_train_days,
                block_days=arguments.block_days,
                iterations=arguments.iterations,
                n_scenarios=arguments.n_scenarios,
                random_state=arguments.random_state,
                pv_capacity_kwh_per_hour=arguments.pv_capacity_kwh_per_hour,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-quantile-spread-copula-experiment":
        from energy.training.quantile_spread_copula_experiment import (
            QuantileSpreadCopulaExperimentConfig,
            run_quantile_spread_copula_experiment,
        )

        result = run_quantile_spread_copula_experiment(
            QuantileSpreadCopulaExperimentConfig(
                project_root=arguments.project_root,
                source_artifact_name=arguments.source_artifact_name,
                residual_year=arguments.residual_year,
                test_start=arguments.test_start,
                test_end=arguments.test_end,
                iterations=arguments.iterations,
                n_scenarios=arguments.n_scenarios,
                random_state=arguments.random_state,
                pv_capacity_kwh_per_hour=arguments.pv_capacity_kwh_per_hour,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-quantile-spread-economic-backtest":
        from energy.training.quantile_spread_economic_backtest import (
            QuantileSpreadEconomicBacktestConfig,
            run_quantile_spread_economic_backtest,
        )

        result = run_quantile_spread_economic_backtest(
            QuantileSpreadEconomicBacktestConfig(
                project_root=arguments.project_root,
                start=arguments.start,
                end=arguments.end,
                quantile_artifact_name=arguments.quantile_artifact_name,
                generators=tuple(arguments.generators),
                n_scenarios=arguments.n_scenarios,
                cvar_alpha=arguments.cvar_alpha,
                risk_weights=tuple(arguments.risk_weights),
                random_state=arguments.random_state,
                economic_bootstrap_resamples=(
                    arguments.economic_bootstrap_resamples
                ),
                economic_bootstrap_block_days=(
                    arguments.economic_bootstrap_block_days
                ),
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-residual-bootstrap-spread-economic-backtest":
        from energy.training.residual_bootstrap_spread_economic_backtest import (
            ResidualBootstrapSpreadEconomicBacktestConfig,
            run_residual_bootstrap_spread_economic_backtest,
        )

        result = run_residual_bootstrap_spread_economic_backtest(
            ResidualBootstrapSpreadEconomicBacktestConfig(
                project_root=arguments.project_root,
                start=arguments.start,
                end=arguments.end,
                residual_artifact_name=arguments.residual_artifact_name,
                generators=tuple(arguments.generators),
                risk_generators=tuple(arguments.risk_generators),
                seasonal_bandwidth_days=arguments.seasonal_bandwidth_days,
                global_mixture_weight=arguments.global_mixture_weight,
                n_scenarios=arguments.n_scenarios,
                cvar_alpha=arguments.cvar_alpha,
                risk_weights=tuple(arguments.risk_weights),
                random_state=arguments.random_state,
                economic_bootstrap_resamples=(
                    arguments.economic_bootstrap_resamples
                ),
                economic_bootstrap_block_days=(
                    arguments.economic_bootstrap_block_days
                ),
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-cqr-copula-experiment":
        from energy.training.cqr_copula_experiment import (
            CqrCopulaExperimentConfig,
            run_cqr_copula_experiment,
        )

        result = run_cqr_copula_experiment(
            CqrCopulaExperimentConfig(
                project_root=arguments.project_root,
                source_artifact_name=arguments.source_artifact_name,
                n_scenarios=arguments.n_scenarios,
                random_state=arguments.random_state,
                pv_capacity_kwh_per_hour=arguments.pv_capacity_kwh_per_hour,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-temporal-backtest-2025":
        from energy.training.temporal_backtest_2025 import (
            TemporalBacktest2025Config,
            run_temporal_backtest_2025,
        )

        result = run_temporal_backtest_2025(
            TemporalBacktest2025Config(
                project_root=arguments.project_root,
                train_year=arguments.train_year,
                test_year=arguments.test_year,
                n_lags=arguments.n_lags,
                catboost_iterations=arguments.catboost_iterations,
                neural_epochs=arguments.neural_epochs,
                random_state=arguments.random_state,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-economic-backtest":
        from energy.training.economic_backtest import (
            EconomicBacktestConfig,
            run_economic_backtest,
        )

        result = run_economic_backtest(
            EconomicBacktestConfig(
                project_root=arguments.project_root,
                start=arguments.start,
                end=arguments.end,
                artifact_name=arguments.artifact_name,
                correction_weight=arguments.correction_weight,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-dynamic-charge-economic-backtest":
        from energy.optimization.grid_tariff import (
            NO_GRID_TARIFF,
            PFORZHEIM_SLP_2025_GRID_TARIFF,
        )
        from energy.training.dynamic_charge_economic_backtest import (
            DynamicChargeEconomicBacktestConfig,
            run_dynamic_charge_economic_backtest,
        )

        tariff = (
            PFORZHEIM_SLP_2025_GRID_TARIFF
            if arguments.grid_tariff_preset == "pforzheim-slp-2025"
            else NO_GRID_TARIFF
        )
        result = run_dynamic_charge_economic_backtest(
            DynamicChargeEconomicBacktestConfig(
                project_root=arguments.project_root,
                start=arguments.start,
                end=arguments.end,
                artifact_name=arguments.artifact_name,
                correction_weight=arguments.correction_weight,
                pv_mpc_prediction_path=arguments.pv_mpc_prediction_path,
                include_stochastic=arguments.include_stochastic,
                grid_tariff=tariff,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-stochastic-economic-backtest":
        from energy.training.stochastic_economic_backtest import (
            StochasticEconomicBacktestConfig,
            run_stochastic_economic_backtest,
        )

        result = run_stochastic_economic_backtest(
            StochasticEconomicBacktestConfig(
                project_root=arguments.project_root,
                start=arguments.start,
                end=arguments.end,
                residual_library_path=arguments.residual_library_path,
                n_scenarios=arguments.n_scenarios,
                cvar_alpha=arguments.cvar_alpha,
                risk_weights=tuple(arguments.risk_weights),
                center_residuals=arguments.center_residuals,
                pv_capacity_kwh_per_hour=arguments.pv_capacity_kwh_per_hour,
                random_state=arguments.random_state,
                correction_weight=arguments.correction_weight,
                economic_bootstrap_resamples=arguments.economic_bootstrap_resamples,
                economic_bootstrap_block_days=arguments.economic_bootstrap_block_days,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-quantile-copula-economic-backtest":
        from energy.training.quantile_copula_economic_backtest import (
            QuantileCopulaEconomicBacktestConfig,
            run_quantile_copula_economic_backtest,
        )

        result = run_quantile_copula_economic_backtest(
            QuantileCopulaEconomicBacktestConfig(
                project_root=arguments.project_root,
                quantile_artifact_name=arguments.quantile_artifact_name,
                start=arguments.start,
                end=arguments.end,
                n_scenarios=arguments.n_scenarios,
                cvar_alpha=arguments.cvar_alpha,
                risk_weights=tuple(arguments.risk_weights),
                random_state=arguments.random_state,
                economic_bootstrap_resamples=arguments.economic_bootstrap_resamples,
                economic_bootstrap_block_days=arguments.economic_bootstrap_block_days,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-cqr-copula-economic-backtest":
        from energy.training.quantile_copula_economic_backtest import (
            QuantileCopulaEconomicBacktestConfig,
            run_quantile_copula_economic_backtest,
        )

        result = run_quantile_copula_economic_backtest(
            QuantileCopulaEconomicBacktestConfig(
                project_root=arguments.project_root,
                quantile_artifact_name=arguments.quantile_artifact_name,
                quantile_forecast_filename="test_calibrated_quantile_forecasts.parquet",
                strategy_prefix="cqr_copula",
                start=arguments.start,
                end=arguments.end,
                n_scenarios=arguments.n_scenarios,
                cvar_alpha=arguments.cvar_alpha,
                risk_weights=tuple(arguments.risk_weights),
                random_state=arguments.random_state,
                economic_bootstrap_resamples=arguments.economic_bootstrap_resamples,
                economic_bootstrap_block_days=arguments.economic_bootstrap_block_days,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "train-pv-day-ahead":
        # Keep the base data command usable without the optional training stack.
        from energy.training.pv_day_ahead import PVDayAheadTrainingConfig, train_day_ahead_pv

        if arguments.register and not arguments.registered_model_name:
            raise SystemExit("--register requires --registered-model-name")
        if arguments.registered_model_name and not arguments.register:
            raise SystemExit("--registered-model-name requires --register")

        result = train_day_ahead_pv(
            PVDayAheadTrainingConfig(
                project_root=arguments.project_root,
                data_path=arguments.data_path,
                year=arguments.year,
                tracking_uri=arguments.tracking_uri,
                experiment_name=arguments.experiment_name,
                hour_start=arguments.hour_start,
                hour_end=arguments.hour_end,
                min_clear_sky_ghi=arguments.min_clear_sky_ghi,
                random_state=arguments.random_state,
                registered_model_name=arguments.registered_model_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "train-pv-mpc-residual":
        from energy.training.pv_mpc_residual import (
            PVMpcResidualTrainingConfig,
            train_pv_mpc_residual,
        )

        if arguments.register and not arguments.registered_model_name:
            raise SystemExit("--register requires --registered-model-name")
        if arguments.registered_model_name and not arguments.register:
            raise SystemExit("--registered-model-name requires --register")

        result = train_pv_mpc_residual(
            PVMpcResidualTrainingConfig(
                project_root=arguments.project_root,
                data_path=arguments.data_path,
                weather_snapshot_path=arguments.weather_snapshot_path,
                year=arguments.year,
                train_start_date=arguments.train_start_date,
                tracking_uri=arguments.tracking_uri,
                experiment_name=arguments.experiment_name,
                n_lags=arguments.n_lags,
                min_clear_sky_ghi=arguments.min_clear_sky_ghi,
                cv_splits=arguments.cv_splits,
                iterations=arguments.iterations,
                random_state=arguments.random_state,
                updated_weather_feature_mode=arguments.updated_weather_feature_mode,
                registered_model_name=arguments.registered_model_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-pv-mpc-ifs-weather-experiment":
        from energy.training.pv_mpc_ifs_weather_experiment import (
            PVMpcIfsWeatherExperimentConfig,
            run_pv_mpc_ifs_weather_experiment,
        )

        result = run_pv_mpc_ifs_weather_experiment(
            PVMpcIfsWeatherExperimentConfig(
                project_root=arguments.project_root,
                train_start_date=arguments.train_start_date,
                test_start_date=arguments.test_start_date,
                test_end_date=arguments.test_end_date,
                n_lags=arguments.n_lags,
                iterations=arguments.iterations,
                random_state=arguments.random_state,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "run-pv-mpc-weather-refresh-evaluation":
        from energy.training.pv_mpc_weather_refresh_evaluation import (
            PVMpcWeatherRefreshEvaluationConfig,
            run_pv_mpc_weather_refresh_evaluation,
        )

        result = run_pv_mpc_weather_refresh_evaluation(
            PVMpcWeatherRefreshEvaluationConfig(
                project_root=arguments.project_root,
                train_start_date=arguments.train_start_date,
                rolling_start_date=arguments.rolling_start_date,
                selection_end_date=arguments.selection_end_date,
                holdout_start_date=arguments.holdout_start_date,
                test_end_date=arguments.test_end_date,
                n_lags=arguments.n_lags,
                iterations=arguments.iterations,
                random_state=arguments.random_state,
                artifact_name=arguments.artifact_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif arguments.command == "train-intraday-price":
        from energy.training.intraday_price import (
            IntradayPriceTrainingConfig,
            train_intraday_price,
        )

        if arguments.register and not arguments.registered_model_name:
            raise SystemExit("--register requires --registered-model-name")
        if arguments.registered_model_name and not arguments.register:
            raise SystemExit("--registered-model-name requires --register")

        result = train_intraday_price(
            IntradayPriceTrainingConfig(
                project_root=arguments.project_root,
                data_path=arguments.data_path,
                year=arguments.year,
                validation_start=arguments.validation_start,
                correction_weight=arguments.correction_weight,
                iterations=arguments.iterations,
                random_state=arguments.random_state,
                tracking_uri=arguments.tracking_uri,
                experiment_name=arguments.experiment_name,
                registered_model_name=arguments.registered_model_name,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

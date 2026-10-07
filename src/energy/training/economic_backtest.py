"""Frozen 2025 economic replay for the V1 rule-based and deterministic policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from energy.optimization import (
    DayReplayInputs,
    EconomicLedger,
    ReferenceScenario,
    V1_REFERENCE_SCENARIO,
    fit_rule_based_price_shape,
    replay_day_ahead_only_day,
    replay_deterministic_day,
    replay_oracle_day,
    replay_rule_based_day,
)


DA_PRICE_VARIANT = "spatial_weather_catboost"
DA_PV_VARIANT = "day_ahead_catboost"
MPC_PV_VARIANT = "direct_residual_catboost"
INTRADAY_BASELINE_VARIANT = "day_ahead_baseline"
INTRADAY_CHALLENGER_VARIANT = "spread_catboost_huber_depth4"
INTRADAY_BLEND_WEIGHT = 0.70
NIGHT_REFERENCE_LOOKBACK_DAYS = 14


@dataclass(frozen=True)
class EconomicBacktestConfig:
    """Paths and frozen choices for the first economic comparison."""

    project_root: Path
    start: str = "2025-01-01"
    end: str = "2025-09-30"
    artifact_name: str = "economic_backtest_v1"
    price_prediction_path: Path | None = None
    pv_day_ahead_prediction_path: Path | None = None
    pv_mpc_prediction_path: Path | None = None
    intraday_prediction_path: Path | None = None
    correction_weight: float = INTRADAY_BLEND_WEIGHT


@dataclass(frozen=True)
class EconomicBacktestResult:
    artifact_dir: Path
    daily_results: pd.DataFrame
    coverage: pd.DataFrame
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "summary": self.summary,
        }


def run_economic_backtest(config: EconomicBacktestConfig) -> EconomicBacktestResult:
    """Replay the fixed 2025 forecast artifacts with one common economic ledger.

    The function never refits or selects a forecasting model.  It only combines
    artifacts whose configurations were fixed before the 2025 holdout.
    """

    if not 0.0 <= config.correction_weight <= 1.0:
        raise ValueError("correction_weight must lie in [0, 1]")
    root = config.project_root.expanduser().resolve()
    start, end = _validate_date_window(config.start, config.end)
    scenario = V1_REFERENCE_SCENARIO
    history = _read_canonical(root, 2024)
    test = _read_canonical(root, 2025)
    intraday_actual = _read_intraday_actual(root)
    test = test.merge(intraday_actual, on="valid_time_utc", how="left", validate="one_to_one")
    price_predictions = _read_price_predictions(config, root)
    pv_day_ahead_predictions = _read_pv_day_ahead_predictions(config, root)
    pv_mpc_predictions = _read_pv_mpc_predictions(config, root)
    intraday_predictions = _read_intraday_predictions(config, root)
    night_references = _night_reference_prices(root)
    price_shape = fit_rule_based_price_shape(
        history,
        price_column="day_ahead_price_eur_per_mwh",
        date_column="delivery_date_local",
        hour_column="hour_local",
    )

    inputs, coverage = _assemble_replay_inputs(
        start=start,
        end=end,
        scenario=scenario,
        test=test,
        price_predictions=price_predictions,
        pv_day_ahead_predictions=pv_day_ahead_predictions,
        pv_mpc_predictions=pv_mpc_predictions,
        intraday_predictions=intraday_predictions,
        night_references=night_references,
        correction_weight=config.correction_weight,
    )
    if not inputs:
        raise ValueError("No complete regular delivery days remain for the economic replay")

    daily_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    for day_inputs in inputs:
        rule = replay_rule_based_day(
            inputs=day_inputs,
            scenario=scenario,
            price_shape=price_shape,
        )
        day_ahead_only = replay_day_ahead_only_day(inputs=day_inputs, scenario=scenario)
        deterministic = replay_deterministic_day(inputs=day_inputs, scenario=scenario)
        oracle = replay_oracle_day(inputs=day_inputs, scenario=scenario)
        daily_rows.extend(
            (
                _ledger_row("rule_based", day_inputs, rule.ledger, rule.rule_plan.load_kwh, rule.rule_plan.discharge_kwh, rule.rule_plan.day_ahead_position_kwh, scenario),
                _ledger_row("day_ahead_only", day_inputs, day_ahead_only.ledger, day_ahead_only.day_ahead_plan.load_kwh, day_ahead_only.day_ahead_plan.discharge_kwh, day_ahead_only.day_ahead_plan.day_ahead_position_kwh, scenario),
                _ledger_row("deterministic", day_inputs, deterministic.ledger, deterministic.executed_load_kwh, deterministic.executed_discharge_kwh, deterministic.day_ahead_plan.day_ahead_position_kwh, scenario),
                _ledger_row("oracle", day_inputs, oracle.ledger, oracle.oracle_plan.actual_load_kwh, oracle.oracle_plan.actual_discharge_kwh, oracle.oracle_plan.day_ahead_position_kwh, scenario),
            )
        )
        hourly_rows.extend(
            _hourly_rows("rule_based", day_inputs, rule.rule_plan.load_kwh, rule.rule_plan.discharge_kwh, rule.rule_plan.day_ahead_position_kwh)
        )
        hourly_rows.extend(
            _hourly_rows("day_ahead_only", day_inputs, day_ahead_only.day_ahead_plan.load_kwh, day_ahead_only.day_ahead_plan.discharge_kwh, day_ahead_only.day_ahead_plan.day_ahead_position_kwh)
        )
        hourly_rows.extend(
            _hourly_rows("deterministic", day_inputs, deterministic.executed_load_kwh, deterministic.executed_discharge_kwh, deterministic.day_ahead_plan.day_ahead_position_kwh)
        )
        hourly_rows.extend(
            _hourly_rows("oracle", day_inputs, oracle.oracle_plan.actual_load_kwh, oracle.oracle_plan.actual_discharge_kwh, oracle.oracle_plan.day_ahead_position_kwh)
        )

    daily_results = pd.DataFrame(daily_rows).sort_values(["delivery_date_local", "strategy"])
    summary = _summary(daily_results, scenario, config)
    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    daily_results.to_csv(artifact_dir / "daily_results.csv", index=False)
    pd.DataFrame(hourly_rows).to_parquet(artifact_dir / "hourly_decisions.parquet", index=False)
    coverage.to_csv(artifact_dir / "coverage.csv", index=False)
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "start": config.start,
                "end": config.end,
                "scenario": {
                    "daily_load_kwh": scenario.daily_load_kwh,
                    "working_hours": list(scenario.working_hours),
                    "load_min_kwh_per_hour": scenario.load_min_kwh_per_hour,
                    "load_max_kwh_per_hour": scenario.load_max_kwh_per_hour,
                    "pv_profile_scale": scenario.pv_profile_scale,
                    "battery_available_energy_kwh": scenario.battery_available_energy_kwh,
                    "battery_max_discharge_kwh_per_hour": scenario.battery_max_discharge_kwh_per_hour,
                    "charge_efficiency": scenario.charge_efficiency,
                    "discharge_efficiency": scenario.discharge_efficiency,
                    "degradation_eur_per_kwh": scenario.degradation_eur_per_kwh,
                },
                "oracle_policy": {
                    "information": "factual PV and factual day-ahead prices only",
                    "intraday_price_used": False,
                    "nomination_equals_physical_schedule": True,
                    "deliberate_imbalance_allowed": False,
                },
                "frozen_forecasts": {
                    "day_ahead_price_variant": DA_PRICE_VARIANT,
                    "day_ahead_pv_variant": DA_PV_VARIANT,
                    "mpc_pv_variant": MPC_PV_VARIANT,
                    "intraday_baseline_variant": INTRADAY_BASELINE_VARIANT,
                    "intraday_challenger_variant": INTRADAY_CHALLENGER_VARIANT,
                    "intraday_lead_1_correction_weight": config.correction_weight,
                },
                "selection_contract": (
                    "Forecast models and the 0.70 intraday blend were selected without "
                    "using 2025 economic outcomes. The replay only evaluates the frozen policy."
                ),
                "excluded_days": int((coverage["status"] != "used").sum()),
            },
            indent=2,
        )
    )
    return EconomicBacktestResult(
        artifact_dir=artifact_dir,
        daily_results=daily_results,
        coverage=coverage,
        summary=summary,
    )


def _assemble_replay_inputs(
    *,
    start: date,
    end: date,
    scenario: ReferenceScenario,
    test: pd.DataFrame,
    price_predictions: pd.DataFrame,
    pv_day_ahead_predictions: pd.DataFrame,
    pv_mpc_predictions: pd.DataFrame,
    intraday_predictions: pd.DataFrame,
    night_references: dict[date, float],
    correction_weight: float,
) -> tuple[list[DayReplayInputs], pd.DataFrame]:
    inputs: list[DayReplayInputs] = []
    coverage_rows: list[dict[str, object]] = []
    by_day_test = {day: group for day, group in test.groupby("delivery_date_local", sort=False)}
    by_day_price = {day: group for day, group in price_predictions.groupby("delivery_date_local", sort=False)}
    by_day_pv = {day: group for day, group in pv_day_ahead_predictions.groupby("delivery_date_local", sort=False)}
    by_day_mpc_pv = {day: group for day, group in pv_mpc_predictions.groupby("delivery_date_local", sort=False)}
    by_day_intraday = {day: group for day, group in intraday_predictions.groupby("delivery_date_local", sort=False)}

    current = start
    while current <= end:
        day = current
        day_test = by_day_test.get(day)
        day_price = by_day_price.get(day)
        day_pv = by_day_pv.get(day)
        day_mpc_pv = by_day_mpc_pv.get(day)
        day_intraday = by_day_intraday.get(day)
        reason = _missing_day_reason(day, day_test, day_price, day_pv, day_intraday, night_references, scenario)
        if reason is not None:
            coverage_rows.append({"delivery_date_local": day.isoformat(), "status": reason})
            current += timedelta(days=1)
            continue
        assert day_test is not None and day_price is not None and day_pv is not None and day_intraday is not None
        day_test = day_test.sort_values("hour_local")
        actual_pv = scenario.scale_pv_energy(day_test["pv_energy_kwh"].to_numpy(dtype=float))
        actual_da_price = day_test["day_ahead_price_eur_per_mwh"].to_numpy(dtype=float)
        actual_id_price = day_test["actual_intraday_price_eur_per_mwh"].to_numpy(dtype=float)
        da_price_forecast = _hour_vector(day_price, "prediction_eur_per_mwh")
        da_pv_forecast = np.zeros(24, dtype=float)
        da_pv_forecast[list(scenario.working_hours)] = (
            _hour_vector(day_pv, "prediction_w", required_hours=scenario.working_hours) / 1_000.0 * scenario.pv_profile_scale
        )
        mpc_pv_forecast = np.tile(da_pv_forecast, (24, 1))
        if day_mpc_pv is not None:
            _apply_pv_mpc_predictions(mpc_pv_forecast, day_mpc_pv, scenario)
        mpc_intraday_forecast = np.tile(actual_da_price, (24, 1))
        _apply_intraday_predictions(
            mpc_intraday_forecast,
            day_intraday,
            correction_weight=correction_weight,
        )
        inputs.append(
            DayReplayInputs(
                delivery_day=day,
                day_ahead_pv_forecast_kwh=da_pv_forecast,
                day_ahead_price_forecast_eur_per_mwh=da_price_forecast,
                actual_pv_kwh=actual_pv,
                actual_day_ahead_price_eur_per_mwh=actual_da_price,
                actual_intraday_price_eur_per_mwh=actual_id_price,
                mpc_pv_forecast_kwh=mpc_pv_forecast,
                mpc_intraday_price_forecast_eur_per_mwh=mpc_intraday_forecast,
                night_reference_price_eur_per_mwh=night_references[day],
            )
        )
        coverage_rows.append({"delivery_date_local": day.isoformat(), "status": "used"})
        current += timedelta(days=1)
    return inputs, pd.DataFrame(coverage_rows)


def _missing_day_reason(
    day: date,
    day_test: pd.DataFrame | None,
    day_price: pd.DataFrame | None,
    day_pv: pd.DataFrame | None,
    day_intraday: pd.DataFrame | None,
    night_references: dict[date, float],
    scenario: ReferenceScenario,
) -> str | None:
    if day_test is None or len(day_test) != 24 or set(day_test["hour_local"]) != set(range(24)):
        return "non_regular_or_missing_canonical_day"
    needed = ("pv_energy_kwh", "day_ahead_price_eur_per_mwh", "actual_intraday_price_eur_per_mwh")
    if day_test.loc[:, list(needed)].isna().any().any():
        return "missing_realised_value"
    if day_price is None or len(day_price) != 24 or set(day_price["hour_local"]) != set(range(24)):
        return "missing_day_ahead_price_forecast"
    if day_pv is None or set(scenario.working_hours).difference(set(day_pv["hour_local"])):
        return "missing_day_ahead_pv_forecast"
    if day_intraday is None:
        return "missing_intraday_forecast"
    required_next_hour_decisions = set(range(23))
    if required_next_hour_decisions.difference(set(day_intraday["decision_hour_local"])):
        return "missing_intraday_next_hour_forecast"
    if day not in night_references:
        return "missing_previous_night_reference"
    return None


def _apply_pv_mpc_predictions(matrix: np.ndarray, rows: pd.DataFrame, scenario: ReferenceScenario) -> None:
    for row in rows.itertuples(index=False):
        decision = int(row.decision_hour_local)
        target = decision + int(row.lead_hours)
        execution_hour = decision + 1
        if decision < 0 or target >= 24 or execution_hour >= 24:
            continue
        prediction_kwh = max(float(row.prediction_w), 0.0) / 1_000.0 * scenario.pv_profile_scale
        matrix[execution_hour, target] = prediction_kwh


def _apply_intraday_predictions(matrix: np.ndarray, rows: pd.DataFrame, *, correction_weight: float) -> None:
    index = rows.set_index(["decision_hour_local", "lead_hours"])
    for decision in range(23):
        target = decision + 1
        key = (decision, 1)
        if key not in index.index:
            raise ValueError(f"Missing intraday next-hour prediction for decision {decision}")
        row = index.loc[key]
        if isinstance(row, pd.DataFrame):
            raise ValueError("Intraday prediction rows must be unique by decision and lead")
        baseline = float(row["baseline_prediction_eur_per_mwh"])
        challenger = float(row["challenger_prediction_eur_per_mwh"])
        matrix[target, target] = baseline + correction_weight * (challenger - baseline)


def _hour_vector(
    frame: pd.DataFrame,
    value_column: str,
    *,
    required_hours: tuple[int, ...] | range | None = None,
) -> np.ndarray:
    hours = tuple(range(24)) if required_hours is None else tuple(required_hours)
    indexed = frame.set_index("hour_local")[value_column]
    if indexed.index.duplicated().any() or set(hours).difference(indexed.index):
        raise ValueError(f"Missing or duplicate hour in {value_column}")
    values = indexed.loc[list(hours)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{value_column} contains non-finite values")
    return values


def _read_canonical(root: Path, year: int) -> pd.DataFrame:
    path = root / "data" / "curated" / f"canonical_hourly_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Canonical hourly data is missing: {path}")
    frame = pd.read_parquet(path)
    frame["valid_time_utc"] = pd.to_datetime(frame["valid_time_utc"], utc=True)
    frame["delivery_date_local"] = pd.to_datetime(frame["delivery_date_local"]).dt.date
    return frame


def _read_intraday_actual(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "prices_intraday_continuous_de_lu_hourly.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Intraday settlement proxy is missing: {path}")
    frame = pd.read_csv(path, usecols=["valid_time_utc", "intraday_continuous_avg_price_eur_per_mwh"])
    frame["valid_time_utc"] = pd.to_datetime(frame["valid_time_utc"], utc=True)
    return frame.rename(columns={"intraday_continuous_avg_price_eur_per_mwh": "actual_intraday_price_eur_per_mwh"})


def _night_reference_prices(root: Path) -> dict[date, float]:
    """Return a causal 14-night rolling battery-recharge benchmark.

    Each raw night uses local price labels 22:00 and 23:00 plus 00:00--06:00.
    Equal weighting is equivalent to buying 5 kWh in each of nine labels, or
    45 kWh in total.  A raw nightly mean is floored at zero before averaging
    the latest 14 completed nights.  For delivery day D, the newest included
    night ends on D-1, so the benchmark is known before the D-1 day-ahead
    decision.
    """

    path = root / "data" / "processed" / "prices_day_ahead_de_lu.csv.gz"
    frame = pd.read_csv(path)
    timestamps = pd.to_datetime(frame["valid_time_utc"], utc=True).dt.tz_convert("Europe/Berlin")
    frame["delivery_date_local"] = timestamps.dt.date
    frame["hour_local"] = timestamps.dt.hour
    night_hours = set(V1_REFERENCE_SCENARIO.night_hours)
    frame = frame.loc[frame["hour_local"].isin(night_hours)].copy()
    frame["night_end_date_local"] = frame["delivery_date_local"]
    late = frame["hour_local"] >= 22
    frame.loc[late, "night_end_date_local"] = frame.loc[
        late, "delivery_date_local"
    ].map(lambda value: value + timedelta(days=1))
    frame["reference_delivery_date_local"] = frame["night_end_date_local"].map(
        lambda value: value + timedelta(days=1)
    )
    nightly = frame.groupby("reference_delivery_date_local")[
        "day_ahead_price_eur_per_mwh"
    ].mean().sort_index()
    nightly = nightly.clip(lower=0.0)
    rolling = nightly.rolling(
        window=NIGHT_REFERENCE_LOOKBACK_DAYS,
        min_periods=NIGHT_REFERENCE_LOOKBACK_DAYS,
    ).mean()
    return {
        day: float(value)
        for day, value in rolling.items()
        if np.isfinite(value)
    }


def _read_price_predictions(config: EconomicBacktestConfig, root: Path) -> pd.DataFrame:
    path = config.price_prediction_path or root / "artifacts" / "experiments" / "day_ahead_price_spatial_v1" / "hourly_predictions.parquet"
    frame = pd.read_parquet(path)
    result = frame.loc[frame["variant"] == DA_PRICE_VARIANT, ["delivery_date_local", "hour_local", "prediction_eur_per_mwh"]].copy()
    return _normalise_prediction_dates(result)


def _read_pv_day_ahead_predictions(config: EconomicBacktestConfig, root: Path) -> pd.DataFrame:
    path = config.pv_day_ahead_prediction_path or root / "artifacts" / "experiments" / "temporal_backtest_v1" / "2024_to_2025" / "pv_day_ahead_hourly_predictions.parquet"
    frame = pd.read_parquet(path)
    result = frame.loc[frame["variant"] == DA_PV_VARIANT, ["delivery_date_local", "hour_local", "prediction_w"]].copy()
    return _normalise_prediction_dates(result)


def _read_pv_mpc_predictions(config: EconomicBacktestConfig, root: Path) -> pd.DataFrame:
    path = config.pv_mpc_prediction_path or root / "artifacts" / "experiments" / "temporal_backtest_v1" / "2024_to_2025" / "pv_mpc_trajectory_predictions.parquet"
    frame = pd.read_parquet(path)
    result = frame.loc[
        frame["variant"] == MPC_PV_VARIANT,
        ["delivery_date_local", "decision_hour_local", "lead_hours", "prediction_w"],
    ].copy()
    return _normalise_prediction_dates(result)


def _read_intraday_predictions(config: EconomicBacktestConfig, root: Path) -> pd.DataFrame:
    path = config.intraday_prediction_path or root / "artifacts" / "experiments" / "intraday_price_v2_features" / "predictions.parquet"
    frame = pd.read_parquet(path)
    keys = ["delivery_date_local", "decision_hour_local", "lead_hours"]
    baseline = frame.loc[frame["variant"] == INTRADAY_BASELINE_VARIANT, [*keys, "prediction_eur_per_mwh"]].rename(columns={"prediction_eur_per_mwh": "baseline_prediction_eur_per_mwh"})
    challenger = frame.loc[frame["variant"] == INTRADAY_CHALLENGER_VARIANT, [*keys, "prediction_eur_per_mwh"]].rename(columns={"prediction_eur_per_mwh": "challenger_prediction_eur_per_mwh"})
    result = baseline.merge(challenger, on=keys, how="inner", validate="one_to_one")
    return _normalise_prediction_dates(result)


def _normalise_prediction_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["delivery_date_local"] = pd.to_datetime(result["delivery_date_local"]).dt.date
    return result


def _ledger_row(
    strategy: str,
    inputs: DayReplayInputs,
    ledger: EconomicLedger,
    load: np.ndarray,
    discharge: np.ndarray,
    position: np.ndarray,
    scenario: ReferenceScenario,
) -> dict[str, object]:
    return {
        "delivery_date_local": inputs.delivery_day.isoformat(),
        "strategy": strategy,
        "total_cost_eur": float(ledger.total_cost_eur),
        "day_ahead_cost_eur": float(ledger.day_ahead_cost_eur),
        "intraday_deviation_cost_eur": float(ledger.intraday_deviation_cost_eur),
        "battery_cost_eur": float(ledger.battery_cost_eur),
        "day_ahead_position_kwh": float(position.sum()),
        "intraday_deviation_kwh": float(ledger.actual_deviation_kwh.sum()),
        "actual_pv_kwh": float(inputs.actual_pv_kwh.sum()),
        "battery_discharge_kwh": float(discharge.sum()),
        "equivalent_full_cycles": float(discharge.sum() / scenario.battery_available_energy_kwh),
        "production_load_kwh": float(load.sum()),
    }


def _hourly_rows(
    strategy: str,
    inputs: DayReplayInputs,
    load: np.ndarray,
    discharge: np.ndarray,
    position: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "delivery_date_local": inputs.delivery_day.isoformat(),
            "strategy": strategy,
            "hour_local": hour,
            "load_kwh": float(load[hour]),
            "battery_discharge_kwh": float(discharge[hour]),
            "day_ahead_position_kwh": float(position[hour]),
            "actual_pv_kwh": float(inputs.actual_pv_kwh[hour]),
            "actual_day_ahead_price_eur_per_mwh": float(inputs.actual_day_ahead_price_eur_per_mwh[hour]),
            "actual_intraday_price_eur_per_mwh": float(inputs.actual_intraday_price_eur_per_mwh[hour]),
        }
        for hour in range(24)
    ]


def _summary(daily_results: pd.DataFrame, scenario: ReferenceScenario, config: EconomicBacktestConfig) -> dict[str, object]:
    totals = daily_results.groupby("strategy")[["total_cost_eur", "day_ahead_cost_eur", "intraday_deviation_cost_eur", "battery_cost_eur"]].sum()
    rule = daily_results.loc[daily_results["strategy"] == "rule_based"].set_index("delivery_date_local")
    day_ahead_only = daily_results.loc[daily_results["strategy"] == "day_ahead_only"].set_index("delivery_date_local")
    deterministic = daily_results.loc[daily_results["strategy"] == "deterministic"].set_index("delivery_date_local")
    delta = rule["total_cost_eur"] - deterministic["total_cost_eur"]
    mpc_delta = day_ahead_only["total_cost_eur"] - deterministic["total_cost_eur"]
    rule_total = float(totals.loc["rule_based", "total_cost_eur"])
    day_ahead_only_total = float(totals.loc["day_ahead_only", "total_cost_eur"])
    deterministic_total = float(totals.loc["deterministic", "total_cost_eur"])
    oracle_total = float(totals.loc["oracle", "total_cost_eur"])
    total_saving = rule_total - deterministic_total
    day_ahead_effect = rule_total - day_ahead_only_total
    mpc_effect = day_ahead_only_total - deterministic_total
    return {
        "delivery_days": int(len(rule)),
        "total_load_kwh": float(len(rule) * scenario.daily_load_kwh),
        "strategy_costs_eur": {
            strategy: {component: float(value) for component, value in row.items()}
            for strategy, row in totals.iterrows()
        },
        "deterministic_saving_vs_rule_eur": total_saving,
        "deterministic_saving_vs_rule_percent": total_saving / abs(rule_total) * 100.0 if rule_total else None,
        "day_ahead_effect_vs_rule_eur": day_ahead_effect,
        "mpc_incremental_effect_vs_day_ahead_only_eur": mpc_effect,
        "day_ahead_effect_share_of_total_saving": day_ahead_effect / total_saving if total_saving else None,
        "mpc_effect_share_of_total_saving": mpc_effect / total_saving if total_saving else None,
        "median_daily_saving_eur": float(delta.median()),
        "share_days_deterministic_cheaper": float((delta > 0).mean()),
        "median_daily_mpc_saving_eur": float(mpc_delta.median()),
        "share_days_mpc_cheaper_than_day_ahead_only": float((mpc_delta > 0).mean()),
        "deterministic_gap_to_oracle_eur": deterministic_total - oracle_total,
        "rule_based_gap_to_oracle_eur": rule_total - oracle_total,
        "intraday_blend_weight": config.correction_weight,
    }


def _validate_date_window(start: str, end: str) -> tuple[date, date]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    if end_date < start_date:
        raise ValueError("end must be on or after start")
    return start_date, end_date

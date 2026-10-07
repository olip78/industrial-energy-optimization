"""Economic replay for conditional-quantile empirical-copula scenarios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy.optimization import (
    V1_REFERENCE_SCENARIO,
    evaluate_actual_day,
    replay_day_ahead_only_day,
    replay_deterministic_day,
    replay_oracle_day,
    solve_stochastic_day_ahead,
)
from energy.training.quantile_copula_experiment import (
    DEFAULT_QUANTILES,
    _quantile_column,
)
from energy.training.stochastic_economic_backtest import (
    StochasticEconomicBacktestConfig,
    _append_strategy,
    _economic_summary,
    _expand_plan,
    _load_replay_inputs,
)
from energy.training.economic_backtest import EconomicBacktestConfig, _validate_date_window
from energy.uncertainty import QuantileEmpiricalCopula


@dataclass(frozen=True)
class QuantileCopulaEconomicBacktestConfig:
    project_root: Path
    start: str = "2025-01-01"
    end: str = "2025-09-30"
    quantile_artifact_name: str = "quantile_copula_v1"
    quantile_forecast_filename: str = "test_quantile_forecasts.parquet"
    strategy_prefix: str = "quantile_copula"
    n_scenarios: int = 500
    cvar_alpha: float = 0.95
    risk_weights: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.0)
    pv_capacity_kwh_per_hour: float = 10.0
    random_state: int = 42
    correction_weight: float = 0.70
    economic_bootstrap_resamples: int = 5_000
    economic_bootstrap_block_days: int = 7
    artifact_name: str = "quantile_copula_economic_backtest_v1"


@dataclass(frozen=True)
class QuantileCopulaEconomicBacktestResult:
    artifact_dir: Path
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"artifact_dir": str(self.artifact_dir), "summary": self.summary}


def run_quantile_copula_economic_backtest(
    config: QuantileCopulaEconomicBacktestConfig,
) -> QuantileCopulaEconomicBacktestResult:
    """Replay quantile-copula SAA/CVaR policies on factual 2025 outcomes."""

    _validate_config(config)
    root = config.project_root.expanduser().resolve()
    start, end = _validate_date_window(config.start, config.end)
    scenario = V1_REFERENCE_SCENARIO
    base_config = EconomicBacktestConfig(
        project_root=root,
        start=config.start,
        end=config.end,
        correction_weight=config.correction_weight,
    )
    inputs, coverage = _load_replay_inputs(
        root=root,
        start=start,
        end=end,
        scenario=scenario,
        base_config=base_config,
    )
    source_dir = root / "artifacts" / "experiments" / config.quantile_artifact_name
    copula_library = pd.read_parquet(source_dir / "empirical_copula_library.parquet")
    quantile_forecasts = pd.read_parquet(source_dir / config.quantile_forecast_filename)
    quantile_forecasts["delivery_date_local"] = pd.to_datetime(
        quantile_forecasts["delivery_date_local"]
    ).dt.date
    by_day_quantiles = {
        day: frame.sort_values("hour_local")
        for day, frame in quantile_forecasts.groupby("delivery_date_local", sort=False)
    }
    generator = QuantileEmpiricalCopula(
        copula_library,
        quantile_levels=DEFAULT_QUANTILES,
        hours_local=scenario.working_hours,
    )
    pv_columns = [_quantile_column("pv", value) for value in DEFAULT_QUANTILES]
    price_columns = [_quantile_column("price", value) for value in DEFAULT_QUANTILES]
    working = np.asarray(scenario.working_hours, dtype=int)
    lower, upper = scenario.load_bounds()
    allowed = scenario.discharge_allowed()
    daily_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for day_index, day_inputs in enumerate(inputs):
        if day_inputs.delivery_day not in by_day_quantiles:
            raise ValueError(
                f"Missing quantile forecasts for {day_inputs.delivery_day}"
            )
        day_quantiles = by_day_quantiles[day_inputs.delivery_day]
        baseline = replay_day_ahead_only_day(inputs=day_inputs, scenario=scenario)
        mpc = replay_deterministic_day(inputs=day_inputs, scenario=scenario)
        oracle = replay_oracle_day(inputs=day_inputs, scenario=scenario)
        _append_strategy(
            daily_rows,
            hourly_rows,
            strategy="deterministic_day_ahead_only",
            day_inputs=day_inputs,
            ledger=baseline.ledger,
            load=baseline.day_ahead_plan.load_kwh,
            discharge=baseline.day_ahead_plan.discharge_kwh,
            position=baseline.day_ahead_plan.day_ahead_position_kwh,
            scenario=scenario,
        )
        _append_strategy(
            daily_rows,
            hourly_rows,
            strategy="deterministic_mpc_context",
            day_inputs=day_inputs,
            ledger=mpc.ledger,
            load=mpc.executed_load_kwh,
            discharge=mpc.executed_discharge_kwh,
            position=mpc.day_ahead_plan.day_ahead_position_kwh,
            scenario=scenario,
        )
        _append_strategy(
            daily_rows,
            hourly_rows,
            strategy="oracle_context",
            day_inputs=day_inputs,
            ledger=oracle.ledger,
            load=oracle.oracle_plan.actual_load_kwh,
            discharge=oracle.oracle_plan.actual_discharge_kwh,
            position=oracle.oracle_plan.day_ahead_position_kwh,
            scenario=scenario,
        )
        batch = generator.sample(
            pv_quantiles_kwh=day_quantiles[pv_columns].to_numpy(dtype=float),
            price_quantiles_eur_per_mwh=day_quantiles[price_columns].to_numpy(dtype=float),
            n_scenarios=config.n_scenarios,
            random_state=config.random_state + day_index,
            as_of_date=day_inputs.delivery_day,
            pv_capacity_kwh_per_hour=config.pv_capacity_kwh_per_hour,
        )
        # Keep the submitted-position centre identical to the deterministic
        # and residual-bootstrap comparisons. Quantile marginals influence the
        # stochastic objective without silently replacing the frozen point PV
        # model used to calculate q_DA.
        point_pv = day_inputs.day_ahead_pv_forecast_kwh[working]
        for risk_weight in (0.0, *config.risk_weights):
            plan = solve_stochastic_day_ahead(
                pv_scenarios_kwh=batch.pv_kwh,
                price_scenarios_eur_per_mwh=batch.day_ahead_price_eur_per_mwh,
                point_pv_forecast_kwh=point_pv,
                load_min_kwh=lower[working],
                load_max_kwh=upper[working],
                required_load_kwh=scenario.daily_load_kwh,
                battery=scenario.battery(day_inputs.night_reference_price_eur_per_mwh),
                discharge_allowed=allowed[working],
                cvar_alpha=config.cvar_alpha,
                risk_weight=risk_weight,
            )
            load, discharge, position = _expand_plan(plan, working)
            ledger = evaluate_actual_day(
                day_ahead_position_kwh=position,
                day_ahead_price_eur_per_mwh=day_inputs.actual_day_ahead_price_eur_per_mwh,
                actual_load_kwh=load,
                actual_pv_kwh=day_inputs.actual_pv_kwh,
                actual_discharge_kwh=discharge,
                actual_intraday_price_eur_per_mwh=day_inputs.actual_intraday_price_eur_per_mwh,
                battery=scenario.battery(day_inputs.night_reference_price_eur_per_mwh),
            )
            strategy = _strategy_name(risk_weight, config.strategy_prefix)
            _append_strategy(
                daily_rows,
                hourly_rows,
                strategy=strategy,
                day_inputs=day_inputs,
                ledger=ledger,
                load=load,
                discharge=discharge,
                position=position,
                scenario=scenario,
            )
            diagnostics.append(
                {
                    "delivery_date_local": day_inputs.delivery_day.isoformat(),
                    "strategy": strategy,
                    "risk_weight": risk_weight,
                    "predicted_expected_cost_eur": plan.expected_cost_eur,
                    "predicted_cvar_cost_eur": plan.cvar_cost_eur,
                    "predicted_objective_eur": plan.objective_eur,
                    "unique_source_copula_days": len(set(batch.source_copula_days)),
                }
            )

    daily = pd.DataFrame(daily_rows).sort_values(["delivery_date_local", "strategy"])
    summary_config = StochasticEconomicBacktestConfig(
        project_root=root,
        start=config.start,
        end=config.end,
        n_scenarios=config.n_scenarios,
        cvar_alpha=config.cvar_alpha,
        risk_weights=config.risk_weights,
        pv_capacity_kwh_per_hour=config.pv_capacity_kwh_per_hour,
        random_state=config.random_state,
        correction_weight=config.correction_weight,
        economic_bootstrap_resamples=config.economic_bootstrap_resamples,
        economic_bootstrap_block_days=config.economic_bootstrap_block_days,
    )
    summary = _economic_summary(
        daily,
        scenario=scenario,
        alpha=config.cvar_alpha,
        config=summary_config,
        residual_days=len(generator.library_days),
        candidate_prefix=f"{config.strategy_prefix}_",
        library_days_label="copula_library_days",
        include_residual_centering=False,
    )
    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(artifact_dir / "daily_results.csv", index=False)
    pd.DataFrame(hourly_rows).to_parquet(
        artifact_dir / "hourly_decisions.parquet", index=False
    )
    pd.DataFrame(diagnostics).to_csv(
        artifact_dir / "scenario_objectives.csv", index=False
    )
    coverage.to_csv(artifact_dir / "coverage.csv", index=False)
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "start": config.start,
                "end": config.end,
                "quantile_artifact_name": config.quantile_artifact_name,
                "quantile_forecast_filename": config.quantile_forecast_filename,
                "strategy_prefix": config.strategy_prefix,
                "quantile_levels": list(DEFAULT_QUANTILES),
                "n_scenarios": config.n_scenarios,
                "cvar_alpha": config.cvar_alpha,
                "risk_weights": list(config.risk_weights),
                "random_state": config.random_state,
                "point_position_contract": (
                    "day-ahead position uses the frozen RMSE point PV forecast, "
                    "matching deterministic and residual-bootstrap comparisons"
                ),
                "market_approximation": (
                    "The sampled day-ahead price is also the intraday price proxy "
                    "inside optimization; factual replay uses observed settlement values"
                ),
            },
            indent=2,
        )
    )
    return QuantileCopulaEconomicBacktestResult(artifact_dir=artifact_dir, summary=summary)


def _strategy_name(risk_weight: float, prefix: str) -> str:
    if risk_weight == 0:
        return f"{prefix}_saa"
    return f"{prefix}_cvar_lambda_{risk_weight:g}".replace(".", "_")


def _validate_config(config: QuantileCopulaEconomicBacktestConfig) -> None:
    if config.n_scenarios < 2:
        raise ValueError("n_scenarios must be at least two")
    if not 0 < config.cvar_alpha < 1:
        raise ValueError("cvar_alpha must lie inside (0, 1)")
    if any(weight <= 0 or not np.isfinite(weight) for weight in config.risk_weights):
        raise ValueError("risk_weights must be finite and positive")
    if len(set(config.risk_weights)) != len(config.risk_weights):
        raise ValueError("risk_weights must be unique")
    if config.economic_bootstrap_resamples < 100:
        raise ValueError("economic_bootstrap_resamples must be at least 100")
    if config.economic_bootstrap_block_days < 1:
        raise ValueError("economic_bootstrap_block_days must be positive")
    if not config.strategy_prefix or not config.strategy_prefix.replace("_", "").isalnum():
        raise ValueError("strategy_prefix must be a non-empty identifier")

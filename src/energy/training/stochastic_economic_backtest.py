"""Economic replay of residual-bootstrap SAA and CVaR day-ahead schedules."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy.optimization import (
    ReferenceScenario,
    V1_REFERENCE_SCENARIO,
    evaluate_actual_day,
    replay_day_ahead_only_day,
    replay_deterministic_day,
    replay_oracle_day,
    solve_stochastic_day_ahead,
)
from energy.training.economic_backtest import (
    EconomicBacktestConfig,
    _assemble_replay_inputs,
    _hourly_rows,
    _ledger_row,
    _night_reference_prices,
    _read_canonical,
    _read_intraday_actual,
    _read_intraday_predictions,
    _read_price_predictions,
    _read_pv_day_ahead_predictions,
    _read_pv_mpc_predictions,
    _validate_date_window,
)
from energy.uncertainty import JointResidualBootstrap


@dataclass(frozen=True)
class StochasticEconomicBacktestConfig:
    """Frozen choices for the first stochastic day-ahead economic replay."""

    project_root: Path
    start: str = "2025-01-01"
    end: str = "2025-09-30"
    residual_library_path: Path | None = None
    n_scenarios: int = 500
    cvar_alpha: float = 0.95
    risk_weights: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.0)
    center_residuals: bool = True
    pv_capacity_kwh_per_hour: float = 10.0
    random_state: int = 42
    correction_weight: float = 0.70
    economic_bootstrap_resamples: int = 5_000
    economic_bootstrap_block_days: int = 7
    artifact_name: str = "stochastic_economic_backtest_v1"


@dataclass(frozen=True)
class StochasticEconomicBacktestResult:
    artifact_dir: Path
    daily_results: pd.DataFrame
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"artifact_dir": str(self.artifact_dir), "summary": self.summary}


def run_stochastic_economic_backtest(
    config: StochasticEconomicBacktestConfig,
) -> StochasticEconomicBacktestResult:
    """Replay common stochastic schedules against the factual 2025 ledger."""

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
    if not inputs:
        raise ValueError("No complete regular delivery days remain for the replay")

    residual_path = config.residual_library_path or (
        root
        / "artifacts"
        / "experiments"
        / "joint_residual_bootstrap_v1"
        / "rolling_origin_residual_library.parquet"
    )
    residual_library = pd.read_parquet(residual_path)
    generator = JointResidualBootstrap(
        residual_library,
        hours_local=scenario.working_hours,
        center_residuals=config.center_residuals,
    )

    daily_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    working = np.asarray(scenario.working_hours, dtype=int)
    lower_24, upper_24 = scenario.load_bounds()
    allowed_24 = scenario.discharge_allowed()

    for day_index, day_inputs in enumerate(inputs):
        baseline = replay_day_ahead_only_day(inputs=day_inputs, scenario=scenario)
        deterministic_mpc = replay_deterministic_day(inputs=day_inputs, scenario=scenario)
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
            ledger=deterministic_mpc.ledger,
            load=deterministic_mpc.executed_load_kwh,
            discharge=deterministic_mpc.executed_discharge_kwh,
            position=deterministic_mpc.day_ahead_plan.day_ahead_position_kwh,
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

        point_pv = day_inputs.day_ahead_pv_forecast_kwh[working]
        point_price = day_inputs.day_ahead_price_forecast_eur_per_mwh[working]
        batch = generator.sample(
            point_pv_kwh=point_pv,
            point_price_eur_per_mwh=point_price,
            n_scenarios=config.n_scenarios,
            random_state=config.random_state + day_index,
            as_of_date=day_inputs.delivery_day,
            pv_capacity_kwh_per_hour=config.pv_capacity_kwh_per_hour,
        )
        for risk_weight in (0.0, *config.risk_weights):
            plan = solve_stochastic_day_ahead(
                pv_scenarios_kwh=batch.pv_kwh,
                price_scenarios_eur_per_mwh=batch.day_ahead_price_eur_per_mwh,
                point_pv_forecast_kwh=point_pv,
                load_min_kwh=lower_24[working],
                load_max_kwh=upper_24[working],
                required_load_kwh=scenario.daily_load_kwh,
                battery=scenario.battery(
                    day_inputs.night_reference_price_eur_per_mwh
                ),
                discharge_allowed=allowed_24[working],
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
                battery=scenario.battery(
                    day_inputs.night_reference_price_eur_per_mwh
                ),
            )
            strategy = _strategy_name(risk_weight)
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
            diagnostic_rows.append(
                {
                    "delivery_date_local": day_inputs.delivery_day.isoformat(),
                    "strategy": strategy,
                    "risk_weight": risk_weight,
                    "predicted_expected_cost_eur": plan.expected_cost_eur,
                    "predicted_cvar_cost_eur": plan.cvar_cost_eur,
                    "predicted_objective_eur": plan.objective_eur,
                    "unique_source_residual_days": len(set(batch.source_residual_days)),
                }
            )

    daily_results = pd.DataFrame(daily_rows).sort_values(
        ["delivery_date_local", "strategy"]
    )
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["delivery_date_local", "strategy"]
    )
    summary = _economic_summary(
        daily_results,
        scenario=scenario,
        alpha=config.cvar_alpha,
        config=config,
        residual_days=len(generator.library_days),
    )
    artifact_dir = root / "artifacts" / "experiments" / config.artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    daily_results.to_csv(artifact_dir / "daily_results.csv", index=False)
    pd.DataFrame(hourly_rows).to_parquet(
        artifact_dir / "hourly_decisions.parquet", index=False
    )
    diagnostics.to_csv(artifact_dir / "scenario_objectives.csv", index=False)
    coverage.to_csv(artifact_dir / "coverage.csv", index=False)
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "start": config.start,
                "end": config.end,
                "residual_library_path": str(residual_path),
                "n_scenarios": config.n_scenarios,
                "cvar_alpha": config.cvar_alpha,
                "risk_weights": list(config.risk_weights),
                "center_residuals": config.center_residuals,
                "pv_capacity_kwh_per_hour": config.pv_capacity_kwh_per_hour,
                "random_state": config.random_state,
                "economic_bootstrap_resamples": config.economic_bootstrap_resamples,
                "economic_bootstrap_block_days": config.economic_bootstrap_block_days,
                "market_approximation": (
                    "Within scenario optimization the sampled day-ahead price is "
                    "also the intraday settlement proxy. Factual replay uses the "
                    "observed intraday continuous average price."
                ),
                "decision_contract": (
                    "One load/discharge schedule is shared across all scenarios; "
                    "the submitted position uses the point PV forecast."
                ),
            },
            indent=2,
        )
    )
    return StochasticEconomicBacktestResult(
        artifact_dir=artifact_dir,
        daily_results=daily_results,
        summary=summary,
    )


def _load_replay_inputs(
    *,
    root: Path,
    start,
    end,
    scenario: ReferenceScenario,
    base_config: EconomicBacktestConfig,
):
    test = _read_canonical(root, 2025)
    test = test.merge(
        _read_intraday_actual(root),
        on="valid_time_utc",
        how="left",
        validate="one_to_one",
    )
    return _assemble_replay_inputs(
        start=start,
        end=end,
        scenario=scenario,
        test=test,
        price_predictions=_read_price_predictions(base_config, root),
        pv_day_ahead_predictions=_read_pv_day_ahead_predictions(base_config, root),
        pv_mpc_predictions=_read_pv_mpc_predictions(base_config, root),
        intraday_predictions=_read_intraday_predictions(base_config, root),
        night_references=_night_reference_prices(root),
        correction_weight=base_config.correction_weight,
    )


def _expand_plan(plan, working: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    load = np.zeros(24, dtype=float)
    discharge = np.zeros(24, dtype=float)
    position = np.zeros(24, dtype=float)
    load[working] = plan.load_kwh
    discharge[working] = plan.discharge_kwh
    position[working] = plan.day_ahead_position_kwh
    return load, discharge, position


def _append_strategy(
    daily_rows,
    hourly_rows,
    *,
    strategy,
    day_inputs,
    ledger,
    load,
    discharge,
    position,
    scenario,
) -> None:
    daily_rows.append(
        _ledger_row(
            strategy,
            day_inputs,
            ledger,
            load,
            discharge,
            position,
            scenario,
        )
    )
    hourly_rows.extend(
        _hourly_rows(strategy, day_inputs, load, discharge, position)
    )


def _strategy_name(risk_weight: float) -> str:
    if risk_weight == 0:
        return "bootstrap_saa"
    return f"bootstrap_cvar_lambda_{risk_weight:g}".replace(".", "_")


def _economic_summary(
    daily: pd.DataFrame,
    *,
    scenario: ReferenceScenario,
    alpha: float,
    config: StochasticEconomicBacktestConfig,
    residual_days: int,
    candidate_prefix: str = "bootstrap_",
    library_days_label: str = "residual_library_days",
    include_residual_centering: bool = True,
) -> dict[str, object]:
    by_strategy: dict[str, dict[str, float]] = {}
    for strategy, group in daily.groupby("strategy", sort=True):
        costs = group["total_cost_eur"].to_numpy(dtype=float)
        by_strategy[strategy] = {
            "total_cost_eur": float(costs.sum()),
            "mean_daily_cost_eur": float(costs.mean()),
            "std_daily_cost_eur": float(costs.std(ddof=1)),
            "p95_daily_cost_eur": float(np.quantile(costs, alpha)),
            "realized_cvar95_daily_cost_eur": _upper_tail_mean(costs, alpha),
            "maximum_daily_cost_eur": float(costs.max()),
            "day_ahead_cost_eur": float(group["day_ahead_cost_eur"].sum()),
            "intraday_deviation_cost_eur": float(
                group["intraday_deviation_cost_eur"].sum()
            ),
            "battery_cost_eur": float(group["battery_cost_eur"].sum()),
            "battery_discharge_kwh": float(group["battery_discharge_kwh"].sum()),
        }

    baseline = daily.loc[
        daily["strategy"].eq("deterministic_day_ahead_only"),
        ["delivery_date_local", "total_cost_eur"],
    ].set_index("delivery_date_local")["total_cost_eur"]
    tail_count = max(1, math.ceil((1.0 - alpha) * len(baseline)))
    baseline_tail_days = baseline.nlargest(tail_count).index
    comparisons: dict[str, dict[str, float]] = {}
    for comparison_index, strategy in enumerate(
        name
        for name in by_strategy
        if name.startswith(candidate_prefix)
    ):
        candidate = daily.loc[
            daily["strategy"].eq(strategy),
            ["delivery_date_local", "total_cost_eur"],
        ].set_index("delivery_date_local")["total_cost_eur"]
        saving = baseline - candidate
        uncertainty = _paired_block_bootstrap(
            baseline.to_numpy(dtype=float),
            candidate.to_numpy(dtype=float),
            alpha=alpha,
            block_days=config.economic_bootstrap_block_days,
            n_resamples=config.economic_bootstrap_resamples,
            random_state=config.random_state + 10_000 + comparison_index,
        )
        comparisons[strategy] = {
            "total_saving_vs_deterministic_day_ahead_eur": float(saving.sum()),
            "mean_daily_saving_eur": float(saving.mean()),
            "median_daily_saving_eur": float(saving.median()),
            "share_days_cheaper": float((saving > 1e-9).mean()),
            "share_days_more_expensive": float((saving < -1e-9).mean()),
            "realized_cvar95_reduction_eur": float(
                by_strategy["deterministic_day_ahead_only"][
                    "realized_cvar95_daily_cost_eur"
                ]
                - by_strategy[strategy]["realized_cvar95_daily_cost_eur"]
            ),
            "mean_saving_on_baseline_worst_5pct_days_eur": float(
                saving.loc[baseline_tail_days].mean()
            ),
            **uncertainty,
        }
    summary = {
        "delivery_days": int(baseline.size),
        "total_load_kwh": float(baseline.size * scenario.daily_load_kwh),
        "scenario_count_per_day": config.n_scenarios,
        library_days_label: residual_days,
        "cvar_alpha": alpha,
        "realized_tail_day_count": tail_count,
        "strategy_metrics": by_strategy,
        "comparisons_vs_deterministic_day_ahead": comparisons,
    }
    if include_residual_centering:
        summary["residuals_centered"] = config.center_residuals
    return summary


def _upper_tail_mean(values: np.ndarray, alpha: float) -> float:
    count = max(1, math.ceil((1.0 - alpha) * len(values)))
    return float(np.sort(np.asarray(values, dtype=float))[-count:].mean())


def _paired_block_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    alpha: float,
    block_days: int,
    n_resamples: int,
    random_state: int,
) -> dict[str, float]:
    """Paired moving-block intervals for mean cost and tail-cost differences."""

    n_days = len(baseline)
    blocks_per_sample = math.ceil(n_days / block_days)
    max_start = n_days - block_days
    rng = np.random.default_rng(random_state)
    total_savings = np.empty(n_resamples, dtype=float)
    cvar_reductions = np.empty(n_resamples, dtype=float)
    offsets = np.arange(block_days, dtype=int)
    for sample_index in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=blocks_per_sample)
        indices = (starts[:, None] + offsets[None, :]).ravel()[:n_days]
        baseline_sample = baseline[indices]
        candidate_sample = candidate[indices]
        total_savings[sample_index] = float(
            (baseline_sample - candidate_sample).mean() * n_days
        )
        cvar_reductions[sample_index] = (
            _upper_tail_mean(baseline_sample, alpha)
            - _upper_tail_mean(candidate_sample, alpha)
        )
    total_interval = np.quantile(total_savings, [0.025, 0.975])
    cvar_interval = np.quantile(cvar_reductions, [0.025, 0.975])
    return {
        "total_saving_95pct_block_bootstrap_low_eur": float(total_interval[0]),
        "total_saving_95pct_block_bootstrap_high_eur": float(total_interval[1]),
        "cvar95_reduction_95pct_block_bootstrap_low_eur": float(cvar_interval[0]),
        "cvar95_reduction_95pct_block_bootstrap_high_eur": float(cvar_interval[1]),
    }


def _validate_config(config: StochasticEconomicBacktestConfig) -> None:
    if config.n_scenarios < 2:
        raise ValueError("n_scenarios must be at least two")
    if not 0 < config.cvar_alpha < 1:
        raise ValueError("cvar_alpha must lie strictly between zero and one")
    if any(weight <= 0 or not np.isfinite(weight) for weight in config.risk_weights):
        raise ValueError("risk_weights must contain only finite positive values")
    if len(set(config.risk_weights)) != len(config.risk_weights):
        raise ValueError("risk_weights must be unique")
    if config.pv_capacity_kwh_per_hour <= 0:
        raise ValueError("pv_capacity_kwh_per_hour must be positive")
    if config.economic_bootstrap_resamples < 100:
        raise ValueError("economic_bootstrap_resamples must be at least 100")
    if config.economic_bootstrap_block_days < 1:
        raise ValueError("economic_bootstrap_block_days must be positive")

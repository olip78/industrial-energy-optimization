"""Analyse stochastic-policy errors against the non-speculative DA Oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy.optimization.scenario import V1_REFERENCE_SCENARIO


STOCHASTIC = "cqr_copula_cvar_lambda_0_1"
ORACLE = "oracle"
PRICE_VARIANT = "spatial_weather_catboost"
PV_VARIANT = "day_ahead_catboost"


def _date_string(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["delivery_date_local"]).dt.strftime("%Y-%m-%d")


def _strategy_daily(daily: pd.DataFrame, strategy: str, prefix: str) -> pd.DataFrame:
    columns = [
        "delivery_date_local",
        "total_cost_eur",
        "day_ahead_cost_eur",
        "intraday_deviation_cost_eur",
        "battery_reference_cost_eur",
        "day_ahead_position_kwh",
        "intraday_deviation_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
    ]
    result = daily.loc[daily["strategy"] == strategy, columns].copy()
    return result.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in columns
            if column != "delivery_date_local"
        }
    )


def _forecast_diagnostics(
    root: Path,
    hourly: pd.DataFrame,
    quantiles: pd.DataFrame,
    pv_scale: float,
) -> pd.DataFrame:
    hours = set(V1_REFERENCE_SCENARIO.working_hours)
    price = pd.read_parquet(
        root
        / "artifacts"
        / "experiments"
        / "day_ahead_price_spatial_v1"
        / "hourly_predictions.parquet"
    )
    price["delivery_date_local"] = _date_string(price)
    price = price.loc[
        (price["variant"] == PRICE_VARIANT) & price["hour_local"].isin(hours)
    ].copy()
    price["price_error_eur_per_mwh"] = (
        price["prediction_eur_per_mwh"] - price["actual_price_eur_per_mwh"]
    )
    price_daily = price.groupby("delivery_date_local", as_index=False).agg(
        day_ahead_price_mae_eur_per_mwh=(
            "price_error_eur_per_mwh",
            lambda values: float(values.abs().mean()),
        ),
        day_ahead_price_bias_eur_per_mwh=("price_error_eur_per_mwh", "mean"),
        maximum_actual_day_ahead_price_eur_per_mwh=(
            "actual_price_eur_per_mwh",
            "max",
        ),
    )
    price_rank = (
        price.groupby("delivery_date_local")
        .apply(
            lambda frame: frame["prediction_eur_per_mwh"].corr(
                frame["actual_price_eur_per_mwh"], method="spearman"
            ),
            include_groups=False,
        )
        .rename("day_ahead_price_spearman")
        .reset_index()
    )
    price_daily = price_daily.merge(price_rank, on="delivery_date_local")

    pv = pd.read_parquet(
        root
        / "artifacts"
        / "experiments"
        / "temporal_backtest_v1"
        / "2024_to_2025"
        / "pv_day_ahead_hourly_predictions.parquet"
    )
    pv["delivery_date_local"] = _date_string(pv)
    pv = pv.loc[(pv["variant"] == PV_VARIANT) & pv["hour_local"].isin(hours)].copy()
    pv["forecast_pv_kwh"] = pv["prediction_w"] / 1_000.0 * pv_scale
    pv["actual_pv_kwh"] = pv["target_pv_power_w"] / 1_000.0 * pv_scale
    pv["pv_error_kwh"] = pv["forecast_pv_kwh"] - pv["actual_pv_kwh"]
    pv_daily = pv.groupby("delivery_date_local", as_index=False).agg(
        forecast_pv_energy_kwh=("forecast_pv_kwh", "sum"),
        actual_pv_energy_kwh=("actual_pv_kwh", "sum"),
        pv_energy_bias_kwh=("pv_error_kwh", "sum"),
        pv_hourly_mae_kwh=("pv_error_kwh", lambda values: float(values.abs().mean())),
    )

    quantiles = quantiles.copy()
    quantiles["delivery_date_local"] = _date_string(quantiles)
    quantiles["price_above_q95"] = (
        quantiles["actual_price_eur_per_mwh"] > quantiles["price_q95"]
    )
    quantiles["q50_error"] = (
        quantiles["price_q50"] - quantiles["actual_price_eur_per_mwh"]
    )
    probabilistic = quantiles.groupby("delivery_date_local", as_index=False).agg(
        price_q50_mae_eur_per_mwh=(
            "q50_error", lambda values: float(values.abs().mean())
        ),
        price_q50_bias_eur_per_mwh=("q50_error", "mean"),
        price_q95_breach_hours=("price_above_q95", "sum"),
    )

    stochastic = hourly.loc[hourly["strategy"] == STOCHASTIC].copy()
    oracle = hourly.loc[hourly["strategy"] == ORACLE].copy()
    keys = ["delivery_date_local", "hour_local"]
    operations = stochastic[
        keys + ["load_kwh", "battery_discharge_kwh", "battery_charge_kwh"]
    ].merge(
        oracle[keys + ["load_kwh", "battery_discharge_kwh", "battery_charge_kwh"]],
        on=keys,
        suffixes=("_stochastic", "_oracle"),
        validate="one_to_one",
    )
    for name in ("load", "battery_discharge", "battery_charge"):
        operations[f"{name}_l1_kwh"] = (
            operations[f"{name}_kwh_stochastic"]
            - operations[f"{name}_kwh_oracle"]
        ).abs()
    operation_daily = operations.groupby("delivery_date_local", as_index=False).agg(
        stochastic_vs_oracle_load_l1_kwh=("load_l1_kwh", "sum"),
        stochastic_vs_oracle_discharge_l1_kwh=("battery_discharge_l1_kwh", "sum"),
        stochastic_vs_oracle_charge_l1_kwh=("battery_charge_l1_kwh", "sum"),
    )
    return (
        price_daily.merge(pv_daily, on="delivery_date_local", validate="one_to_one")
        .merge(probabilistic, on="delivery_date_local", validate="one_to_one")
        .merge(operation_daily, on="delivery_date_local", validate="one_to_one")
    )


def _plot_decomposition(analysis: pd.DataFrame, output_dir: Path) -> None:
    top = analysis.nlargest(15, "oracle_regret_eur").sort_values(
        "oracle_regret_eur"
    )
    components = [
        ("day_ahead_settlement_gap_eur", "Day-ahead schedule/position", "#3978C5"),
        ("intraday_settlement_gap_eur", "PV-error intraday settlement", "#D47A3A"),
        ("battery_cost_gap_eur", "Battery reference cost", "#7A5AA6"),
    ]
    y = np.arange(len(top))
    positive_left = np.zeros(len(top), dtype=float)
    negative_left = np.zeros(len(top), dtype=float)
    figure, axis = plt.subplots(figsize=(13.5, 8.5))
    figure.patch.set_facecolor("#FAFAF8")
    axis.set_facecolor("#FAFAF8")
    for column, label, color in components:
        values = top[column].to_numpy(dtype=float)
        left = np.where(values >= 0.0, positive_left, negative_left)
        axis.barh(y, values, left=left, color=color, alpha=0.86, label=label)
        positive_left += np.where(values >= 0.0, values, 0.0)
        negative_left += np.where(values < 0.0, values, 0.0)
    axis.axvline(0.0, color="#777777", linewidth=0.8)
    axis.set_yticks(y, top["delivery_date_local"])
    axis.grid(axis="x", color="#D8D8D4", linewidth=0.8, alpha=0.85)
    axis.spines[["top", "right"]].set_visible(False)
    for row_index, (_, row) in enumerate(top.iterrows()):
        axis.text(
            positive_left[row_index] + 0.18,
            row_index,
            f"net €{row['oracle_regret_eur']:.2f}",
            va="center",
            fontsize=8.6,
            color="#333333",
        )
    axis.set_title(
        "Largest errors against the day-ahead Oracle",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=22,
    )
    axis.text(
        0.0,
        1.01,
        "Oracle knows actual DA price and PV · one physical schedule · zero deliberate imbalance",
        transform=axis.transAxes,
        fontsize=10.5,
        color="#555555",
    )
    axis.set_xlabel("Accounting contribution to daily cost gap, EUR")
    axis.legend(loc="lower right", frameon=True, framealpha=0.95)
    figure.tight_layout(pad=2.0)
    figure.savefig(output_dir / "oracle_regret_decomposition_top_days.png", dpi=220)
    figure.savefig(output_dir / "oracle_regret_decomposition_top_days.svg")
    plt.close(figure)


def _plot_case_studies(
    root: Path,
    analysis: pd.DataFrame,
    hourly: pd.DataFrame,
    quantiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    top_days = analysis.nlargest(4, "oracle_regret_eur")[
        "delivery_date_local"
    ].tolist()
    point = pd.read_parquet(
        root
        / "artifacts"
        / "experiments"
        / "day_ahead_price_spatial_v1"
        / "hourly_predictions.parquet"
    )
    point["delivery_date_local"] = _date_string(point)
    point = point.loc[point["variant"] == PRICE_VARIANT]
    quantiles = quantiles.copy()
    quantiles["delivery_date_local"] = _date_string(quantiles)

    figure, axes = plt.subplots(
        len(top_days), 2, figsize=(16, 3.65 * len(top_days)), squeeze=False
    )
    figure.patch.set_facecolor("#FAFAF8")
    for row_index, day in enumerate(top_days):
        price_axis, action_axis = axes[row_index]
        q = quantiles.loc[quantiles["delivery_date_local"] == day].sort_values(
            "hour_local"
        )
        p = point.loc[point["delivery_date_local"] == day].sort_values("hour_local")
        stochastic = hourly.loc[
            (hourly["delivery_date_local"] == day)
            & (hourly["strategy"] == STOCHASTIC)
        ].sort_values("hour_local")
        oracle = hourly.loc[
            (hourly["delivery_date_local"] == day)
            & (hourly["strategy"] == ORACLE)
        ].sort_values("hour_local")
        hours = q["hour_local"].to_numpy(dtype=int)
        p = p.loc[p["hour_local"].isin(hours)]
        row = analysis.loc[analysis["delivery_date_local"] == day].iloc[0]

        price_axis.set_facecolor("#FAFAF8")
        price_axis.fill_between(
            hours,
            q["price_q05"],
            q["price_q95"],
            color="#7BA6DF",
            alpha=0.2,
            label="CQR P05–P95",
        )
        price_axis.plot(
            hours,
            p["prediction_eur_per_mwh"],
            color="#2F6BBD",
            linewidth=1.35,
            label="Point DA forecast",
        )
        price_axis.plot(
            hours,
            stochastic["actual_day_ahead_price_eur_per_mwh"],
            color="#202020",
            linewidth=1.45,
            label="Actual DA",
        )
        price_axis.set_title(
            f"{day} · regret €{row['oracle_regret_eur']:.2f}",
            loc="left",
            fontweight="bold",
        )
        price_axis.set_ylabel("EUR/MWh")
        price_axis.grid(axis="y", color="#D8D8D4", linewidth=0.7, alpha=0.8)
        price_axis.spines[["top", "right"]].set_visible(False)

        action_axis.set_facecolor("#FAFAF8")
        action_axis.plot(
            hours,
            stochastic["load_kwh"],
            color="#3978C5",
            linewidth=1.35,
            marker="o",
            markersize=3,
            label="Stochastic load",
        )
        action_axis.plot(
            hours,
            oracle["load_kwh"],
            color="#168A6B",
            linewidth=1.35,
            marker="o",
            markersize=3,
            label="Oracle load",
        )
        action_axis.plot(
            hours,
            stochastic["battery_discharge_kwh"],
            color="#7BA6DF",
            linewidth=1.1,
            linestyle="--",
            label="Stochastic discharge",
        )
        action_axis.plot(
            hours,
            oracle["battery_discharge_kwh"],
            color="#D47A3A",
            linewidth=1.1,
            linestyle="--",
            label="Oracle discharge",
        )
        action_axis.set_title(
            (
                f"DA €{row['day_ahead_settlement_gap_eur']:.2f} · "
                f"PV settlement €{row['intraday_settlement_gap_eur']:.2f} · "
                f"battery €{row['battery_cost_gap_eur']:.2f}"
            ),
            loc="left",
            fontsize=10.5,
        )
        action_axis.set_ylabel("Energy per hour, kWh")
        action_axis.grid(axis="y", color="#D8D8D4", linewidth=0.7, alpha=0.75)
        action_axis.spines[["top", "right"]].set_visible(False)
        if row_index == 0:
            price_axis.legend(loc="upper left", ncol=2, fontsize=8.5)
            action_axis.legend(loc="upper left", ncol=2, fontsize=8.2)
        if row_index == len(top_days) - 1:
            price_axis.set_xlabel("Local delivery hour")
            action_axis.set_xlabel("Local delivery hour")
    figure.suptitle(
        "Day-ahead Oracle regret case studies",
        x=0.055,
        y=0.999,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985), pad=1.8)
    figure.savefig(output_dir / "oracle_regret_case_studies.png", dpi=220)
    figure.savefig(output_dir / "oracle_regret_case_studies.svg")
    plt.close(figure)


def _write_report(analysis: pd.DataFrame, output_dir: Path) -> None:
    total = float(analysis["oracle_regret_eur"].sum())
    top = analysis.nlargest(10, "oracle_regret_eur")
    lines = [
        "# Day-ahead Oracle regret error analysis",
        "",
        "The Oracle knows factual PV and factual day-ahead prices. It chooses "
        "one physical load and battery schedule, and its nomination equals that "
        "schedule's net position. Intraday prices do not enter the Oracle "
        "decision and its realised imbalance is zero.",
        "",
        "$$",
        "R_d = C_{strategy,d} - C_{DA\\ Oracle,d}",
        "= \\Delta C_{DA,d} + \\Delta C_{ID,d} + \\Delta C_{battery,d}.",
        "$$",
        "",
        f"Across 261 delivery days, stochastic-policy regret against this "
        f"non-speculative Oracle is **EUR {total:.2f}**.",
        "",
        "| Date | Regret, EUR | DA component | PV settlement | Battery component | DA price MAE | PV bias, kWh |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['delivery_date_local']} | {row['oracle_regret_eur']:.2f} "
            f"| {row['day_ahead_settlement_gap_eur']:.2f} "
            f"| {row['intraday_settlement_gap_eur']:.2f} "
            f"| {row['battery_cost_gap_eur']:.2f} "
            f"| {row['day_ahead_price_mae_eur_per_mwh']:.2f} "
            f"| {row['pv_energy_bias_kwh']:.2f} |"
        )
    lines.append("")
    (output_dir / "oracle_regret_error_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = (
        root / "artifacts" / "experiments" / "economic_backtest_v2_dynamic_charge"
    )
    daily = pd.read_csv(output_dir / "daily_results.csv")
    hourly = pd.read_parquet(output_dir / "hourly_decisions.parquet")
    daily["delivery_date_local"] = _date_string(daily)
    hourly["delivery_date_local"] = _date_string(hourly)
    config = json.loads((output_dir / "experiment_config.json").read_text())
    quantiles = pd.read_parquet(
        root
        / "artifacts"
        / "experiments"
        / "cqr_copula_v1"
        / "test_calibrated_quantile_forecasts.parquet"
    )

    oracle_daily = daily.loc[daily["strategy"] == ORACLE]
    if not np.allclose(oracle_daily["intraday_deviation_cost_eur"], 0.0, atol=1e-9):
        raise AssertionError("Oracle has non-zero intraday settlement")
    if not np.allclose(oracle_daily["intraday_deviation_kwh"], 0.0, atol=1e-9):
        raise AssertionError("Oracle has non-zero physical imbalance")

    stochastic = _strategy_daily(daily, STOCHASTIC, "stochastic")
    oracle = _strategy_daily(daily, ORACLE, "oracle")
    analysis = stochastic.merge(
        oracle, on="delivery_date_local", validate="one_to_one"
    )
    analysis["oracle_regret_eur"] = (
        analysis["stochastic_total_cost_eur"] - analysis["oracle_total_cost_eur"]
    )
    analysis["day_ahead_settlement_gap_eur"] = (
        analysis["stochastic_day_ahead_cost_eur"]
        - analysis["oracle_day_ahead_cost_eur"]
    )
    analysis["intraday_settlement_gap_eur"] = (
        analysis["stochastic_intraday_deviation_cost_eur"]
        - analysis["oracle_intraday_deviation_cost_eur"]
    )
    analysis["battery_cost_gap_eur"] = (
        analysis["stochastic_battery_reference_cost_eur"]
        - analysis["oracle_battery_reference_cost_eur"]
    )
    diagnostics = _forecast_diagnostics(
        root, hourly, quantiles, float(config["pv_profile_scale"])
    )
    analysis = analysis.merge(
        diagnostics, on="delivery_date_local", validate="one_to_one"
    ).sort_values("delivery_date_local")
    identity_error = (
        analysis["oracle_regret_eur"]
        - analysis["day_ahead_settlement_gap_eur"]
        - analysis["intraday_settlement_gap_eur"]
        - analysis["battery_cost_gap_eur"]
    ).abs().max()
    if identity_error > 1e-8:
        raise AssertionError(f"Regret decomposition failed by {identity_error:.3g}")

    analysis.to_csv(output_dir / "oracle_regret_daily_error_analysis.csv", index=False)
    analysis.nlargest(15, "oracle_regret_eur").to_csv(
        output_dir / "oracle_regret_top_days.csv", index=False
    )
    _plot_decomposition(analysis, output_dir)
    _plot_case_studies(root, analysis, hourly, quantiles, output_dir)
    _write_report(analysis, output_dir)

    for obsolete in (
        "physical_day_ahead_oracle_hourly.parquet",
        "forecast_schedule_regret_top_days.csv",
        "forecast_schedule_error_decomposition.png",
        "forecast_schedule_error_decomposition.svg",
    ):
        path = output_dir / obsolete
        if path.exists():
            path.unlink()

    print(f"Total regret against DA Oracle: EUR {analysis['oracle_regret_eur'].sum():.6f}")
    print(
        analysis.nlargest(10, "oracle_regret_eur")[
            [
                "delivery_date_local",
                "oracle_regret_eur",
                "day_ahead_settlement_gap_eur",
                "intraday_settlement_gap_eur",
                "battery_cost_gap_eur",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

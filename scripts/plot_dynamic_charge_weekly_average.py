"""Plot weekly-smoothed economic costs and frozen forecast diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


STYLE = {
    "oracle": ("Day-ahead Oracle (actual price + PV)", "#168A6B", 1.6),
    "rule_based": ("Rule-based", "#777777", 1.2),
    "day_ahead_only": ("Deterministic day-ahead", "#2F6BBD", 1.3),
    "deterministic": ("Deterministic + MPC", "#6F4AA8", 1.6),
    "cqr_copula_cvar_lambda_0_1": (
        "CQR + empirical copula, CVaR λ=0.10",
        "#C4514A",
        1.45,
    ),
}

PRICE_VARIANT = "spatial_weather_catboost"
PV_VARIANT = "day_ahead_catboost"
OPERATING_HOURS = tuple(range(6, 22))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--artifact-name",
        default="economic_backtest_v3_grid_tariff",
    )
    args = parser.parse_args()
    artifact_dir = (
        args.project_root.resolve()
        / "artifacts"
        / "experiments"
        / args.artifact_name
    )
    daily = pd.read_csv(artifact_dir / "daily_results.csv")
    daily["delivery_date_local"] = pd.to_datetime(daily["delivery_date_local"])
    daily = daily.sort_values(["strategy", "delivery_date_local"])
    totals = daily.groupby("strategy")["total_cost_eur"].sum().to_dict()
    oracle_potential_eur = totals["rule_based"] - totals["oracle"]

    figure, axis = plt.subplots(figsize=(15, 7.5))
    figure.patch.set_facecolor("#FAFAF8")
    axis.set_facecolor("#FAFAF8")
    axis.grid(axis="y", color="#D8D8D4", linewidth=0.8, alpha=0.85)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#B8B8B2")

    for strategy, (label, color, linewidth) in STYLE.items():
        frame = daily.loc[daily["strategy"] == strategy].copy()
        frame = frame.set_index("delivery_date_local")
        frame["weekly_mean_eur"] = frame["total_cost_eur"].rolling(
            "7D", min_periods=1
        ).mean()
        total = float(totals[strategy])
        captured_potential = 100.0 * (totals["rule_based"] - total) / oracle_potential_eur
        axis.plot(
            frame.index,
            frame["weekly_mean_eur"],
            color=color,
            linewidth=linewidth,
            label=(
                f"{label}  ·  {captured_potential:.1f}% of potential  "
                f"·  €{total:,.0f}"
            ),
        )

    axis.set_title(
        "Economic simulation with physical grid tariffs",
        loc="left",
        fontsize=19,
        fontweight="bold",
        pad=24,
    )
    axis.text(
        0.0,
        1.015,
        (
            "Trailing 7-day mean of realised daily operating cost · "
            "261 delivery days, Jan–Sep 2025 · lower is better"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color="#4A4A48",
    )
    axis.set_ylabel("7-day mean cost, EUR per day")
    axis.set_xlabel("Delivery date in 2025")
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axis.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.95,
        facecolor="#FFFFFF",
        edgecolor="#D8D8D4",
        fontsize=10,
    )
    axis.text(
        0.995,
        0.025,
        (
            "Captured potential = (Rule-based cost − strategy cost) / "
            "(Rule-based cost − Oracle cost)\n"
            f"Rule-based → Oracle gap: €{oracle_potential_eur:,.2f} = 100%"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#FFFFFF",
            "edgecolor": "#D8D8D4",
            "alpha": 0.94,
        },
    )
    figure.tight_layout(pad=2.2)
    figure.savefig(artifact_dir / "economic_strategy_7d_average.png", dpi=220)
    figure.savefig(artifact_dir / "economic_strategy_7d_average.svg")
    plt.close(figure)

    _plot_cost_components(daily, artifact_dir)
    _plot_day_ahead_plan_fact(args.project_root.resolve(), artifact_dir)


def _plot_cost_components(daily: pd.DataFrame, artifact_dir: Path) -> None:
    """Show how market, battery and regulated charges form total cost."""

    order = [
        "rule_based",
        "day_ahead_only",
        "cqr_copula_cvar_lambda_0_1",
        "deterministic",
        "oracle",
    ]
    labels = [STYLE[strategy][0] for strategy in order]
    grouped = daily.groupby("strategy").sum(numeric_only=True).loc[order]
    components = {
        "Wholesale market settlement": (
            grouped["day_ahead_cost_eur"]
            + grouped["intraday_deviation_cost_eur"]
        ),
        "Night battery energy": grouped["battery_night_energy_cost_eur"],
        "Battery degradation": grouped["battery_degradation_cost_eur"],
        "Night grid charges": grouped["battery_night_grid_cost_eur"],
        "Day network use": grouped["daytime_network_cost_eur"],
        "Levies + concession + tax": (
            grouped["levies_cost_eur"]
            + grouped["concession_fee_cost_eur"]
            + grouped["electricity_tax_cost_eur"]
            + grouped["export_fee_cost_eur"]
        ),
        "Fixed grid + meter": grouped["grid_fixed_cost_eur"],
    }
    colors = [
        "#365F91",
        "#D7A34A",
        "#A45C40",
        "#8A6FAD",
        "#599D8E",
        "#7FAAC8",
        "#A7A7A1",
    ]

    figure, axis = plt.subplots(figsize=(14.5, 8.2))
    figure.patch.set_facecolor("#FAFAF8")
    axis.set_facecolor("#FAFAF8")
    axis.grid(axis="y", color="#D8D8D4", linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#B8B8B2")

    bottom = pd.Series(0.0, index=order)
    for (name, values), color in zip(components.items(), colors):
        axis.bar(
            labels,
            values.to_numpy(dtype=float),
            bottom=bottom.to_numpy(dtype=float),
            label=name,
            color=color,
            width=0.66,
        )
        bottom = bottom + values

    for index, total in enumerate(grouped["total_cost_eur"]):
        axis.text(
            index,
            float(total) + 75.0,
            f"€{float(total):,.0f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )
    axis.set_title(
        "What the 2025 operating cost contains",
        loc="left",
        fontsize=19,
        fontweight="bold",
        pad=22,
    )
    axis.text(
        0.0,
        1.01,
        "261 delivery days · Pforzheim SLP 2025 · values exclude VAT",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color="#4A4A48",
    )
    axis.set_ylabel("Cost over replay period, EUR")
    axis.tick_params(axis="x", labelrotation=12)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    figure.tight_layout(pad=2.2)
    figure.savefig(artifact_dir / "economic_cost_component_breakdown.png", dpi=220)
    figure.savefig(artifact_dir / "economic_cost_component_breakdown.svg")
    plt.close(figure)


def _plot_day_ahead_plan_fact(project_root: Path, artifact_dir: Path) -> None:
    """Plot the frozen forecasts that feed the economic replay against facts."""

    experiment_config = json.loads(
        (artifact_dir / "experiment_config.json").read_text()
    )
    pv_profile_scale = float(experiment_config["pv_profile_scale"])
    used_days = set(
        pd.read_csv(artifact_dir / "coverage.csv")
        .query("status == 'used'")["delivery_date_local"]
        .astype(str)
    )
    experiments = project_root / "artifacts" / "experiments"

    price = pd.read_parquet(
        experiments / "day_ahead_price_spatial_v1" / "hourly_predictions.parquet"
    )
    price = price.loc[
        (price["variant"] == PRICE_VARIANT)
        & price["delivery_date_local"].astype(str).isin(used_days)
        & price["hour_local"].isin(OPERATING_HOURS)
    ].copy()
    price["delivery_date_local"] = pd.to_datetime(price["delivery_date_local"])

    pv = pd.read_parquet(
        experiments
        / "temporal_backtest_v1"
        / "2024_to_2025"
        / "pv_day_ahead_hourly_predictions.parquet"
    )
    pv = pv.loc[
        (pv["variant"] == PV_VARIANT)
        & pv["delivery_date_local"].astype(str).isin(used_days)
        & pv["hour_local"].isin(OPERATING_HOURS)
    ].copy()
    pv["delivery_date_local"] = pd.to_datetime(pv["delivery_date_local"])

    price_daily = (
        price.groupby("delivery_date_local", as_index=False)
        .agg(
            actual_day_ahead_price_mean_eur_per_mwh=(
                "actual_price_eur_per_mwh",
                "mean",
            ),
            forecast_day_ahead_price_mean_eur_per_mwh=(
                "prediction_eur_per_mwh",
                "mean",
            ),
        )
        .sort_values("delivery_date_local")
    )
    pv_daily = (
        pv.groupby("delivery_date_local", as_index=False)
        .agg(
            actual_pv_energy_kwh=(
                "target_pv_power_w",
                lambda values: values.sum()
                / 1_000.0
                * pv_profile_scale,
            ),
            forecast_pv_energy_kwh=(
                "prediction_w",
                lambda values: values.sum()
                / 1_000.0
                * pv_profile_scale,
            ),
        )
        .sort_values("delivery_date_local")
    )
    daily = price_daily.merge(
        pv_daily,
        on="delivery_date_local",
        how="inner",
        validate="one_to_one",
    )
    daily.to_csv(artifact_dir / "forecast_plan_fact_daily.csv", index=False)

    price_mae = float(
        (price["prediction_eur_per_mwh"] - price["actual_price_eur_per_mwh"])
        .abs()
        .mean()
    )
    price_rmse = float(
        (
            (price["prediction_eur_per_mwh"] - price["actual_price_eur_per_mwh"])
            .pow(2)
            .mean()
        )
        ** 0.5
    )
    pv_error_kwh = (
        (pv["prediction_w"] - pv["target_pv_power_w"])
        / 1_000.0
        * pv_profile_scale
    )
    pv_mae = float(pv_error_kwh.abs().mean())
    pv_rmse = float(
        (pv_error_kwh.pow(2).mean()) ** 0.5
    )

    indexed = daily.set_index("delivery_date_local")
    smoothed = indexed.rolling("7D", min_periods=1).mean()

    figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    figure.patch.set_facecolor("#FAFAF8")
    for axis in axes:
        axis.set_facecolor("#FAFAF8")
        axis.grid(axis="y", color="#D8D8D4", linewidth=0.8, alpha=0.85)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#B8B8B2")

    axes[0].plot(
        smoothed.index,
        smoothed["actual_day_ahead_price_mean_eur_per_mwh"],
        color="#222222",
        linewidth=1.45,
        label="Actual",
    )
    axes[0].plot(
        smoothed.index,
        smoothed["forecast_day_ahead_price_mean_eur_per_mwh"],
        color="#2F6BBD",
        linewidth=1.25,
        label="Day-ahead forecast",
    )
    axes[0].set_title("Day-ahead price: forecast vs actual", loc="left", fontweight="bold")
    axes[0].set_ylabel("7-day mean price, EUR/MWh")
    axes[0].text(
        0.995,
        0.04,
        f"Hourly MAE {price_mae:.2f} EUR/MWh · RMSE {price_rmse:.2f} EUR/MWh",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    axes[1].plot(
        smoothed.index,
        smoothed["actual_pv_energy_kwh"],
        color="#222222",
        linewidth=1.45,
        label="Actual",
    )
    axes[1].plot(
        smoothed.index,
        smoothed["forecast_pv_energy_kwh"],
        color="#E08A24",
        linewidth=1.25,
        label="Day-ahead forecast",
    )
    axes[1].set_title("PV production: forecast vs actual", loc="left", fontweight="bold")
    axes[1].set_ylabel("7-day mean daily energy, kWh")
    axes[1].set_xlabel("Delivery date in 2025")
    axes[1].text(
        0.995,
        0.04,
        f"Hourly MAE {pv_mae:.2f} kWh · RMSE {pv_rmse:.2f} kWh",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    figure.suptitle(
        "Frozen 2025 forecasts used by the economic simulation",
        x=0.055,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.935,
        (
            "Plan vs fact over the 06:00–22:00 operating window · "
            "trailing 7-day means · 261 delivery days, Jan–Sep 2025"
        ),
        ha="left",
        fontsize=11,
        color="#4A4A48",
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.99, 0.91), h_pad=2.4)
    figure.savefig(artifact_dir / "day_ahead_forecast_plan_fact_7d_average.png", dpi=220)
    figure.savefig(artifact_dir / "day_ahead_forecast_plan_fact_7d_average.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()

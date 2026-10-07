"""Plot the realised 2025 economics of the main control strategies.

The figure combines strategies from the deterministic economic replay and the
centred residual-bootstrap replay. Hourly costs are reconstructed from the
stored physical decisions and checked against the authoritative daily ledger.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    key: str
    label: str
    source: str
    color: str
    linewidth: float = 1.8


STRATEGIES = (
    StrategySpec("oracle", "Day-ahead Oracle", "deterministic", "#168A6B", 2.2),
    StrategySpec("rule_based", "Rule-based", "deterministic", "#7A7A7A", 1.8),
    StrategySpec(
        "day_ahead_only",
        "Deterministic day-ahead",
        "deterministic",
        "#2F6BBD",
        1.8,
    ),
    StrategySpec(
        "deterministic",
        "Deterministic + MPC",
        "deterministic",
        "#6F4AA8",
        2.1,
    ),
    StrategySpec(
        "bootstrap_cvar_lambda_0_1",
        "Stochastic CVaR95 (λ = 0.10)",
        "stochastic",
        "#E6862A",
        2.1,
    ),
)


def _read_sources(project_root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    experiment_root = project_root / "artifacts" / "experiments"
    paths = {
        "deterministic": experiment_root / "economic_backtest_v1",
        "stochastic": experiment_root / "stochastic_economic_backtest_v1",
    }
    return {
        name: {
            "daily": pd.read_csv(path / "daily_results.csv"),
            "hourly": pd.read_parquet(path / "hourly_decisions.parquet"),
        }
        for name, path in paths.items()
    }


def _select_strategies(
    sources: dict[str, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames: list[pd.DataFrame] = []
    hourly_frames: list[pd.DataFrame] = []
    for spec in STRATEGIES:
        daily = sources[spec.source]["daily"]
        hourly = sources[spec.source]["hourly"]
        selected_daily = daily.loc[daily["strategy"] == spec.key].copy()
        selected_hourly = hourly.loc[hourly["strategy"] == spec.key].copy()
        if selected_daily.empty or selected_hourly.empty:
            raise ValueError(f"Missing stored results for strategy {spec.key!r}")
        selected_daily["display_label"] = spec.label
        selected_hourly["display_label"] = spec.label
        daily_frames.append(selected_daily)
        hourly_frames.append(selected_hourly)

    daily = pd.concat(daily_frames, ignore_index=True)
    hourly = pd.concat(hourly_frames, ignore_index=True)
    daily["delivery_date_local"] = pd.to_datetime(daily["delivery_date_local"])
    hourly["delivery_date_local"] = pd.to_datetime(hourly["delivery_date_local"])

    date_sets = [
        set(group["delivery_date_local"])
        for _, group in daily.groupby("strategy", sort=False)
    ]
    common_dates = set.intersection(*date_sets)
    if not common_dates:
        raise ValueError("The selected strategies have no common delivery dates")
    daily = daily.loc[daily["delivery_date_local"].isin(common_dates)].copy()
    hourly = hourly.loc[hourly["delivery_date_local"].isin(common_dates)].copy()
    return daily, hourly


def _reconstruct_hourly_costs(
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["delivery_date_local", "strategy"]
    battery_ledger = daily[
        [*keys, "battery_cost_eur", "battery_discharge_kwh"]
    ].rename(
        columns={
            "battery_cost_eur": "daily_battery_cost_eur",
            "battery_discharge_kwh": "daily_battery_discharge_kwh",
        }
    )
    result = hourly.merge(battery_ledger, on=keys, how="left", validate="many_to_one")
    result["intraday_deviation_kwh"] = (
        result["load_kwh"]
        - result["actual_pv_kwh"]
        - result["battery_discharge_kwh"]
        - result["day_ahead_position_kwh"]
    )
    result["day_ahead_cost_eur"] = (
        result["actual_day_ahead_price_eur_per_mwh"]
        * result["day_ahead_position_kwh"]
        / 1_000.0
    )
    result["intraday_deviation_cost_eur"] = (
        result["actual_intraday_price_eur_per_mwh"]
        * result["intraday_deviation_kwh"]
        / 1_000.0
    )
    result["battery_cost_eur"] = np.where(
        result["daily_battery_discharge_kwh"] > 0,
        result["daily_battery_cost_eur"]
        * result["battery_discharge_kwh"]
        / result["daily_battery_discharge_kwh"],
        0.0,
    )
    result["hourly_total_cost_eur"] = result[
        ["day_ahead_cost_eur", "intraday_deviation_cost_eur", "battery_cost_eur"]
    ].sum(axis=1)
    result["timestamp_local"] = result["delivery_date_local"] + pd.to_timedelta(
        result["hour_local"], unit="h"
    )
    result = result.sort_values(["strategy", "timestamp_local"]).reset_index(drop=True)

    reconstructed = (
        result.groupby(keys, as_index=False)["hourly_total_cost_eur"]
        .sum()
        .rename(columns={"hourly_total_cost_eur": "reconstructed_total_cost_eur"})
    )
    audit = daily[keys + ["total_cost_eur"]].merge(
        reconstructed, on=keys, how="left", validate="one_to_one"
    )
    max_error = float(
        (audit["total_cost_eur"] - audit["reconstructed_total_cost_eur"])
        .abs()
        .max()
    )
    if max_error > 1e-8:
        raise ValueError(f"Hourly ledger does not reconcile; max error = {max_error:.3g}")
    return result


def _add_plot_series(
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = hourly.copy()
    daily = daily.sort_values(["strategy", "delivery_date_local"]).copy()
    hourly["cost_6h_mean_eur"] = hourly.groupby("strategy", sort=False)[
        "hourly_total_cost_eur"
    ].transform(lambda values: values.rolling(6, min_periods=1).mean())
    daily["cost_14d_mean_eur"] = daily.groupby("strategy", sort=False)[
        "total_cost_eur"
    ].transform(lambda values: values.rolling(14, min_periods=1).mean())

    rule = daily.loc[
        daily["strategy"] == "rule_based",
        ["delivery_date_local", "total_cost_eur"],
    ].rename(columns={"total_cost_eur": "rule_based_cost_eur"})
    daily = daily.merge(rule, on="delivery_date_local", how="left", validate="many_to_one")
    daily["saving_vs_rule_eur"] = (
        daily["rule_based_cost_eur"] - daily["total_cost_eur"]
    )
    daily["cumulative_saving_vs_rule_eur"] = daily.groupby(
        "strategy", sort=False
    )["saving_vs_rule_eur"].cumsum()
    return daily, hourly


def _plot(daily: pd.DataFrame, hourly: pd.DataFrame, output_dir: Path) -> None:
    totals = daily.groupby("strategy")["total_cost_eur"].sum().to_dict()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.05, 0.9], "hspace": 0.12},
    )
    fig.patch.set_facecolor("#FAFAF8")
    for axis in axes:
        axis.set_facecolor("#FAFAF8")
        axis.grid(axis="y", color="#D8D8D4", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#B8B8B2")

    for spec in STRATEGIES:
        h = hourly.loc[hourly["strategy"] == spec.key]
        d = daily.loc[daily["strategy"] == spec.key]
        legend_label = f"{spec.label}  (€{totals[spec.key]:,.0f})"
        axes[0].plot(
            h["timestamp_local"],
            h["cost_6h_mean_eur"],
            color=spec.color,
            linewidth=spec.linewidth,
            alpha=0.92,
            label=legend_label,
        )
        axes[1].plot(
            d["delivery_date_local"],
            d["total_cost_eur"],
            color=spec.color,
            linewidth=0.65,
            alpha=0.18,
        )
        axes[1].plot(
            d["delivery_date_local"],
            d["cost_14d_mean_eur"],
            color=spec.color,
            linewidth=spec.linewidth,
            alpha=0.95,
        )
        axes[2].plot(
            d["delivery_date_local"],
            d["cumulative_saving_vs_rule_eur"],
            color=spec.color,
            linewidth=spec.linewidth,
            alpha=0.95,
        )

    axes[0].set_title(
        "Realised hourly operating cost — trailing 6-hour mean",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_ylabel("EUR per hour")
    axes[0].legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=0.94,
        facecolor="#FFFFFF",
        edgecolor="#D8D8D4",
        fontsize=9,
    )

    axes[1].set_title(
        "Realised daily cost — raw values and trailing 14-day mean",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    axes[1].set_ylabel("EUR per day")

    axes[2].set_title(
        "Cumulative saving relative to the rule-based strategy",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    axes[2].axhline(0, color="#555555", linewidth=0.9, linestyle="--")
    axes[2].set_ylabel("EUR saved")
    axes[2].set_xlabel("Delivery date in 2025")
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.suptitle(
        "Economic simulation of the energy-management strategies",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.955,
        (
            "261 common delivery days (Jan–Sep 2025); lower cost is better. "
            "Oracle uses future facts and is a lower bound."
        ),
        ha="left",
        fontsize=10.5,
        color="#4A4A48",
    )
    fig.text(
        0.07,
        0.018,
        (
            "Stochastic line: centred joint-residual bootstrap with CVaR95, λ = 0.10. "
            "It optimises the day-ahead schedule; the deterministic + MPC line also "
            "uses intraday re-optimisation."
        ),
        ha="left",
        fontsize=9.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.075)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "economic_strategy_comparison.png", dpi=220)
    fig.savefig(output_dir / "economic_strategy_comparison.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        project_root
        / "artifacts"
        / "experiments"
        / "economic_strategy_comparison_v1"
    )

    sources = _read_sources(project_root)
    daily, hourly = _select_strategies(sources)
    hourly = _reconstruct_hourly_costs(daily, hourly)
    daily, hourly = _add_plot_series(daily, hourly)
    _plot(daily, hourly, output_dir)

    daily.to_csv(output_dir / "daily_strategy_costs.csv", index=False)
    hourly.to_parquet(output_dir / "hourly_strategy_costs.parquet", index=False)
    summary = {
        "delivery_days": int(daily["delivery_date_local"].nunique()),
        "hourly_rolling_window": 6,
        "daily_rolling_window": 14,
        "stochastic_selection": (
            "bootstrap_cvar_lambda_0_1: previously selected risk/cost trade-off"
        ),
        "strategies": {
            spec.key: {
                "label": spec.label,
                "total_cost_eur": float(
                    daily.loc[daily["strategy"] == spec.key, "total_cost_eur"].sum()
                ),
                "saving_vs_rule_eur": float(
                    daily.loc[
                        daily["strategy"] == spec.key,
                        "saving_vs_rule_eur",
                    ].sum()
                ),
            }
            for spec in STRATEGIES
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

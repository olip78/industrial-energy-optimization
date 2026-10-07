"""Plot calibrated conditional price intervals around the largest surprise."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = (
        root
        / "artifacts"
        / "experiments"
        / "cqr_copula_v1"
        / "test_calibrated_quantile_forecasts.parquet"
    )
    output_dir = (
        root
        / "artifacts"
        / "experiments"
        / "economic_backtest_v2_dynamic_charge"
    )
    frame = pd.read_parquet(source).copy()
    frame["delivery_date_local"] = pd.to_datetime(frame["delivery_date_local"])
    frame["timestamp_local"] = frame["delivery_date_local"] + pd.to_timedelta(
        frame["hour_local"], unit="h"
    )
    frame["median_error"] = (
        frame["actual_price_eur_per_mwh"] - frame["price_q50"]
    )
    largest = frame.loc[frame["median_error"].abs().idxmax()]
    centre = pd.Timestamp(largest["delivery_date_local"])
    start = centre - timedelta(days=7)
    end = centre + timedelta(days=7)
    window = frame.loc[
        frame["delivery_date_local"].between(start, end, inclusive="both")
    ].sort_values("timestamp_local")
    window.to_csv(output_dir / "cqr_price_interval_stress_window.csv", index=False)

    coverage_80 = float(
        frame["actual_price_eur_per_mwh"].between(
            frame["price_q10"], frame["price_q90"], inclusive="both"
        ).mean()
    )
    coverage_90 = float(
        frame["actual_price_eur_per_mwh"].between(
            frame["price_q05"], frame["price_q95"], inclusive="both"
        ).mean()
    )
    width_80 = float((frame["price_q90"] - frame["price_q10"]).mean())
    width_90 = float((frame["price_q95"] - frame["price_q05"]).mean())

    figure, axis = plt.subplots(figsize=(16, 7.8))
    figure.patch.set_facecolor("#FAFAF8")
    axis.set_facecolor("#FAFAF8")
    axis.grid(axis="y", color="#D8D8D4", linewidth=0.8, alpha=0.85)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#B8B8B2")
    axis.fill_between(
        window["timestamp_local"],
        window["price_q05"],
        window["price_q95"],
        color="#7BA6DF",
        alpha=0.16,
        linewidth=0,
        label="P05–P95",
    )
    axis.fill_between(
        window["timestamp_local"],
        window["price_q10"],
        window["price_q90"],
        color="#3978C5",
        alpha=0.24,
        linewidth=0,
        label="P10–P90",
    )
    axis.plot(
        window["timestamp_local"],
        window["price_q50"],
        color="#2F6BBD",
        linewidth=1.25,
        label="Conditional median P50",
    )
    axis.plot(
        window["timestamp_local"],
        window["actual_price_eur_per_mwh"],
        color="#202020",
        linewidth=1.35,
        label="Actual day-ahead price",
    )
    axis.axvline(
        largest["timestamp_local"],
        color="#C4514A",
        linewidth=1.0,
        linestyle="--",
        alpha=0.9,
    )
    axis.scatter(
        [largest["timestamp_local"]],
        [largest["actual_price_eur_per_mwh"]],
        color="#C4514A",
        s=32,
        zorder=5,
    )
    axis.set_title(
        "CQR price forecast around the largest 2025 surprise",
        loc="left",
        fontsize=19,
        fontweight="bold",
        pad=24,
    )
    axis.text(
        0.0,
        1.015,
        (
            "Conditional quantile regression · conformal calibration · "
            "empirical-copula scenarios for stochastic optimization"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color="#4A4A48",
    )
    axis.set_ylabel("Day-ahead price, EUR/MWh")
    axis.set_xlabel("Local delivery time")
    axis.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axis.tick_params(axis="x", rotation=35)
    axis.legend(
        loc="upper left",
        ncol=4,
        frameon=True,
        framealpha=0.94,
        facecolor="#FFFFFF",
        edgecolor="#D8D8D4",
        fontsize=9.5,
    )
    actual = float(largest["actual_price_eur_per_mwh"])
    median = float(largest["price_q50"])
    axis.text(
        0.995,
        0.025,
        (
            f"Largest |actual − P50|: {abs(actual - median):.1f} EUR/MWh "
            f"at {pd.Timestamp(largest['timestamp_local']):%d %b %H:%M}\n"
            f"Actual {actual:.1f} · P50 {median:.1f} EUR/MWh\n"
            f"2025 coverage: P10–P90 {coverage_80:.1%} "
            f"(mean width {width_80:.1f}) · P05–P95 {coverage_90:.1%} "
            f"(mean width {width_90:.1f})"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.3,
        color="#4A4A48",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#FFFFFF",
            "edgecolor": "#D8D8D4",
            "alpha": 0.94,
        },
    )
    figure.tight_layout(pad=2.2)
    figure.savefig(output_dir / "cqr_price_intervals_largest_surprise.png", dpi=220)
    figure.savefig(output_dir / "cqr_price_intervals_largest_surprise.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()

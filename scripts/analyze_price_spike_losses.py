"""Measure economic losses on unexpected day-ahead price spikes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STOCHASTIC = "cqr_copula_cvar_lambda_0_1"
DETERMINISTIC_DA = "day_ahead_only"
ORACLE = "oracle"
REQUIRED_STRATEGIES = (STOCHASTIC, DETERMINISTIC_DA, ORACLE)


def _local_date(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["delivery_date_local"]).dt.strftime("%Y-%m-%d")


def _reconstruct_hourly_costs(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    hourly = hourly.copy()
    daily = daily.copy()
    hourly["delivery_date_local"] = _local_date(hourly)
    daily["delivery_date_local"] = _local_date(daily)
    hourly["hour_local"] = hourly["hour_local"].astype(int)

    daily_cost = daily[
        [
            "delivery_date_local",
            "strategy",
            "battery_reference_cost_eur",
            "battery_discharge_kwh",
            "total_cost_eur",
        ]
    ].copy()
    daily_cost["battery_unit_cost_eur_per_kwh"] = 0.0
    discharged = daily_cost["battery_discharge_kwh"] > 1e-12
    daily_cost.loc[discharged, "battery_unit_cost_eur_per_kwh"] = (
        daily_cost.loc[discharged, "battery_reference_cost_eur"]
        / daily_cost.loc[discharged, "battery_discharge_kwh"]
    )
    hourly = hourly.merge(
        daily_cost[
            ["delivery_date_local", "strategy", "battery_unit_cost_eur_per_kwh"]
        ],
        on=["delivery_date_local", "strategy"],
        how="left",
        validate="many_to_one",
    )
    hourly["deviation_kwh"] = (
        hourly["load_kwh"]
        + hourly["battery_charge_kwh"]
        - hourly["actual_pv_kwh"]
        - hourly["battery_discharge_kwh"]
        - hourly["day_ahead_position_kwh"]
    )
    hourly["reconstructed_hourly_cost_eur"] = (
        hourly["actual_day_ahead_price_eur_per_mwh"]
        * hourly["day_ahead_position_kwh"]
        / 1_000.0
        + hourly["actual_intraday_price_eur_per_mwh"]
        * hourly["deviation_kwh"]
        / 1_000.0
        + hourly["battery_discharge_kwh"]
        * hourly["battery_unit_cost_eur_per_kwh"]
    )

    check = hourly.groupby(
        ["delivery_date_local", "strategy"], as_index=False
    )["reconstructed_hourly_cost_eur"].sum()
    check = check.merge(
        daily[["delivery_date_local", "strategy", "total_cost_eur"]],
        on=["delivery_date_local", "strategy"],
        validate="one_to_one",
    )
    max_error = float(
        (
            check["reconstructed_hourly_cost_eur"] - check["total_cost_eur"]
        ).abs().max()
    )
    if max_error > 1e-8:
        raise ValueError(
            f"Hourly cost reconstruction differs from daily output by {max_error:.3g}"
        )
    return hourly, max_error


def _event_summary(frame: pd.DataFrame, flag: str) -> dict[str, float | int]:
    events = frame.loc[frame[flag]].copy()
    event_days = events.drop_duplicates("delivery_date_local")
    return {
        "event_hours": int(len(events)),
        "event_days": int(events["delivery_date_local"].nunique()),
        "signed_direct_price_effect_eur": float(
            events["direct_price_surprise_eur"].sum()
        ),
        "gross_adverse_direct_price_effect_eur": float(
            events["direct_price_surprise_eur"].clip(lower=0).sum()
        ),
        "event_hour_regret_vs_oracle_eur": float(
            events["hourly_regret_vs_oracle_eur"].sum()
        ),
        "event_hour_delta_vs_deterministic_da_eur": float(
            events["hourly_delta_vs_deterministic_da_eur"].sum()
        ),
        "event_day_regret_vs_oracle_eur": float(
            event_days["daily_regret_vs_oracle_eur"].sum()
        ),
        "event_day_delta_vs_deterministic_da_eur": float(
            event_days["daily_delta_vs_deterministic_da_eur"].sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--jump-quantile",
        type=float,
        default=0.95,
        help="Quantile of positive 2024 one-hour increases used as the jump threshold.",
    )
    parser.add_argument(
        "--large-breach-eur-per-mwh",
        type=float,
        default=50.0,
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    artifact_dir = (
        root / "artifacts" / "experiments" / "economic_backtest_v2_dynamic_charge"
    )
    quantile_path = (
        root
        / "artifacts"
        / "experiments"
        / "cqr_copula_v1"
        / "test_calibrated_quantile_forecasts.parquet"
    )

    forecasts = pd.read_parquet(quantile_path).copy()
    forecasts["delivery_date_local"] = _local_date(forecasts)
    forecasts["hour_local"] = forecasts["hour_local"].astype(int)
    analysis_end = forecasts["delivery_date_local"].max()

    history = pd.read_parquet(
        root / "data" / "curated" / "canonical_hourly_2024.parquet"
    ).copy()
    history["valid_time_utc"] = pd.to_datetime(history["valid_time_utc"], utc=True)
    history = history.sort_values("valid_time_utc")
    history["price_jump_1h_eur_per_mwh"] = history[
        "day_ahead_price_eur_per_mwh"
    ].diff()
    positive_jumps = history.loc[
        history["price_jump_1h_eur_per_mwh"] > 0,
        "price_jump_1h_eur_per_mwh",
    ]
    jump_threshold = float(positive_jumps.quantile(args.jump_quantile))

    actuals = pd.read_parquet(
        root / "data" / "curated" / "canonical_hourly_2025.parquet"
    ).copy()
    actuals["delivery_date_local"] = _local_date(actuals)
    actuals["hour_local"] = actuals["hour_local"].astype(int)
    actuals["valid_time_utc"] = pd.to_datetime(actuals["valid_time_utc"], utc=True)
    actuals = actuals.loc[
        actuals["delivery_date_local"] <= analysis_end
    ].sort_values("valid_time_utc")
    actuals["price_jump_1h_eur_per_mwh"] = actuals[
        "day_ahead_price_eur_per_mwh"
    ].diff()
    jump_frame = actuals[
        ["delivery_date_local", "hour_local", "price_jump_1h_eur_per_mwh"]
    ].copy()
    if jump_frame.duplicated(["delivery_date_local", "hour_local"]).any():
        raise ValueError("Duplicate local delivery hour inside the analysis window")

    events = forecasts.merge(
        jump_frame,
        on=["delivery_date_local", "hour_local"],
        how="left",
        validate="one_to_one",
    )
    events["price_surprise_vs_q50_eur_per_mwh"] = (
        events["actual_price_eur_per_mwh"] - events["price_q50"]
    )
    events["price_breach_q95_eur_per_mwh"] = (
        events["actual_price_eur_per_mwh"] - events["price_q95"]
    )
    events["q95_breach"] = events["price_breach_q95_eur_per_mwh"] > 0
    events["large_q95_breach"] = (
        events["price_breach_q95_eur_per_mwh"]
        >= args.large_breach_eur_per_mwh
    )
    events["unexpected_sudden_spike"] = events["q95_breach"] & (
        events["price_jump_1h_eur_per_mwh"] > jump_threshold
    )

    hourly = pd.read_parquet(artifact_dir / "hourly_decisions.parquet")
    daily = pd.read_csv(artifact_dir / "daily_results.csv")
    available = set(hourly["strategy"].unique())
    missing = set(REQUIRED_STRATEGIES) - available
    if missing:
        raise ValueError(f"Missing strategies: {sorted(missing)}")
    hourly, reconstruction_error = _reconstruct_hourly_costs(hourly, daily)
    daily["delivery_date_local"] = _local_date(daily)

    wide = hourly.pivot_table(
        index=["delivery_date_local", "hour_local"],
        columns="strategy",
        values=["reconstructed_hourly_cost_eur", "day_ahead_position_kwh"],
        aggfunc="first",
    )
    wide.columns = [f"{measure}__{strategy}" for measure, strategy in wide.columns]
    wide = wide.reset_index()
    events = events.merge(
        wide,
        on=["delivery_date_local", "hour_local"],
        how="left",
        validate="one_to_one",
    )

    events["direct_price_surprise_eur"] = (
        events[f"day_ahead_position_kwh__{STOCHASTIC}"]
        * events["price_surprise_vs_q50_eur_per_mwh"]
        / 1_000.0
    )
    events["hourly_regret_vs_oracle_eur"] = (
        events[f"reconstructed_hourly_cost_eur__{STOCHASTIC}"]
        - events[f"reconstructed_hourly_cost_eur__{ORACLE}"]
    )
    events["hourly_delta_vs_deterministic_da_eur"] = (
        events[f"reconstructed_hourly_cost_eur__{STOCHASTIC}"]
        - events[f"reconstructed_hourly_cost_eur__{DETERMINISTIC_DA}"]
    )

    daily_wide = daily.pivot(
        index="delivery_date_local", columns="strategy", values="total_cost_eur"
    ).reset_index()
    daily_wide["daily_regret_vs_oracle_eur"] = (
        daily_wide[STOCHASTIC] - daily_wide[ORACLE]
    )
    daily_wide["daily_delta_vs_deterministic_da_eur"] = (
        daily_wide[STOCHASTIC] - daily_wide[DETERMINISTIC_DA]
    )
    events = events.merge(
        daily_wide[
            [
                "delivery_date_local",
                "daily_regret_vs_oracle_eur",
                "daily_delta_vs_deterministic_da_eur",
            ]
        ],
        on="delivery_date_local",
        how="left",
        validate="many_to_one",
    )

    event_hours = events.loc[events["q95_breach"]].copy()
    event_hours.to_csv(artifact_dir / "price_spike_hourly_analysis.csv", index=False)

    strict = events.loc[events["unexpected_sudden_spike"]].copy()
    strict["gross_adverse_direct_price_effect_eur"] = strict[
        "direct_price_surprise_eur"
    ].clip(lower=0)
    event_days = (
        strict.groupby("delivery_date_local", as_index=False)
        .agg(
            event_hours=("unexpected_sudden_spike", "size"),
            max_actual_price_eur_per_mwh=("actual_price_eur_per_mwh", "max"),
            max_q95_eur_per_mwh=("price_q95", "max"),
            max_q95_breach_eur_per_mwh=("price_breach_q95_eur_per_mwh", "max"),
            max_one_hour_jump_eur_per_mwh=("price_jump_1h_eur_per_mwh", "max"),
            signed_direct_price_effect_eur=("direct_price_surprise_eur", "sum"),
            gross_adverse_direct_price_effect_eur=(
                "gross_adverse_direct_price_effect_eur",
                "sum",
            ),
            event_hour_regret_vs_oracle_eur=(
                "hourly_regret_vs_oracle_eur",
                "sum",
            ),
            event_hour_delta_vs_deterministic_da_eur=(
                "hourly_delta_vs_deterministic_da_eur",
                "sum",
            ),
        )
        .merge(
            daily_wide[
                [
                    "delivery_date_local",
                    STOCHASTIC,
                    DETERMINISTIC_DA,
                    ORACLE,
                    "daily_regret_vs_oracle_eur",
                    "daily_delta_vs_deterministic_da_eur",
                ]
            ],
            on="delivery_date_local",
            validate="one_to_one",
        )
        .sort_values("daily_regret_vs_oracle_eur", ascending=False)
    )
    event_days.to_csv(artifact_dir / "price_spike_daily_analysis.csv", index=False)

    summaries = {
        "q95_breach": _event_summary(events, "q95_breach"),
        "large_q95_breach": _event_summary(events, "large_q95_breach"),
        "unexpected_sudden_spike": _event_summary(
            events, "unexpected_sudden_spike"
        ),
    }
    strict_summary = summaries["unexpected_sudden_spike"]
    strict_dates = set(strict["delivery_date_local"])
    total_regret = float(daily_wide["daily_regret_vs_oracle_eur"].sum())
    event_regret = float(strict_summary["event_day_regret_vs_oracle_eur"])
    event_hour_regret = float(strict_summary["event_hour_regret_vs_oracle_eur"])
    summary = {
        "definition": {
            "unexpected_condition": "actual day-ahead price > calibrated conditional P95",
            "sudden_condition": (
                "one-hour increase > the selected quantile of positive one-hour "
                "increases observed in 2024"
            ),
            "jump_quantile": args.jump_quantile,
            "jump_threshold_eur_per_mwh": jump_threshold,
            "large_q95_breach_threshold_eur_per_mwh": args.large_breach_eur_per_mwh,
        },
        "analysis": {
            "first_date": forecasts["delivery_date_local"].min(),
            "last_date": analysis_end,
            "working_hour_rows": int(len(events)),
            "delivery_days": int(daily_wide["delivery_date_local"].nunique()),
            "max_hourly_cost_reconstruction_error_eur": reconstruction_error,
        },
        "event_summaries": summaries,
        "strict_spike_diagnostics": {
            "event_hour_share": float(strict_summary["event_hours"] / len(events)),
            "event_day_share": float(
                strict_summary["event_days"]
                / daily_wide["delivery_date_local"].nunique()
            ),
            "total_period_regret_vs_oracle_eur": total_regret,
            "share_of_total_regret_on_event_days": float(event_regret / total_regret),
            "share_of_event_day_regret_at_event_hours": float(
                event_hour_regret / event_regret
            ),
            "mean_event_day_regret_vs_oracle_eur": float(
                daily_wide.loc[
                    daily_wide["delivery_date_local"].isin(strict_dates),
                    "daily_regret_vs_oracle_eur",
                ].mean()
            ),
            "mean_non_event_day_regret_vs_oracle_eur": float(
                daily_wide.loc[
                    ~daily_wide["delivery_date_local"].isin(strict_dates),
                    "daily_regret_vs_oracle_eur",
                ].mean()
            ),
        },
    }
    (artifact_dir / "price_spike_loss_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

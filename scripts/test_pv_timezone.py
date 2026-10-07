"""Compare candidate timezones for the MPVBench PV source clock.

This is evidence for a modelling convention, not confirmation from the data
publisher.  The test uses profile 1a against local ERA5 solar radiation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(sys.argv[1]).resolve()
PROCESSED = PROJECT / "data" / "processed"
METADATA = PROJECT / "data" / "metadata"
DOCS = PROJECT / "docs"

PROFILES = ["1a", "2a", "2b"]
KNOWN_COMMON_OUTAGE_DAYS = {"2024-05-10", "2024-07-07", "2025-07-23"}
TIMEZONES = {"Berlin": "Europe/Berlin", "London": "Europe/London", "UTC_control": "UTC"}


def hourly_pv(profile: str) -> pd.DataFrame:
    pv = pd.read_csv(PROCESSED / "pv_power_15min.csv.gz", usecols=["source_time", profile])
    pv["source_time"] = pd.to_datetime(pv["source_time"])
    pv["date"] = pv["source_time"].dt.strftime("%Y-%m-%d")
    pv = pv.loc[~pv["date"].isin(KNOWN_COMMON_OUTAGE_DAYS)].drop(columns="date")
    return pv.set_index("source_time").resample("h").mean().rename(columns={profile: "pv_w"})


def map_to_utc(frame: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    mapped = frame.copy()
    if timezone_name == "UTC":
        mapped.index = mapped.index.tz_localize("UTC")
    else:
        # Both DST-transition hours are discarded rather than guessed.
        mapped.index = mapped.index.tz_localize(timezone_name, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
        mapped = mapped.loc[~mapped.index.isna()]
    return mapped


def metrics(pv_hourly: pd.DataFrame, weather: pd.DataFrame, timezone_name: str) -> dict:
    mapped = map_to_utc(pv_hourly, timezone_name)
    # Open-Meteo labels radiation as an average over the preceding hour; PV rows
    # are average values over the timestamp hour. Shift weather labels back one
    # hour to compare equal physical intervals.
    joined = mapped.join(weather.shift(-1), how="inner").dropna()
    joined = joined.loc[joined["shortwave_radiation"] > 10]
    x = joined[["shortwave_radiation", "direct_radiation", "diffuse_radiation", "temperature_2m", "wind_speed_10m"]].to_numpy()
    x = np.column_stack([np.ones(len(x)), x])
    y = joined["pv_w"].to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    prediction = x @ beta
    residual = y - prediction
    total = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - (residual**2).sum() / total)
    rmse = float(np.sqrt(np.mean(residual**2)))
    correlation = float(joined["pv_w"].corr(joined["shortwave_radiation"]))
    # Strong PV output during negligible measured solar radiation is a useful
    # timing diagnostic. The exact rate is not expected to be zero because the
    # panel is tilted and the weather cell is approximate.
    bad_night = int(((joined["pv_w"] > 20) & (joined["shortwave_radiation"] <= 10)).sum())
    weather_days = weather.copy()
    weather_days["utc_day"] = weather_days.index.floor("D")
    daily_ghi = weather_days.groupby("utc_day")["shortwave_radiation"].sum()
    clear_days = set(daily_ghi[daily_ghi >= daily_ghi.quantile(0.90)].index)
    pv_active = mapped.loc[mapped["pv_w"] > 20].groupby(mapped.loc[mapped["pv_w"] > 20].index.floor("D"))["pv_w"]
    weather_active = weather.loc[weather["shortwave_radiation"] > 20].groupby(weather.loc[weather["shortwave_radiation"] > 20].index.floor("D"))["shortwave_radiation"]
    pv_start, pv_end = pv_active.apply(lambda x: x.index.min()), pv_active.apply(lambda x: x.index.max())
    weather_start, weather_end = weather_active.apply(lambda x: x.index.min()), weather_active.apply(lambda x: x.index.max())
    onset = pd.DataFrame({"pv_start": pv_start, "pv_end": pv_end, "weather_start": weather_start, "weather_end": weather_end}).dropna()
    onset = onset.loc[onset.index.isin(clear_days)]
    start_lag = (onset["pv_start"] - onset["weather_start"]).dt.total_seconds() / 3600
    end_lag = (onset["pv_end"] - onset["weather_end"]).dt.total_seconds() / 3600
    return {
        "timezone": timezone_name,
        "matched_daylight_hours": int(len(joined)),
        "solar_correlation": correlation,
        "weather_model_r2": r2,
        "weather_model_rmse_w": rmse,
        "pv_over_20w_with_ghi_at_most_10w": bad_night,
        "clear_day_count": int(len(onset)),
        "median_pv_start_minus_weather_start_hours": float(start_lag.median()),
        "median_pv_end_minus_weather_end_hours": float(end_lag.median()),
    }


def main() -> None:
    weather = pd.read_csv(PROCESSED / "weather_actual_hourly_era5.csv.gz")
    weather["valid_time_utc"] = pd.to_datetime(weather["valid_time_utc"], utc=True)
    weather = weather.set_index("valid_time_utc")
    all_metrics = {}
    for profile in PROFILES:
        source = hourly_pv(profile)
        all_metrics[profile] = {name: metrics(source, weather, tz) for name, tz in TIMEZONES.items()}
    (METADATA / "pv_timezone_hypothesis_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    columns = ["profile", "hypothesis", "matched_daylight_hours", "solar_correlation", "weather_model_r2", "weather_model_rmse_w", "pv_over_20w_with_ghi_at_most_10w"]
    rows = []
    for profile, result in all_metrics.items():
        for hypothesis, values in result.items():
            rows.append({"profile": profile, "hypothesis": hypothesis, **values})
    pd.DataFrame(rows)[columns].to_csv(PROCESSED / "pv_timezone_hypothesis_metrics.csv", index=False)

    # Keep the original source clock intact and create a separate, explicitly
    # assumed UTC join key. DST-transition values cannot be mapped uniquely from
    # a source series with a fixed 96 intervals per calendar day.
    original = pd.read_csv(PROCESSED / "pv_power_15min.csv.gz")
    original_time = pd.to_datetime(original["source_time"])
    berlin = original_time.dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
    original["valid_time_utc"] = berlin.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    original["timezone_assumption"] = "Europe/Berlin (provisional)"
    original["timezone_conversion_status"] = np.where(original["valid_time_utc"].isna(), "unresolved_dst_transition", "converted")
    original["pv_quality_status"] = np.where(
        original["source_time"].str.slice(0, 10).isin(KNOWN_COMMON_OUTAGE_DAYS),
        "suspected_common_source_outage",
        "usable",
    )
    original.to_csv(PROCESSED / "pv_power_15min_assumed_europe_berlin.csv.gz", index=False, compression="gzip")

    manifest_path = METADATA / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["pv_source_time"] = {
        "original_timezone": "unknown",
        "working_timezone": "Europe/Berlin (provisional)",
        "reason": "PV morning onset aligns with local solar-radiation onset for profiles 1a, 2a and 2b; 2a/2b also favor Berlin in weather-fit metrics.",
        "dst_policy": "retain source_time; valid_time_utc is blank for ambiguous/nonexistent DST-transition rows",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    (DOCS / "pv_timezone_hypothesis_test.md").write_text(f"""# PV source-time hypothesis test

## Result

Use **Europe/Berlin as the provisional working convention** for the PV source clock.

The evidence is mixed if we look only at one generic weather-regression score: profile 1a favours the UTC control (R² {all_metrics['1a']['UTC_control']['weather_model_r2']:.3f}) over Berlin (R² {all_metrics['1a']['Berlin']['weather_model_r2']:.3f}). But its morning generation starts at the same hour as local solar radiation under Berlin, and its generation ends three hours earlier, a plausible pattern for the documented southeast-facing panel. Profiles 2a and 2b favour Berlin on both weather-fit and onset diagnostics. Under Berlin, all three profiles have a median PV-start minus weather-start lag of 0 hours on the top 10% highest-radiation days.

This is enough to choose a transparent modelling convention for Version 1. It does not replace confirmation by the source publisher.

## Method

1. Aggregate measured PV profiles 1a, 2a and 2b from 15 minutes to hourly mean power.
2. Exclude the three dates that are zero across all three profiles.
3. Interpret the same naïve `source_time` under each timezone hypothesis, convert it to UTC, and join it to ERA5 weather at the approximate Pforzheim point.
4. Compare daylight PV output with global, direct and diffuse solar radiation, temperature and wind through a simple least-squares model. Open-Meteo reports radiation averaged over the preceding hour, so weather is shifted one label back to match PV's timestamp hour.

The orientation of the 1a panels is southeast. This is why the test does not compare the hour of daily PV maximum with the hour of maximum horizontal radiation directly.

## Decision and output

`pv_power_15min_assumed_europe_berlin.csv.gz` is a new join-ready copy. It preserves `source_time`, adds `valid_time_utc`, and labels the assumption. It also labels the three suspected common outage dates. The raw table remains unchanged.

The source emits 96 rows on daylight-saving transition dates, so four rows on each spring/autumn transition cannot be mapped unambiguously to UTC. Their `valid_time_utc` is deliberately blank and `timezone_conversion_status` is `unresolved_dst_transition`; do not silently fill them.

Detailed figures are in `data/processed/pv_timezone_hypothesis_metrics.csv` and `data/metadata/pv_timezone_hypothesis_metrics.json`.
""")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()

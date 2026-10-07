"""Collect a reproducible first data layer for the Energy project.

The forecast archive uses ECMWF IFS single runs.  Open-Meteo documents the
2024–2025 archive as Cycle 49R1 hindcasts, so this output is explicitly a
retrospective forecast scenario, not evidence of a live operational feed.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

PROJECT = Path(sys.argv[1]).resolve()
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
METADATA = PROJECT / "data" / "metadata"

LATITUDE, LONGITUDE = 48.89, 8.70
START, END = date(2024, 1, 1), date(2025, 12, 31)
FORECAST_START = date(2024, 1, 1)
WEATHER_VARIABLES = "temperature_2m,shortwave_radiation,direct_radiation,diffuse_radiation,wind_speed_10m,cloud_cover"


def fetch(url: str, attempts: int = 4) -> dict | bytes:
    request = Request(url, headers={"User-Agent": "energy-pet-project-data-collection/0.1"})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=90) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read()
                return json.loads(body) if "json" in content_type or body.startswith(b"{") else body
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"{url}: {error}") from error
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def collect_static_sources() -> None:
    pv_url = "https://api.github.com/repos/KIT-IAI/MPVBench/git/blobs/b85464d1f14244a988ec868c234a63f9b043a87e"
    response = fetch(pv_url)
    if not isinstance(response, dict) or "content" not in response:
        raise RuntimeError("Unexpected GitHub PV response")
    import base64
    (RAW / "mpvbench_p_watt_15min.csv").write_bytes(base64.b64decode(response["content"]))
    for name, url in {
        "mpvbench_01a_metadata.md": "https://api.github.com/repos/KIT-IAI/MPVBench/git/blobs/fcc555b5fc81d2b0ccbbfc089111901c25e3e4df",
        "mpvbench_license.txt": "https://api.github.com/repos/KIT-IAI/MPVBench/contents/LICENSE",
    }.items():
        payload = fetch(url)
        if isinstance(payload, dict) and "content" in payload:
            (RAW / name).write_bytes(base64.b64decode(payload["content"]))


def collect_actual_weather() -> None:
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}&start_date={START}&end_date={END}"
        f"&hourly={WEATHER_VARIABLES}&models=era5&timezone=UTC"
    )
    payload = fetch(url)
    assert isinstance(payload, dict)
    write_json(RAW / "weather_actual_era5.json", payload)


def collect_prices() -> None:
    url = f"https://api.energy-charts.info/price?bzn=DE-LU&start={START}&end={END}"
    payload = fetch(url)
    assert isinstance(payload, dict)
    write_json(RAW / "prices_day_ahead_de_lu.json", payload)


def forecast_url(period_start: date, period_end: date) -> str:
    return (
        "https://previous-runs-api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}&start_date={period_start}&end_date={period_end}"
        "&hourly=" + ",".join(
            f"{variable}_previous_day{lead}"
            for variable in WEATHER_VARIABLES.split(",")
            for lead in (1, 2)
        ) + "&models=icon_seamless&timezone=UTC"
    )


def collect_forecasts() -> None:
    forecast_dir = RAW / "weather_forecasts_icon_previous_runs_monthly"
    forecast_dir.mkdir(exist_ok=True)
    periods = []
    current = FORECAST_START
    while current <= END:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        period_end = min(END, next_month - timedelta(days=1))
        path = forecast_dir / f"{current.isoformat()}.json"
        if not path.exists():
            periods.append((current, period_end))
        current = next_month
    print(f"Forecast months to fetch: {len(periods)}", flush=True)
    failures: dict[str, str] = {}
    for done, (period_start, period_end) in enumerate(periods, 1):
        try:
            time.sleep(2)
            payload = fetch(forecast_url(period_start, period_end), attempts=2)
            if not isinstance(payload, dict) or "hourly" not in payload:
                raise RuntimeError("Unexpected previous-runs response")
            write_json(forecast_dir / f"{period_start.isoformat()}.json", payload)
        except Exception as error:
            failures[period_start.isoformat()] = str(error)
        print(f"Forecast months complete: {done}/{len(periods)}; failures: {len(failures)}", flush=True)
    write_json(METADATA / "forecast_download_failures.json", failures)
    if failures:
        raise RuntimeError(f"Forecast downloads failed for {len(failures)} runs")


def build_processed_tables() -> None:
    pv = pd.read_csv(RAW / "mpvbench_p_watt_15min.csv")
    pv = pv.rename(columns={"time": "source_time"})
    pv["source_timezone"] = "unknown"
    pv["interval_minutes"] = 15
    pv.to_csv(PROCESSED / "pv_power_15min.csv.gz", index=False, compression="gzip")

    actual = json.loads((RAW / "weather_actual_era5.json").read_text())
    actual_df = pd.DataFrame(actual["hourly"]).rename(columns={"time": "valid_time_utc"})
    actual_df.to_csv(PROCESSED / "weather_actual_hourly_era5.csv.gz", index=False, compression="gzip")

    prices = json.loads((RAW / "prices_day_ahead_de_lu.json").read_text())
    price_df = pd.DataFrame({
        "valid_time_utc": pd.to_datetime(prices["unix_seconds"], unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "day_ahead_price_eur_per_mwh": prices["price"],
    })
    price_df.to_csv(PROCESSED / "prices_day_ahead_de_lu.csv.gz", index=False, compression="gzip")

    rows: list[dict] = []
    forecast_dir = RAW / "weather_forecasts_icon_previous_runs_monthly"
    for path in sorted(forecast_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        hourly = payload["hourly"]
        for index, valid_time in enumerate(hourly["time"]):
            row = {"valid_time_utc": valid_time}
            for variable in hourly:
                if variable != "time":
                    row[variable] = hourly[variable][index]
            rows.append(row)
    forecast_df = pd.DataFrame(rows)
    forecast_df.to_csv(PROCESSED / "weather_forecast_hourly_icon_lead_24_48.csv.gz", index=False, compression="gzip")


def write_readme() -> None:
    (PROJECT / "README.md").write_text("""# Energy optimization project

## Data layer created 2026-09-18

The project stores original data under `data/raw` and analysis-ready tables under `data/processed`.

- `pv_power_15min.csv.gz`: measured KIT MPVBench active power for five PV profiles. The selected working profile is `1a`; its source timestamps have an **unknown timezone**.
- `weather_actual_hourly_era5.csv.gz`: ERA5 reanalysis for an approximate Pforzheim location (48.89, 8.70), used as a historical weather estimate.
- `weather_forecast_hourly_icon_lead_24_48.csv.gz`: archived ICON weather forecasts at fixed 24- and 48-hour lead times. It supports day-ahead-style experiments, but is not a complete forecast trajectory from one historical run.
- `prices_day_ahead_de_lu.csv.gz`: DE-LU day-ahead auction outcomes in EUR/MWh. The series is hourly before 1 October 2025 and 15-minute thereafter. These are settled auction prices, not a forecast available before bidding or intraday prices.

The approximate Pforzheim coordinate is a modelling choice because KIT metadata gives district/city rather than confirmed panel coordinates. It is not a claim about the actual location of profile 1a.

To refresh data, run `scripts/collect_energy_data.py` with the project directory as its argument. The collector is resumable for forecast runs.
""")


def write_manifest() -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "location": {"latitude": LATITUDE, "longitude": LONGITUDE, "status": "approximate point near Pforzheim, not verified PV coordinates"},
        "time_window": {"pv": "2024-01-01 through 2025-12-31", "actual_weather": "2024-01-01 through 2025-12-31", "prices": "2024-01-01 through 2025-12-31", "forecasts": "2024-01-01 through 2025-12-31"},
        "forecast": {"provider": "Open-Meteo", "model": "DWD ICON seamless", "format": "previous-runs fixed lead times", "lead_hours": [24, 48], "historical_status": "forecast values indexed by valid time and lead, not full individual forecast runs"},
        "sources": {"pv": "https://github.com/KIT-IAI/MPVBench", "actual_weather": "https://archive-api.open-meteo.com/v1/archive", "forecast_weather": "https://single-runs-api.open-meteo.com/v1/forecast", "prices": "https://api.energy-charts.info/price"},
    }
    write_json(METADATA / "data_manifest.json", manifest)


def main() -> None:
    for directory in (RAW, PROCESSED, METADATA):
        directory.mkdir(parents=True, exist_ok=True)
    collect_static_sources()
    collect_actual_weather()
    collect_prices()
    collect_forecasts()
    build_processed_tables()
    write_manifest()
    write_readme()
    print("Collection complete", flush=True)


if __name__ == "__main__":
    main()

"""Parallel multi-site PV training dataset built from the MPVBench profiles.

The source does not publish confirmed panel coordinates for profiles 2a and 2b.
Consequently, this dataset keeps one shared Pforzheim weather/solar proxy and
uses a one-hot site identifier.  It does not overwrite the single-site 1a
training tables.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ELIGIBLE_PROFILES = ("1a", "2a", "2b")
EXCLUDED_PROFILES = {
    "1b": "147 all-zero days in the available period",
    "1c": "313 all-zero days in the available period",
}
SHARED_PROXY = {
    "latitude": 48.89,
    "longitude": 8.70,
    "timezone": "Europe/Berlin",
    "provenance": "Shared weather/solar proxy; confirmed panel coordinates unavailable",
}
DATASET_NAME = "pv_multisite_v1"


@dataclass(frozen=True)
class MultiSitePVBuildResult:
    """Locations and lineage of a materialised multi-site training table."""

    data_path: Path
    manifest_path: Path
    site_metadata_path: Path
    rows_by_site: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("data_path", "manifest_path", "site_metadata_path"):
            result[key] = str(result[key])
        return result


def build_multisite_day_ahead_pv(
    project_root: str | Path,
    *,
    year: int = 2024,
    profiles: tuple[str, ...] = ELIGIBLE_PROFILES,
) -> MultiSitePVBuildResult:
    """Materialise a long multi-site view in a dedicated parallel directory.

    The existing 1a day-ahead table supplies the leakage-safe temporal,
    calendar and weather columns. Each eligible MPVBench profile supplies its
    own PV target after the same provisional Europe/Berlin conversion.
    """

    root = Path(project_root).expanduser().resolve()
    selected_profiles = _validate_profiles(profiles)
    base_path = root / "data" / "features" / "day_ahead_pv" / f"day_ahead_pv_{year}.parquet"
    pv_path = root / "data" / "processed" / "pv_power_15min_assumed_europe_berlin.csv.gz"
    if not base_path.exists():
        raise FileNotFoundError(f"Single-site day-ahead source is missing: {base_path}")
    if not pv_path.exists():
        raise FileNotFoundError(f"Converted MPVBench source is missing: {pv_path}")

    base = pd.read_parquet(base_path)
    pv = pd.read_csv(pv_path, low_memory=False)
    pv["valid_time_utc"] = pd.to_datetime(pv["valid_time_utc"], utc=True, errors="coerce")
    pv = pv.dropna(subset=["valid_time_utc"]).set_index("valid_time_utc")

    target_columns = {
        "pv_power_mean_w",
        "pv_energy_kwh",
        "pv_quality_status",
        "pv_source_profile",
        "pv_timezone_assumption",
        "target_available_at_utc",
    }
    feature_base = base.drop(columns=list(target_columns)).copy()
    frames: list[pd.DataFrame] = []
    for profile in selected_profiles:
        hourly = pv[profile].resample("1h").agg(["mean", "count"]).reset_index()
        hourly = hourly.rename(columns={"mean": "pv_power_mean_w", "count": "pv_intervals"})
        hourly["pv_energy_kwh"] = hourly["pv_power_mean_w"] / 1000.0
        hourly["target_available_at_utc"] = hourly["valid_time_utc"] + pd.Timedelta(hours=1)
        frame = feature_base.merge(hourly, on="valid_time_utc", how="inner", validate="one_to_one")
        # The base table has already excluded the common suspected outages and
        # unavailable weather / target rows using the established 1a contract.
        frame = frame.loc[(frame["pv_intervals"] == 4) & frame["pv_power_mean_w"].notna()].copy()
        frame["pv_quality_status"] = "usable"
        frame["pv_source_profile"] = profile
        frame["site_id"] = profile
        frame["pv_timezone_assumption"] = "Europe/Berlin (provisional)"
        frame["feature_version"] = "da-pv-multisite-v1"
        frame["data_snapshot_id"] = f"pv-multisite-v1-{year}"
        frames.append(frame.drop(columns="pv_intervals"))

    result = pd.concat(frames, ignore_index=True)
    for profile in selected_profiles:
        result[f"site_id_{profile}"] = (result["site_id"] == profile).astype("int8")
    result = result.sort_values(["site_id", "valid_time_utc"]).reset_index(drop=True)
    _validate_multisite_frame(result, selected_profiles)

    output_dir = root / "data" / "features" / DATASET_NAME
    metadata_dir = root / "data" / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"day_ahead_pv_multisite_{year}.parquet"
    manifest_path = metadata_dir / f"{DATASET_NAME}_{year}_manifest.json"
    site_metadata_path = metadata_dir / f"{DATASET_NAME}_sites.json"
    result.to_parquet(data_path, index=False)

    rows_by_site = {
        profile: int((result["site_id"] == profile).sum()) for profile in selected_profiles
    }
    site_metadata = _site_metadata(result, selected_profiles)
    site_metadata_path.write_text(json.dumps(site_metadata, indent=2))
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": DATASET_NAME,
                "year": year,
                "source_day_ahead_table": str(base_path.relative_to(root)),
                "source_pv_table": str(pv_path.relative_to(root)),
                "profiles": list(selected_profiles),
                "excluded_profiles": EXCLUDED_PROFILES,
                "weather_and_solar_proxy": SHARED_PROXY,
                "site_identifier_encoding": "one-hot numeric columns site_id_<profile>",
                "target_units": "W",
                "rows_by_site": rows_by_site,
                "columns": result.columns.tolist(),
                "data_sha256": _file_sha256(data_path),
            },
            indent=2,
        )
    )
    return MultiSitePVBuildResult(
        data_path=data_path,
        manifest_path=manifest_path,
        site_metadata_path=site_metadata_path,
        rows_by_site=rows_by_site,
    )


def _validate_profiles(profiles: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(profiles))
    if not selected:
        raise ValueError("At least one PV profile must be selected")
    unknown = sorted(set(selected).difference(ELIGIBLE_PROFILES))
    if unknown:
        raise ValueError(
            f"Profiles {unknown} are not eligible for multisite_v1; "
            f"available profiles: {list(ELIGIBLE_PROFILES)}"
        )
    return selected


def _validate_multisite_frame(frame: pd.DataFrame, profiles: tuple[str, ...]) -> None:
    if frame.empty:
        raise ValueError("Multi-site training table is empty")
    if frame.duplicated(["site_id", "valid_time_utc"]).any():
        raise ValueError("Multi-site table has duplicate site / timestamp rows")
    one_hot_columns = [f"site_id_{profile}" for profile in profiles]
    if not frame[one_hot_columns].sum(axis=1).eq(1).all():
        raise ValueError("Site one-hot columns are invalid")
    for profile in profiles:
        if not frame.loc[frame["site_id"] == profile, "pv_power_mean_w"].notna().all():
            raise ValueError(f"Profile {profile} has a missing PV target")


def _site_metadata(frame: pd.DataFrame, profiles: tuple[str, ...]) -> dict[str, object]:
    sites = []
    for profile in profiles:
        target = frame.loc[frame["site_id"] == profile, "pv_power_mean_w"]
        sites.append(
            {
                "site_id": profile,
                "pv_source_profile": profile,
                "include_in_multisite_v1": True,
                "observed_max_power_w": float(target.max()),
                "observed_p995_power_w": float(target.quantile(0.995)),
                "installed_power": (
                    "600 W microinverter documented only for profile 1a; "
                    "unknown for other profiles"
                    if profile == "1a"
                    else "not published in the available source metadata"
                ),
                "coordinates": SHARED_PROXY,
            }
        )
    return {
        "dataset_name": DATASET_NAME,
        "sites": sites,
        "note": (
            "Observed peak statistics are descriptive metadata, not model features. "
            "Do not treat them as confirmed installed power."
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

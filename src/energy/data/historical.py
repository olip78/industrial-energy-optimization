"""Historical tables exposed through point-in-time provider contracts."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from energy.data.contracts import DayAheadContext, MpcContext
from energy.data.builder import WEATHER_VARIABLES

UTC = "UTC"
LOCAL_TZ = "Europe/Berlin"


def _utc(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


class HistoricalDataProvider:
    """Read canonical Parquet while enforcing an explicit ``as_of`` boundary."""

    def __init__(self, canonical_path: str | Path) -> None:
        self.canonical_path = Path(canonical_path)
        self._frame: pd.DataFrame | None = None

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            frame = pd.read_parquet(self.canonical_path)
            time_columns = [column for column in frame if column.endswith("_utc")]
            for column in time_columns:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
            self._frame = frame.sort_values("valid_time_utc").reset_index(drop=True)
        return self._frame

    def get_day_ahead_context(
        self,
        as_of: datetime,
        delivery_day: date,
    ) -> DayAheadContext:
        as_of_utc = _utc(as_of)
        local_start = pd.Timestamp(delivery_day, tz=LOCAL_TZ)
        local_end = local_start + pd.DateOffset(days=1)
        start_utc = local_start.tz_convert(UTC)
        end_utc = local_end.tz_convert(UTC)

        delivery = self.frame.loc[
            (self.frame["valid_time_utc"] >= start_utc)
            & (self.frame["valid_time_utc"] < end_utc)
        ].copy()
        self._apply_availability_masks(delivery, as_of_utc)
        self._add_selected_weather(delivery)

        history = self.frame.loc[self.frame["valid_time_utc"] <= as_of_utc].copy()
        self._apply_availability_masks(history, as_of_utc)
        return DayAheadContext(
            as_of=as_of_utc.to_pydatetime(),
            delivery_day=delivery_day,
            delivery_rows=delivery,
            history_rows=history,
        )

    def get_mpc_context(
        self,
        as_of: datetime,
        horizon_end: datetime,
    ) -> MpcContext:
        as_of_utc = _utc(as_of)
        end_utc = _utc(horizon_end)
        observed = self.frame.loc[self.frame["valid_time_utc"] <= as_of_utc].copy()
        self._apply_availability_masks(observed, as_of_utc)
        future = self.frame.loc[
            (self.frame["valid_time_utc"] > as_of_utc)
            & (self.frame["valid_time_utc"] <= end_utc)
        ].copy()
        self._apply_availability_masks(future, as_of_utc)
        self._add_selected_weather(future)
        return MpcContext(
            as_of=as_of_utc.to_pydatetime(),
            horizon_end=end_utc.to_pydatetime(),
            observed_rows=observed,
            future_rows=future,
        )

    @staticmethod
    def _apply_availability_masks(frame: pd.DataFrame, as_of_utc: pd.Timestamp) -> None:
        """Hide values whose source had not published them at ``as_of_utc``."""

        groups = (
            (
                "pv_available_at_utc",
                [
                    "pv_power_mean_w",
                    "pv_energy_kwh",
                    "pv_intervals",
                    "pv_quality_status",
                ],
            ),
            (
                "weather_actual_available_at_utc",
                [
                    column
                    for column in frame
                    if column.startswith("weather_actual_")
                    and column != "weather_actual_available_at_utc"
                ],
            ),
            (
                "weather_fcst24_available_at_utc",
                [
                    column
                    for column in frame
                    if column.startswith("weather_fcst24_")
                    and column != "weather_fcst24_available_at_utc"
                ],
            ),
            (
                "weather_fcst48_available_at_utc",
                [
                    column
                    for column in frame
                    if column.startswith("weather_fcst48_")
                    and column != "weather_fcst48_available_at_utc"
                ],
            ),
            (
                "day_ahead_price_available_at_utc",
                ["day_ahead_price_eur_per_mwh"],
            ),
        )
        for availability_column, value_columns in groups:
            if availability_column not in frame:
                continue
            existing_columns = [column for column in value_columns if column in frame]
            unavailable = frame[availability_column].isna() | (
                frame[availability_column] > as_of_utc
            )
            frame.loc[unavailable, existing_columns] = pd.NA

    @staticmethod
    def _add_selected_weather(frame: pd.DataFrame) -> None:
        """Expose one online feature schema, preferring the fresher 24-hour layer."""

        columns_24 = [f"weather_fcst24_{variable}" for variable in WEATHER_VARIABLES]
        columns_48 = [f"weather_fcst48_{variable}" for variable in WEATHER_VARIABLES]
        if not set(columns_24 + columns_48).issubset(frame.columns):
            return
        use_24 = frame[columns_24].notna().all(axis=1)
        use_48 = ~use_24 & frame[columns_48].notna().all(axis=1)
        for variable in WEATHER_VARIABLES:
            frame[f"weather_forecast_{variable}"] = frame[
                f"weather_fcst24_{variable}"
            ].where(use_24, frame[f"weather_fcst48_{variable}"].where(use_48))
        frame["weather_forecast_lead_hours"] = pd.Series(
            pd.NA, index=frame.index, dtype="Int8"
        )
        frame.loc[use_24, "weather_forecast_lead_hours"] = 24
        frame.loc[use_48, "weather_forecast_lead_hours"] = 48
        frame["weather_forecast_available_at_utc"] = frame[
            "weather_fcst24_available_at_utc"
        ].where(use_24, frame["weather_fcst48_available_at_utc"].where(use_48))

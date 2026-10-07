"""Typed contracts exposed by the historical and future API providers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


@dataclass(frozen=True)
class DayAheadContext:
    """Data visible at one day-ahead decision time."""

    as_of: datetime
    delivery_day: date
    delivery_rows: pd.DataFrame
    history_rows: pd.DataFrame


@dataclass(frozen=True)
class MpcContext:
    """Observed history and future time grid visible to one MPC decision."""

    as_of: datetime
    horizon_end: datetime
    observed_rows: pd.DataFrame
    future_rows: pd.DataFrame


FEATURE_TIME_COLUMNS = {
    "weather_forecast_available_at_utc",
    "pv_last_observation_available_at_utc",
    "price_lag_24h_available_at_utc",
    "price_lag_168h_available_at_utc",
}


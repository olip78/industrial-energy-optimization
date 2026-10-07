"""Joint whole-day residual bootstrap for PV and day-ahead prices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapScenarioBatch:
    """Paired PV/price trajectories sampled from the same historical days."""

    pv_kwh: np.ndarray
    day_ahead_price_eur_per_mwh: np.ndarray
    source_residual_days: tuple[date, ...]
    hours_local: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.pv_kwh.shape != self.day_ahead_price_eur_per_mwh.shape:
            raise ValueError("PV and price scenario arrays must have the same shape")
        if self.pv_kwh.ndim != 2 or self.pv_kwh.shape[1] != len(self.hours_local):
            raise ValueError("Scenario arrays must be shaped [scenarios, working hours]")
        if self.pv_kwh.shape[0] != len(self.source_residual_days):
            raise ValueError("Every scenario must retain its sampled source day")
        if not np.isfinite(self.pv_kwh).all() or not np.isfinite(
            self.day_ahead_price_eur_per_mwh
        ).all():
            raise ValueError("Scenario arrays must contain only finite values")


class JointResidualBootstrap:
    """Sample complete paired daily forecast-error trajectories.

    The input library contains out-of-sample errors only. Sampling a complete
    source day preserves dependence across working hours and between PV and
    price without fitting a separate high-dimensional dependence model.
    """

    REQUIRED_COLUMNS = {
        "delivery_date_local",
        "hour_local",
        "pv_residual_kwh",
        "price_residual_eur_per_mwh",
    }

    def __init__(
        self,
        residual_library: pd.DataFrame,
        *,
        hours_local: tuple[int, ...] = tuple(range(6, 22)),
        center_residuals: bool = False,
    ) -> None:
        missing = sorted(self.REQUIRED_COLUMNS.difference(residual_library.columns))
        if missing:
            raise KeyError(f"Residual library is missing columns: {missing}")
        if not hours_local or len(set(hours_local)) != len(hours_local):
            raise ValueError("hours_local must contain unique working hours")
        frame = residual_library.loc[:, sorted(self.REQUIRED_COLUMNS)].copy()
        frame["delivery_date_local"] = pd.to_datetime(
            frame["delivery_date_local"]
        ).dt.date
        frame = frame.loc[frame["hour_local"].isin(hours_local)].copy()
        if frame.duplicated(["delivery_date_local", "hour_local"]).any():
            raise ValueError("Residual library has duplicate day/hour rows")
        counts = frame.groupby("delivery_date_local")["hour_local"].nunique()
        complete_days = counts.loc[counts == len(hours_local)].index
        frame = frame.loc[frame["delivery_date_local"].isin(complete_days)].copy()
        if frame.empty:
            raise ValueError("Residual library has no complete working-window days")
        actual_hours = set(frame["hour_local"].unique())
        if actual_hours != set(hours_local):
            raise ValueError("Residual library does not cover every configured working hour")
        if frame[["pv_residual_kwh", "price_residual_eur_per_mwh"]].isna().any().any():
            raise ValueError("Residual library contains missing residual values")

        self.hours_local = tuple(hours_local)
        self.center_residuals = bool(center_residuals)
        self._pv = self._pivot(frame, "pv_residual_kwh")
        self._price = self._pivot(frame, "price_residual_eur_per_mwh")
        common_days = self._pv.index.intersection(self._price.index).sort_values()
        self._pv = self._pv.loc[common_days]
        self._price = self._price.loc[common_days]

    @property
    def library_days(self) -> tuple[date, ...]:
        return tuple(self._pv.index)

    @property
    def residual_blocks(self) -> pd.DataFrame:
        """Return one row per day with paired hour-labelled residual columns."""

        pv = self._pv.copy()
        pv.columns = [f"pv_h{hour:02d}" for hour in self.hours_local]
        price = self._price.copy()
        price.columns = [f"price_h{hour:02d}" for hour in self.hours_local]
        return pd.concat([pv, price], axis=1)

    def sample(
        self,
        *,
        point_pv_kwh: np.ndarray,
        point_price_eur_per_mwh: np.ndarray,
        n_scenarios: int,
        random_state: int,
        as_of_date: date | None = None,
        pv_capacity_kwh_per_hour: float | None = None,
    ) -> BootstrapScenarioBatch:
        """Add sampled paired errors to one point-forecast trajectory."""

        pv_point = self._validate_point(point_pv_kwh, "point_pv_kwh")
        price_point = self._validate_point(
            point_price_eur_per_mwh, "point_price_eur_per_mwh"
        )
        if n_scenarios < 1:
            raise ValueError("n_scenarios must be positive")
        if pv_capacity_kwh_per_hour is not None and pv_capacity_kwh_per_hour <= 0:
            raise ValueError("pv_capacity_kwh_per_hour must be positive")

        eligible = np.ones(len(self._pv), dtype=bool)
        if as_of_date is not None:
            eligible = np.asarray([day < as_of_date for day in self._pv.index])
        eligible_positions = np.flatnonzero(eligible)
        if not len(eligible_positions):
            raise ValueError("No residual day is strictly earlier than as_of_date")
        rng = np.random.default_rng(random_state)
        positions = rng.choice(eligible_positions, size=n_scenarios, replace=True)
        pv_residual = self._pv.to_numpy(dtype=float)[positions].copy()
        price_residual = self._price.to_numpy(dtype=float)[positions].copy()
        if self.center_residuals:
            pv_residual -= self._pv.to_numpy(dtype=float)[eligible].mean(axis=0)
            price_residual -= self._price.to_numpy(dtype=float)[eligible].mean(axis=0)

        pv_scenarios = np.maximum(pv_point[None, :] + pv_residual, 0.0)
        if pv_capacity_kwh_per_hour is not None:
            pv_scenarios = np.minimum(pv_scenarios, pv_capacity_kwh_per_hour)
        price_scenarios = price_point[None, :] + price_residual
        source_days = tuple(self._pv.index[position] for position in positions)
        return BootstrapScenarioBatch(
            pv_kwh=pv_scenarios,
            day_ahead_price_eur_per_mwh=price_scenarios,
            source_residual_days=source_days,
            hours_local=self.hours_local,
        )

    def _pivot(self, frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
        return (
            frame.pivot(
                index="delivery_date_local", columns="hour_local", values=value_column
            )
            .loc[:, list(self.hours_local)]
            .sort_index()
        )

    def _validate_point(self, values: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape != (len(self.hours_local),) or not np.isfinite(result).all():
            raise ValueError(
                f"{name} must contain one finite value per configured working hour"
            )
        return result

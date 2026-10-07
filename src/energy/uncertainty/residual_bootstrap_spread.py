"""Whole-day joint residual bootstrap for PV, day-ahead price and spread."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThreeTargetBootstrapScenarioBatch:
    pv_kwh: np.ndarray
    day_ahead_price_eur_per_mwh: np.ndarray
    intraday_spread_eur_per_mwh: np.ndarray
    source_residual_days: tuple[date, ...]
    hours_local: tuple[int, ...]
    sampling_scheme: str = "uniform"
    sampling_effective_days: float | None = None

    def __post_init__(self) -> None:
        if not (
            self.pv_kwh.shape
            == self.day_ahead_price_eur_per_mwh.shape
            == self.intraday_spread_eur_per_mwh.shape
        ):
            raise ValueError("All scenario arrays must have the same shape")
        if self.pv_kwh.ndim != 2 or self.pv_kwh.shape[1] != len(self.hours_local):
            raise ValueError("Scenario arrays must be shaped [scenarios, working hours]")
        if self.pv_kwh.shape[0] != len(self.source_residual_days):
            raise ValueError("Every scenario must retain its sampled source day")
        if not all(
            np.isfinite(values).all()
            for values in (
                self.pv_kwh,
                self.day_ahead_price_eur_per_mwh,
                self.intraday_spread_eur_per_mwh,
            )
        ):
            raise ValueError("Scenario arrays must contain finite values")


class JointThreeTargetResidualBootstrap:
    """Sample one complete 48-dimensional historical residual trajectory."""

    REQUIRED_COLUMNS = {
        "delivery_date_local",
        "hour_local",
        "pv_residual_kwh",
        "price_residual_eur_per_mwh",
        "spread_residual_eur_per_mwh",
    }

    def __init__(
        self,
        residual_library: pd.DataFrame,
        *,
        hours_local: tuple[int, ...] = tuple(range(6, 22)),
        center_residuals: bool = True,
        sampling_scheme: str = "uniform",
        seasonal_bandwidth_days: float = 25.0,
        global_mixture_weight: float = 0.15,
        standardize_pv_residuals: bool = False,
    ) -> None:
        missing = sorted(self.REQUIRED_COLUMNS.difference(residual_library.columns))
        if missing:
            raise KeyError(f"Residual library is missing columns: {missing}")
        if not hours_local or len(set(hours_local)) != len(hours_local):
            raise ValueError("hours_local must contain unique working hours")
        if sampling_scheme not in {"uniform", "seasonal"}:
            raise ValueError("sampling_scheme must be 'uniform' or 'seasonal'")
        if seasonal_bandwidth_days <= 0 or not np.isfinite(
            seasonal_bandwidth_days
        ):
            raise ValueError("seasonal_bandwidth_days must be finite and positive")
        if not 0.0 <= global_mixture_weight <= 1.0:
            raise ValueError("global_mixture_weight must lie in [0, 1]")
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
        residual_columns = [
            "pv_residual_kwh",
            "price_residual_eur_per_mwh",
            "spread_residual_eur_per_mwh",
        ]
        if frame.empty or frame[residual_columns].isna().any().any():
            raise ValueError("Residual library has no complete finite residual days")
        if set(frame["hour_local"].unique()) != set(hours_local):
            raise ValueError("Residual library does not cover every configured hour")

        self.hours_local = tuple(hours_local)
        self.center_residuals = bool(center_residuals)
        self.sampling_scheme = sampling_scheme
        self.seasonal_bandwidth_days = float(seasonal_bandwidth_days)
        self.global_mixture_weight = float(global_mixture_weight)
        self.standardize_pv_residuals = bool(standardize_pv_residuals)
        self._pv = self._pivot(frame, "pv_residual_kwh")
        self._price = self._pivot(frame, "price_residual_eur_per_mwh")
        self._spread = self._pivot(frame, "spread_residual_eur_per_mwh")
        common = (
            self._pv.index.intersection(self._price.index)
            .intersection(self._spread.index)
            .sort_values()
        )
        self._pv = self._pv.loc[common]
        self._price = self._price.loc[common]
        self._spread = self._spread.loc[common]

    @property
    def library_days(self) -> tuple[date, ...]:
        return tuple(self._pv.index)

    @property
    def residual_blocks(self) -> pd.DataFrame:
        frames = []
        for prefix, values in (
            ("pv", self._pv),
            ("price", self._price),
            ("spread", self._spread),
        ):
            value = values.copy()
            value.columns = [f"{prefix}_h{hour:02d}" for hour in self.hours_local]
            frames.append(value)
        return pd.concat(frames, axis=1)

    def sample(
        self,
        *,
        point_pv_kwh: np.ndarray,
        point_price_eur_per_mwh: np.ndarray,
        point_spread_eur_per_mwh: np.ndarray,
        n_scenarios: int,
        random_state: int,
        as_of_date: date | None = None,
        pv_capacity_kwh_per_hour: float | None = None,
    ) -> ThreeTargetBootstrapScenarioBatch:
        pv_point = self._validate_point(point_pv_kwh, "point_pv_kwh")
        price_point = self._validate_point(
            point_price_eur_per_mwh, "point_price_eur_per_mwh"
        )
        spread_point = self._validate_point(
            point_spread_eur_per_mwh, "point_spread_eur_per_mwh"
        )
        if n_scenarios < 1:
            raise ValueError("n_scenarios must be positive")
        if pv_capacity_kwh_per_hour is not None and pv_capacity_kwh_per_hour <= 0:
            raise ValueError("pv_capacity_kwh_per_hour must be positive")

        eligible = np.ones(len(self._pv), dtype=bool)
        if as_of_date is not None:
            eligible = np.asarray([day < as_of_date for day in self._pv.index])
        positions_available = np.flatnonzero(eligible)
        if not len(positions_available):
            raise ValueError("No residual day is strictly earlier than as_of_date")
        if self.sampling_scheme == "seasonal" and as_of_date is None:
            raise ValueError("seasonal sampling requires as_of_date")
        eligible_days = tuple(self._pv.index[positions_available])
        probabilities = self._sampling_probabilities(
            eligible_days, target_day=as_of_date
        )
        rng = np.random.default_rng(random_state)
        local_positions = rng.choice(
            len(positions_available),
            size=n_scenarios,
            replace=True,
            p=probabilities,
        )
        positions = positions_available[local_positions]

        residuals = []
        for source_index, source in enumerate((self._pv, self._price, self._spread)):
            values = source.to_numpy(dtype=float)
            eligible_values = values[positions_available]
            if source_index == 0 and self.standardize_pv_residuals:
                source_scales = self._source_pv_scales(
                    eligible_values, eligible_days
                )
                standardized = eligible_values / source_scales
                sampled = standardized[local_positions].copy()
                if self.center_residuals:
                    sampled -= np.sum(
                        probabilities[:, None] * standardized, axis=0
                    )
                target_scale = self._weighted_scale(
                    eligible_values, probabilities
                )
                sampled *= target_scale
            else:
                sampled = eligible_values[local_positions].copy()
                if self.center_residuals:
                    sampled -= np.sum(
                        probabilities[:, None] * eligible_values, axis=0
                    )
            residuals.append(sampled)
        pv_residual, price_residual, spread_residual = residuals
        pv_scenarios = np.maximum(pv_point[None, :] + pv_residual, 0.0)
        if pv_capacity_kwh_per_hour is not None:
            pv_scenarios = np.minimum(pv_scenarios, pv_capacity_kwh_per_hour)
        return ThreeTargetBootstrapScenarioBatch(
            pv_kwh=pv_scenarios,
            day_ahead_price_eur_per_mwh=(
                price_point[None, :] + price_residual
            ),
            intraday_spread_eur_per_mwh=(
                spread_point[None, :] + spread_residual
            ),
            source_residual_days=tuple(
                self._pv.index[position] for position in positions
            ),
            hours_local=self.hours_local,
            sampling_scheme=self.sampling_scheme,
            sampling_effective_days=float(1.0 / np.sum(probabilities**2)),
        )

    def sampling_diagnostics(self, *, as_of_date: date) -> dict[str, float]:
        eligible_days = tuple(day for day in self._pv.index if day < as_of_date)
        if not eligible_days:
            raise ValueError("No residual day is strictly earlier than as_of_date")
        probabilities = self._sampling_probabilities(
            eligible_days, target_day=as_of_date
        )
        return {
            "eligible_days": float(len(eligible_days)),
            "effective_sample_days": float(1.0 / np.sum(probabilities**2)),
            "maximum_day_probability": float(probabilities.max()),
            "minimum_day_probability": float(probabilities.min()),
        }

    def _sampling_probabilities(
        self,
        eligible_days: tuple[date, ...],
        *,
        target_day: date | None,
    ) -> np.ndarray:
        count = len(eligible_days)
        if self.sampling_scheme == "uniform":
            return np.full(count, 1.0 / count)
        if target_day is None:
            raise ValueError("seasonal sampling requires a target day")
        distances = np.asarray(
            [self._circular_calendar_distance(day, target_day) for day in eligible_days],
            dtype=float,
        )
        kernel = np.exp(
            -0.5 * (distances / self.seasonal_bandwidth_days) ** 2
        )
        kernel /= kernel.sum()
        uniform = np.full(count, 1.0 / count)
        probabilities = (
            (1.0 - self.global_mixture_weight) * kernel
            + self.global_mixture_weight * uniform
        )
        return probabilities / probabilities.sum()

    def _source_pv_scales(
        self,
        pv_values: np.ndarray,
        source_days: tuple[date, ...],
    ) -> np.ndarray:
        scales = np.empty_like(pv_values, dtype=float)
        for position, source_day in enumerate(source_days):
            weights = self._seasonal_probabilities_for_scale(
                source_days, source_day
            )
            scales[position] = self._weighted_scale(pv_values, weights)
        return scales

    def _seasonal_probabilities_for_scale(
        self,
        source_days: tuple[date, ...],
        target_day: date,
    ) -> np.ndarray:
        distances = np.asarray(
            [self._circular_calendar_distance(day, target_day) for day in source_days],
            dtype=float,
        )
        kernel = np.exp(
            -0.5 * (distances / self.seasonal_bandwidth_days) ** 2
        )
        kernel /= kernel.sum()
        uniform = np.full(len(source_days), 1.0 / len(source_days))
        weights = (
            (1.0 - self.global_mixture_weight) * kernel
            + self.global_mixture_weight * uniform
        )
        return weights / weights.sum()

    @staticmethod
    def _weighted_scale(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        mean = np.sum(weights[:, None] * values, axis=0)
        variance = np.sum(weights[:, None] * (values - mean) ** 2, axis=0)
        global_scale = np.std(values, axis=0, ddof=0)
        floor = np.maximum(0.10 * global_scale, 1e-6)
        return np.maximum(np.sqrt(np.maximum(variance, 0.0)), floor)

    @staticmethod
    def _circular_calendar_distance(left: date, right: date) -> int:
        # Project month/day onto one leap reference year so leap and non-leap
        # dates share the same circular calendar coordinate.
        left_ordinal = date(2000, left.month, left.day).timetuple().tm_yday
        right_ordinal = date(2000, right.month, right.day).timetuple().tm_yday
        direct = abs(left_ordinal - right_ordinal)
        return min(direct, 366 - direct)

    def _pivot(self, frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
        return (
            frame.pivot(
                index="delivery_date_local",
                columns="hour_local",
                values=value_column,
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

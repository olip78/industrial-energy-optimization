"""Conditional quantile marginals coupled by historical empirical ranks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantileCopulaScenarioBatch:
    """Joint PV/price paths and the historical copula rows that generated them."""

    pv_kwh: np.ndarray
    day_ahead_price_eur_per_mwh: np.ndarray
    source_copula_days: tuple[date, ...]
    hours_local: tuple[int, ...]
    pv_uniform_ranks: np.ndarray
    price_uniform_ranks: np.ndarray

    def __post_init__(self) -> None:
        if self.pv_kwh.shape != self.day_ahead_price_eur_per_mwh.shape:
            raise ValueError("PV and price scenario arrays must have the same shape")
        if self.pv_kwh.shape != self.pv_uniform_ranks.shape:
            raise ValueError("PV paths and ranks must have the same shape")
        if self.pv_kwh.shape != self.price_uniform_ranks.shape:
            raise ValueError("Price paths and ranks must have the same shape")
        if self.pv_kwh.ndim != 2 or self.pv_kwh.shape[1] != len(self.hours_local):
            raise ValueError("Scenario arrays must be shaped [scenarios, working hours]")
        if self.pv_kwh.shape[0] != len(self.source_copula_days):
            raise ValueError("Every scenario must retain its sampled source day")


class QuantileEmpiricalCopula:
    """Apply historical joint ranks to current conditional quantile curves."""

    def __init__(
        self,
        copula_library: pd.DataFrame,
        *,
        quantile_levels: tuple[float, ...],
        hours_local: tuple[int, ...] = tuple(range(6, 22)),
    ) -> None:
        levels = np.asarray(quantile_levels, dtype=float)
        if levels.ndim != 1 or len(levels) < 3:
            raise ValueError("quantile_levels must contain at least three values")
        if not np.all(np.diff(levels) > 0) or levels[0] <= 0 or levels[-1] >= 1:
            raise ValueError("quantile_levels must be strictly increasing inside (0, 1)")
        if not hours_local or len(hours_local) != len(set(hours_local)):
            raise ValueError("hours_local must contain unique values")
        required = {"delivery_date_local"}
        required.update(f"pv_u_h{hour:02d}" for hour in hours_local)
        required.update(f"price_u_h{hour:02d}" for hour in hours_local)
        missing = sorted(required.difference(copula_library.columns))
        if missing:
            raise KeyError(f"Copula library is missing columns: {missing}")
        frame = copula_library.loc[:, sorted(required)].copy()
        frame["delivery_date_local"] = pd.to_datetime(
            frame["delivery_date_local"]
        ).dt.date
        if frame["delivery_date_local"].duplicated().any():
            raise ValueError("Copula library must contain one row per delivery day")
        rank_columns = [column for column in frame if column != "delivery_date_local"]
        ranks = frame[rank_columns].to_numpy(dtype=float)
        if not np.isfinite(ranks).all() or (ranks <= 0).any() or (ranks >= 1).any():
            raise ValueError("Empirical copula ranks must be finite and inside (0, 1)")
        self.quantile_levels = tuple(float(value) for value in levels)
        self.hours_local = tuple(hours_local)
        self._frame = frame.sort_values("delivery_date_local").reset_index(drop=True)

    @property
    def library_days(self) -> tuple[date, ...]:
        return tuple(self._frame["delivery_date_local"])

    def sample(
        self,
        *,
        pv_quantiles_kwh: np.ndarray,
        price_quantiles_eur_per_mwh: np.ndarray,
        n_scenarios: int,
        random_state: int,
        as_of_date: date | None = None,
        pv_capacity_kwh_per_hour: float | None = None,
    ) -> QuantileCopulaScenarioBatch:
        """Sample joint ranks and invert the current conditional marginals."""

        pv_quantiles = self._quantile_matrix(pv_quantiles_kwh, "pv_quantiles_kwh")
        price_quantiles = self._quantile_matrix(
            price_quantiles_eur_per_mwh, "price_quantiles_eur_per_mwh"
        )
        if n_scenarios < 1:
            raise ValueError("n_scenarios must be positive")
        if pv_capacity_kwh_per_hour is not None and pv_capacity_kwh_per_hour <= 0:
            raise ValueError("pv_capacity_kwh_per_hour must be positive")

        eligible = np.ones(len(self._frame), dtype=bool)
        if as_of_date is not None:
            eligible = self._frame["delivery_date_local"].lt(as_of_date).to_numpy()
        positions = np.flatnonzero(eligible)
        if not len(positions):
            raise ValueError("No copula day is strictly earlier than as_of_date")
        rng = np.random.default_rng(random_state)
        sampled_positions = rng.choice(positions, size=n_scenarios, replace=True)
        sampled = self._frame.iloc[sampled_positions]
        pv_u = sampled[
            [f"pv_u_h{hour:02d}" for hour in self.hours_local]
        ].to_numpy(dtype=float)
        price_u = sampled[
            [f"price_u_h{hour:02d}" for hour in self.hours_local]
        ].to_numpy(dtype=float)
        pv_scenarios = self._inverse_marginals(pv_u, pv_quantiles)
        pv_scenarios = np.maximum(pv_scenarios, 0.0)
        if pv_capacity_kwh_per_hour is not None:
            pv_scenarios = np.minimum(pv_scenarios, pv_capacity_kwh_per_hour)
        price_scenarios = self._inverse_marginals(price_u, price_quantiles)
        return QuantileCopulaScenarioBatch(
            pv_kwh=pv_scenarios,
            day_ahead_price_eur_per_mwh=price_scenarios,
            source_copula_days=tuple(sampled["delivery_date_local"]),
            hours_local=self.hours_local,
            pv_uniform_ranks=pv_u,
            price_uniform_ranks=price_u,
        )

    def _quantile_matrix(self, values: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        expected = (len(self.hours_local), len(self.quantile_levels))
        if result.shape != expected or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite matrix shaped {expected}")
        # Monotone rearrangement fixes any finite-sample quantile crossings.
        return np.sort(result, axis=1)

    def _inverse_marginals(
        self, uniform_ranks: np.ndarray, quantile_values: np.ndarray
    ) -> np.ndarray:
        levels = np.asarray(self.quantile_levels, dtype=float)
        result = np.empty_like(uniform_ranks, dtype=float)
        # Clipping avoids unsupported extrapolation beyond the trained P05/P95
        # marginals. Boundary mass is explicit and can be audited in calibration.
        clipped = np.clip(uniform_ranks, levels[0], levels[-1])
        for hour_index in range(len(self.hours_local)):
            result[:, hour_index] = np.interp(
                clipped[:, hour_index], levels, quantile_values[hour_index]
            )
        return result

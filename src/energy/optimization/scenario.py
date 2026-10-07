"""Fixed physical setting for the first economic-replay scenario.

The MPVBench profile supplies a measured *shape*, not verified equipment
metadata.  This module therefore keeps every scaling and asset assumption in
one auditable, version-controlled object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from energy.optimization.deterministic import BatteryParameters


@dataclass(frozen=True)
class ReferenceScenario:
    """Physical and economic assumptions shared by all V1 strategies.

    The battery capacity represents two 22.08 kWh usable reference units.
    The 5 kW hourly delivery cap is an explicit AC connection constraint,
    rather than the DC battery's nameplate power.
    """

    daily_load_kwh: float = 200.0
    working_start_hour: int = 6
    working_end_hour: int = 22
    load_min_kwh_per_hour: float = 5.0
    load_max_kwh_per_hour: float = 20.0
    pv_profile_scale: float = 20.0
    battery_available_energy_kwh: float = 44.16
    battery_max_discharge_kwh_per_hour: float = 5.0
    charge_efficiency: float = 0.98
    discharge_efficiency: float = 0.98
    degradation_eur_per_kwh: float = 0.05
    # Nine equally weighted hourly price labels used by the conservative
    # night-charge accounting proxy: 22:00, 23:00 and 00:00--06:00.
    # At 5 kWh per label this represents a 45 kWh charging purchase.
    night_hours: tuple[int, ...] = (22, 23, 0, 1, 2, 3, 4, 5, 6)

    def __post_init__(self) -> None:
        if not 0 <= self.working_start_hour < self.working_end_hour <= 24:
            raise ValueError("working hours must form a non-empty subset of 0..23")
        if self.daily_load_kwh <= 0:
            raise ValueError("daily_load_kwh must be positive")
        if not 0 <= self.load_min_kwh_per_hour <= self.load_max_kwh_per_hour:
            raise ValueError("load bounds must satisfy 0 <= min <= max")
        if self.pv_profile_scale <= 0:
            raise ValueError("pv_profile_scale must be positive")
        if self.battery_available_energy_kwh < 0 or self.battery_max_discharge_kwh_per_hour < 0:
            raise ValueError("battery limits must be non-negative")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("battery efficiencies must be in (0, 1]")
        if self.degradation_eur_per_kwh < 0:
            raise ValueError("degradation_eur_per_kwh must be non-negative")
        if any(hour not in range(24) for hour in self.night_hours):
            raise ValueError("night_hours must only contain local clock hours")
        if len(self.night_hours) != len(set(self.night_hours)):
            raise ValueError("night_hours must not contain duplicates")
        minimum, maximum = self.daily_load_bounds()
        if not minimum <= self.daily_load_kwh <= maximum:
            raise ValueError(
                "daily_load_kwh is infeasible under the configured working window and load bounds"
            )

    @property
    def working_hours(self) -> tuple[int, ...]:
        """Local delivery hours in which production can consume energy."""

        return tuple(range(self.working_start_hour, self.working_end_hour))

    def daily_load_bounds(self) -> tuple[float, float]:
        """Return the minimum and maximum feasible daily production energy."""

        count = len(self.working_hours)
        return (
            count * self.load_min_kwh_per_hour,
            count * self.load_max_kwh_per_hour,
        )

    def load_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Build 24 hourly load lower and upper energy bounds in kWh."""

        lower = np.zeros(24, dtype=float)
        upper = np.zeros(24, dtype=float)
        lower[list(self.working_hours)] = self.load_min_kwh_per_hour
        upper[list(self.working_hours)] = self.load_max_kwh_per_hour
        return lower, upper

    def discharge_allowed(self) -> np.ndarray:
        """Allow daytime discharge only while production is in its working window."""

        allowed = np.zeros(24, dtype=bool)
        allowed[list(self.working_hours)] = True
        return allowed

    def scale_pv_energy(self, raw_profile_energy_kwh: Iterable[float]) -> np.ndarray:
        """Scale the source PV shape from its roughly 0.5 kW reference to 10 kWp."""

        raw = np.asarray(list(raw_profile_energy_kwh), dtype=float)
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise ValueError("raw_profile_energy_kwh must be a finite one-dimensional vector")
        if (raw < 0).any():
            raise ValueError("raw_profile_energy_kwh must be non-negative")
        return raw * self.pv_profile_scale

    def battery(self, night_reference_price_eur_per_mwh: float) -> BatteryParameters:
        """Construct the daily battery cost contract from its known night reference."""

        if not np.isfinite(night_reference_price_eur_per_mwh):
            raise ValueError("night_reference_price_eur_per_mwh must be finite")
        return BatteryParameters(
            available_energy_kwh=self.battery_available_energy_kwh,
            max_discharge_kwh_per_hour=self.battery_max_discharge_kwh_per_hour,
            night_reference_price_eur_per_mwh=float(night_reference_price_eur_per_mwh),
            charge_efficiency=self.charge_efficiency,
            discharge_efficiency=self.discharge_efficiency,
            degradation_eur_per_kwh=self.degradation_eur_per_kwh,
        )


V1_REFERENCE_SCENARIO = ReferenceScenario()

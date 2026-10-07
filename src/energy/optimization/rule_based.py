"""Historical-clock rule policy used as the economic replay baseline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from energy.optimization.deterministic import battery_full_cost_eur_per_kwh
from energy.optimization.scenario import ReferenceScenario


@dataclass(frozen=True)
class RuleBasedPriceShape:
    """Frozen seasonal clock-price reference estimated before the test window."""

    scores_eur_per_mwh: Mapping[tuple[str, str, int], float]
    fallback_scores_eur_per_mwh: Mapping[tuple[str, int], float]

    def score_for_day(self, delivery_day: date, hour_local: int) -> float:
        """Return the historic score with a day-type fallback for sparse groups."""

        if hour_local not in range(24):
            raise ValueError("hour_local must be in 0..23")
        day_type = _day_type(delivery_day)
        seasonal_key = (_season(delivery_day), day_type, hour_local)
        if seasonal_key in self.scores_eur_per_mwh:
            return float(self.scores_eur_per_mwh[seasonal_key])
        fallback_key = (day_type, hour_local)
        if fallback_key in self.fallback_scores_eur_per_mwh:
            return float(self.fallback_scores_eur_per_mwh[fallback_key])
        raise ValueError(f"No frozen rule-based price score for {seasonal_key}")


@dataclass(frozen=True)
class RuleBasedDayPlan:
    """One non-adaptive daily plan and its fixed day-ahead position."""

    load_kwh: np.ndarray
    discharge_kwh: np.ndarray
    day_ahead_position_kwh: np.ndarray
    price_score_eur_per_mwh: np.ndarray
    price_rank: np.ndarray


def fit_rule_based_price_shape(
    history: pd.DataFrame,
    *,
    price_column: str = "day_ahead_price_eur_per_mwh",
    date_column: str = "delivery_date_local",
    hour_column: str = "hour_local",
) -> RuleBasedPriceShape:
    """Fit frozen median price scores from data strictly before the test period.

    The input is intentionally limited to past realised day-ahead prices.  It
    creates a simple seasonal time-of-use rule, not a forecast for an individual
    delivery day.
    """

    required = {price_column, date_column, hour_column}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"History is missing columns: {sorted(missing)}")
    frame = history.loc[:, [price_column, date_column, hour_column]].copy()
    frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.date
    frame[hour_column] = pd.to_numeric(frame[hour_column], errors="coerce")
    valid = (
        frame[price_column].notna()
        & frame[date_column].notna()
        & frame[hour_column].between(0, 23)
        & (frame[hour_column] % 1 == 0)
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("No valid historical price rows are available")
    frame[hour_column] = frame[hour_column].astype(int)
    frame["_day_type"] = frame[date_column].map(_day_type)
    frame["_season"] = frame[date_column].map(_season)

    seasonal = frame.groupby(["_season", "_day_type", hour_column], sort=False)[price_column].median()
    fallback = frame.groupby(["_day_type", hour_column], sort=False)[price_column].median()
    return RuleBasedPriceShape(
        scores_eur_per_mwh={
            (str(season), str(day_type), int(hour)): float(score)
            for (season, day_type, hour), score in seasonal.items()
        },
        fallback_scores_eur_per_mwh={
            (str(day_type), int(hour)): float(score)
            for (day_type, hour), score in fallback.items()
        },
    )


def build_rule_based_plan(
    *,
    delivery_day: date,
    price_shape: RuleBasedPriceShape,
    scenario: ReferenceScenario,
    night_reference_price_eur_per_mwh: float,
    battery_discharge_cost_eur_per_kwh: float | None = None,
    avoided_import_charge_eur_per_kwh: float = 0.0,
) -> RuleBasedDayPlan:
    """Create the agreed rule policy without a daily price or PV forecast.

    Load is a capped inverse-*rank* allocation of a fixed 200 kWh daily need.
    Ranking avoids undefined reciprocal values at zero or negative electricity
    prices.  The day-ahead position buys only the known production load; factual
    PV and the pre-scheduled discharge subsequently appear as intraday sales.
    """

    active = np.asarray(scenario.working_hours, dtype=int)
    scores_active = np.asarray(
        [price_shape.score_for_day(delivery_day, int(hour)) for hour in active],
        dtype=float,
    )
    if not np.isfinite(scores_active).all():
        raise ValueError("Rule-based price scores must be finite")

    # Stable ties make the policy reproducible: earlier local hour wins a tie.
    order_ascending = np.lexsort((active, scores_active))
    rank_active = np.empty(len(active), dtype=int)
    rank_active[order_ascending] = np.arange(1, len(active) + 1)
    inverse_rank_weight = len(active) + 1 - rank_active

    load_active = _capped_weighted_allocation(
        weights=inverse_rank_weight.astype(float),
        target_kwh=scenario.daily_load_kwh,
        lower_kwh=scenario.load_min_kwh_per_hour,
        upper_kwh=scenario.load_max_kwh_per_hour,
    )

    load = np.zeros(24, dtype=float)
    discharge = np.zeros(24, dtype=float)
    price_score = np.full(24, np.nan, dtype=float)
    rank = np.zeros(24, dtype=int)
    load[active] = load_active
    price_score[active] = scores_active
    rank[active] = rank_active

    battery = scenario.battery(night_reference_price_eur_per_mwh)
    discharge_cost = (
        battery_full_cost_eur_per_kwh(battery)
        if battery_discharge_cost_eur_per_kwh is None
        else float(battery_discharge_cost_eur_per_kwh)
    )
    if discharge_cost < 0 or avoided_import_charge_eur_per_kwh < 0:
        raise ValueError("battery and avoided-import costs must be non-negative")
    # A discharge that offsets physical import earns both the wholesale value
    # and the avoided variable grid charge.  The simple rule does not know PV,
    # so this is an explicit optimistic threshold rather than an hourly fact.
    minimum_profitable_score = (
        discharge_cost - avoided_import_charge_eur_per_kwh
    ) * 1_000.0
    energy_left = battery.available_energy_kwh
    for position in np.lexsort((active, -scores_active)):
        if scores_active[position] <= minimum_profitable_score:
            break
        output = min(battery.max_discharge_kwh_per_hour, energy_left)
        discharge[active[position]] = output
        energy_left -= output
        if energy_left <= 1e-9:
            break

    return RuleBasedDayPlan(
        load_kwh=load,
        discharge_kwh=discharge,
        day_ahead_position_kwh=load.copy(),
        price_score_eur_per_mwh=price_score,
        price_rank=rank,
    )


def _capped_weighted_allocation(
    *,
    weights: np.ndarray,
    target_kwh: float,
    lower_kwh: float,
    upper_kwh: float,
) -> np.ndarray:
    """Solve ``sum(clip(a * weights, lower, upper)) = target`` by bisection."""

    if weights.ndim != 1 or not len(weights) or (weights <= 0).any():
        raise ValueError("weights must be a non-empty positive one-dimensional vector")
    minimum = len(weights) * lower_kwh
    maximum = len(weights) * upper_kwh
    if not minimum <= target_kwh <= maximum:
        raise ValueError("target_kwh is infeasible under the supplied bounds")
    if np.isclose(target_kwh, minimum):
        return np.full(len(weights), lower_kwh, dtype=float)
    if np.isclose(target_kwh, maximum):
        return np.full(len(weights), upper_kwh, dtype=float)

    low = 0.0
    high = upper_kwh / float(weights.min())
    for _ in range(80):
        midpoint = (low + high) / 2.0
        allocation = np.clip(midpoint * weights, lower_kwh, upper_kwh)
        if allocation.sum() < target_kwh:
            low = midpoint
        else:
            high = midpoint
    allocation = np.clip(high * weights, lower_kwh, upper_kwh)
    # The tolerance is well below one watt-hour for this 24-hour LP setting.
    if not np.isclose(allocation.sum(), target_kwh, atol=1e-8):
        raise AssertionError("Capped weighted allocation did not meet its target")
    return allocation


def _day_type(delivery_day: date) -> str:
    return "weekend" if delivery_day.weekday() >= 5 else "workday"


def _season(delivery_day: date) -> str:
    month = delivery_day.month
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"

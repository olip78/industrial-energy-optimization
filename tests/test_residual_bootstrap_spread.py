from __future__ import annotations

import numpy as np
import pandas as pd

from energy.uncertainty.residual_bootstrap_spread import (
    JointThreeTargetResidualBootstrap,
)


def _library() -> pd.DataFrame:
    rows = []
    for day, value in (("2024-01-01", -1.0), ("2024-01-02", 1.0)):
        for hour in (6, 7):
            rows.append(
                {
                    "delivery_date_local": day,
                    "hour_local": hour,
                    "pv_residual_kwh": value,
                    "price_residual_eur_per_mwh": 10.0 * value,
                    "spread_residual_eur_per_mwh": -5.0 * value,
                }
            )
    return pd.DataFrame(rows)


def test_whole_day_sample_preserves_cross_target_residual_path() -> None:
    generator = JointThreeTargetResidualBootstrap(
        _library(), hours_local=(6, 7), center_residuals=True
    )
    batch = generator.sample(
        point_pv_kwh=np.asarray([2.0, 2.0]),
        point_price_eur_per_mwh=np.asarray([50.0, 50.0]),
        point_spread_eur_per_mwh=np.asarray([0.0, 0.0]),
        n_scenarios=20,
        random_state=42,
        as_of_date=pd.Timestamp("2025-01-01").date(),
        pv_capacity_kwh_per_hour=10.0,
    )

    pv_error = batch.pv_kwh - 2.0
    price_error = batch.day_ahead_price_eur_per_mwh - 50.0
    spread_error = batch.intraday_spread_eur_per_mwh
    np.testing.assert_allclose(price_error, 10.0 * pv_error)
    np.testing.assert_allclose(spread_error, -5.0 * pv_error)
    np.testing.assert_allclose(pv_error[:, 0], pv_error[:, 1])
    assert set(batch.source_residual_days) == {
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-01-02").date(),
    }


def test_centering_uses_only_days_before_information_cutoff() -> None:
    generator = JointThreeTargetResidualBootstrap(
        _library(), hours_local=(6, 7), center_residuals=True
    )
    batch = generator.sample(
        point_pv_kwh=np.asarray([2.0, 2.0]),
        point_price_eur_per_mwh=np.asarray([50.0, 50.0]),
        point_spread_eur_per_mwh=np.asarray([0.0, 0.0]),
        n_scenarios=5,
        random_state=42,
        as_of_date=pd.Timestamp("2024-01-02").date(),
    )

    # Only 1 January is eligible, so its residual becomes zero after centering.
    np.testing.assert_allclose(batch.pv_kwh, 2.0)
    np.testing.assert_allclose(batch.day_ahead_price_eur_per_mwh, 50.0)
    np.testing.assert_allclose(batch.intraday_spread_eur_per_mwh, 0.0)


def _seasonal_library() -> pd.DataFrame:
    rows = []
    for day, residual in (
        ("2024-01-01", -1.0),
        ("2024-01-03", 1.0),
        ("2024-07-01", -10.0),
        ("2024-07-03", 10.0),
        ("2024-12-31", 0.5),
    ):
        for hour in (6, 7):
            rows.append(
                {
                    "delivery_date_local": day,
                    "hour_local": hour,
                    "pv_residual_kwh": residual,
                    "price_residual_eur_per_mwh": 10.0 * np.sign(residual),
                    "spread_residual_eur_per_mwh": -5.0 * np.sign(residual),
                }
            )
    return pd.DataFrame(rows)


def test_seasonal_sampling_wraps_around_new_year() -> None:
    generator = JointThreeTargetResidualBootstrap(
        _seasonal_library(),
        hours_local=(6, 7),
        center_residuals=False,
        sampling_scheme="seasonal",
        seasonal_bandwidth_days=7.0,
        global_mixture_weight=0.0,
    )
    batch = generator.sample(
        point_pv_kwh=np.asarray([20.0, 20.0]),
        point_price_eur_per_mwh=np.asarray([50.0, 50.0]),
        point_spread_eur_per_mwh=np.asarray([0.0, 0.0]),
        n_scenarios=500,
        random_state=42,
        as_of_date=pd.Timestamp("2025-01-02").date(),
    )

    source_months = pd.Series(batch.source_residual_days).map(lambda value: value.month)
    assert (source_months.isin([1, 12])).mean() > 0.99
    assert batch.sampling_scheme == "seasonal"
    assert batch.sampling_effective_days is not None
    assert batch.sampling_effective_days < len(generator.library_days)


def test_standardized_seasonal_pv_residuals_use_target_season_scale() -> None:
    generator = JointThreeTargetResidualBootstrap(
        _seasonal_library(),
        hours_local=(6, 7),
        center_residuals=True,
        sampling_scheme="seasonal",
        seasonal_bandwidth_days=7.0,
        global_mixture_weight=0.0,
        standardize_pv_residuals=True,
    )
    batch = generator.sample(
        point_pv_kwh=np.asarray([20.0, 20.0]),
        point_price_eur_per_mwh=np.asarray([50.0, 50.0]),
        point_spread_eur_per_mwh=np.asarray([0.0, 0.0]),
        n_scenarios=2_000,
        random_state=42,
        as_of_date=pd.Timestamp("2025-01-02").date(),
    )

    pv_error = batch.pv_kwh - 20.0
    assert float(pv_error.std()) < 2.0
    assert float(pv_error.std()) > 0.05


def test_global_mixture_retains_nonlocal_tail_days() -> None:
    generator = JointThreeTargetResidualBootstrap(
        _seasonal_library(),
        hours_local=(6, 7),
        sampling_scheme="seasonal",
        seasonal_bandwidth_days=7.0,
        global_mixture_weight=0.20,
    )
    batch = generator.sample(
        point_pv_kwh=np.asarray([20.0, 20.0]),
        point_price_eur_per_mwh=np.asarray([50.0, 50.0]),
        point_spread_eur_per_mwh=np.asarray([0.0, 0.0]),
        n_scenarios=5_000,
        random_state=42,
        as_of_date=pd.Timestamp("2025-01-02").date(),
    )

    source_months = pd.Series(batch.source_residual_days).map(lambda value: value.month)
    assert (source_months == 7).any()

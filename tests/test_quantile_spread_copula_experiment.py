from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy.training.quantile_copula_experiment import DEFAULT_QUANTILES
from energy.training.quantile_spread_copula_experiment import (
    HOURS_LOCAL,
    sample_three_target_scenarios,
)


def _forecast_frame() -> pd.DataFrame:
    frame = pd.DataFrame({"hour_local": HOURS_LOCAL})
    for prefix, offset in (("pv", 1.0), ("price", 50.0), ("spread", -10.0)):
        for level in DEFAULT_QUANTILES:
            frame[f"{prefix}_q{int(round(level * 100)):02d}"] = (
                offset + 10.0 * level
            )
    return frame


def _copula_library() -> pd.DataFrame:
    rows = []
    for day, rank in (("2024-01-01", 0.2), ("2024-01-02", 0.8)):
        row: dict[str, object] = {"delivery_date_local": day}
        for prefix in ("pv", "price", "spread"):
            for hour in HOURS_LOCAL:
                row[f"{prefix}_u_h{hour:02d}"] = rank
        rows.append(row)
    return pd.DataFrame(rows)


def test_sampler_returns_joint_complete_paths_and_respects_cutoff() -> None:
    batch = sample_three_target_scenarios(
        day_forecasts=_forecast_frame(),
        copula_library=_copula_library(),
        quantile_levels=DEFAULT_QUANTILES,
        n_scenarios=20,
        random_state=42,
        as_of_date=pd.Timestamp("2025-01-01").date(),
        pv_capacity_kwh_per_hour=10.0,
    )

    assert batch.pv_kwh.shape == (20, 16)
    assert batch.day_ahead_price_eur_per_mwh.shape == (20, 16)
    assert batch.intraday_spread_eur_per_mwh.shape == (20, 16)
    assert np.all((batch.pv_kwh >= 0.0) & (batch.pv_kwh <= 10.0))
    assert set(batch.source_copula_days) <= {
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-01-02").date(),
    }
    np.testing.assert_allclose(
        batch.uniform_ranks["pv"], batch.uniform_ranks["price"]
    )
    np.testing.assert_allclose(
        batch.uniform_ranks["price"], batch.uniform_ranks["spread"]
    )


def test_sampler_rejects_incomplete_operating_window() -> None:
    with pytest.raises(ValueError, match="complete operating window"):
        sample_three_target_scenarios(
            day_forecasts=_forecast_frame().iloc[:-1],
            copula_library=_copula_library(),
            quantile_levels=DEFAULT_QUANTILES,
            n_scenarios=10,
            random_state=42,
            as_of_date=pd.Timestamp("2025-01-01").date(),
            pv_capacity_kwh_per_hour=10.0,
        )

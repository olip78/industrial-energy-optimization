from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from energy.training.residual_bootstrap_spread_economic_backtest import (
    ResidualBootstrapSpreadEconomicBacktestConfig,
    _assert_frozen_points,
    _validate_config,
)


def test_point_forecast_guard_rejects_a_stale_price_artifact() -> None:
    hours = np.arange(6, 22)
    pv = np.arange(24, dtype=float)
    price = np.arange(24, dtype=float) * 10.0
    inputs = SimpleNamespace(
        day_ahead_pv_forecast_kwh=pv,
        day_ahead_price_forecast_eur_per_mwh=price,
    )

    _assert_frozen_points(inputs, hours, pv[hours], price[hours])
    stale_price = price[hours].copy()
    stale_price[3] += 0.01
    with pytest.raises(AssertionError, match="point price"):
        _assert_frozen_points(inputs, hours, pv[hours], stale_price)


def test_risk_generator_must_also_be_an_enabled_generator() -> None:
    config = ResidualBootstrapSpreadEconomicBacktestConfig(
        project_root=Path("."),
        generators=("raw",),
        risk_generators=("centered",),
    )
    with pytest.raises(ValueError, match="subset"):
        _validate_config(config)

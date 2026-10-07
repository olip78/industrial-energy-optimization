from pathlib import Path

import pytest

from energy.data.builder import WEATHER_VARIABLES
from energy.training.dynamic_charge_economic_backtest import (
    DynamicChargeEconomicBacktestConfig,
)
from energy.training.pv_mpc_residual import _residual_features


def test_updated_weather_replace_removes_stale_weather_levels() -> None:
    features = _residual_features(
        4, uses_updated_weather=True, updated_weather_feature_mode="replace"
    )
    for variable in WEATHER_VARIABLES:
        assert f"weather_forecast_{variable}" not in features
        assert f"weather_update_{variable}" in features
        assert f"weather_update_delta_{variable}" not in features
    assert "weather_update_clear_sky_index" in features
    assert len(features) == len(set(features))


def test_updated_weather_revision_keeps_anchor_and_only_adds_deltas() -> None:
    features = _residual_features(
        4, uses_updated_weather=True, updated_weather_feature_mode="revision"
    )
    for variable in WEATHER_VARIABLES:
        assert f"weather_forecast_{variable}" in features
        assert f"weather_update_delta_{variable}" in features
        assert f"weather_update_{variable}" not in features
    assert len(features) == len(set(features))


def test_invalid_updated_weather_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="updated_weather_feature_mode"):
        _residual_features(
            4, uses_updated_weather=True, updated_weather_feature_mode="unknown"
        )


def test_dynamic_backtest_accepts_an_explicit_pv_mpc_artifact() -> None:
    prediction_path = Path("/tmp/pv_mpc_predictions.parquet")
    config = DynamicChargeEconomicBacktestConfig(
        project_root=Path("/tmp/project"),
        pv_mpc_prediction_path=prediction_path,
    )
    assert config.pv_mpc_prediction_path == prediction_path

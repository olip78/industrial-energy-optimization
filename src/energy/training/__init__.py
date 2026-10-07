"""Trainable model applications for the energy project."""

from energy.training.pv_day_ahead import PVDayAheadTrainingConfig, train_day_ahead_pv
from energy.training.pv_mpc_residual import (
    PVMpcResidualTrainingConfig,
    train_pv_mpc_residual,
)

__all__ = [
    "PVDayAheadTrainingConfig",
    "PVMpcResidualTrainingConfig",
    "train_day_ahead_pv",
    "train_pv_mpc_residual",
]

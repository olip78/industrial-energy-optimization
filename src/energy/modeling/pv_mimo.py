"""Leakage-safe MIMO MLP utilities for intraday PV residual trajectories.

A sample represents one MPC decision hour.  It uses only factual PV values
strictly before that decision and returns a vector of residuals for the future
solar-active hours.  The horizon is fixed by zero-padding short end-of-day
trajectories; a mask excludes that padding from loss and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_TARGET_COLUMN = "pv_power_mean_w"
DEFAULT_DAY_AHEAD_COLUMN = "day_ahead_prediction_w"


@dataclass(frozen=True)
class MIMOSamples:
    """Fixed-horizon MIMO rows with offline labels and delivery metadata."""

    features: np.ndarray
    targets_residual_w: np.ndarray
    target_mask: np.ndarray
    target_pv_w: np.ndarray
    day_ahead_pv_w: np.ndarray
    lead_hours: np.ndarray
    week_index: np.ndarray
    delivery_date_local: np.ndarray
    decision_hour_local: np.ndarray
    feature_names: tuple[str, ...]
    site_id: np.ndarray | None = None

    @property
    def horizon(self) -> int:
        return int(self.targets_residual_w.shape[1])

    def select_weeks(self, weeks: Iterable[int]) -> "MIMOSamples":
        """Keep complete delivery weeks; no day can cross a split boundary."""

        mask = np.isin(self.week_index, list(weeks))
        if not mask.any():
            raise ValueError("The requested weeks contain no MIMO samples")
        return MIMOSamples(
            features=self.features[mask],
            targets_residual_w=self.targets_residual_w[mask],
            target_mask=self.target_mask[mask],
            target_pv_w=self.target_pv_w[mask],
            day_ahead_pv_w=self.day_ahead_pv_w[mask],
            lead_hours=self.lead_hours[mask],
            week_index=self.week_index[mask],
            delivery_date_local=self.delivery_date_local[mask],
            decision_hour_local=self.decision_hour_local[mask],
            feature_names=self.feature_names,
            site_id=self.site_id[mask] if self.site_id is not None else None,
        )


@dataclass(frozen=True)
class MIMOMLPTrainingConfig:
    """Small, regularised MLP settings appropriate for the one-site dataset."""

    hidden_width: int = 48
    hidden_layers: int = 2
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 300
    patience: int | None = 30
    random_state: int = 42


@dataclass(frozen=True)
class MIMOMLPFit:
    """Fitted neural regressor and the selection diagnostics."""

    model: Any
    input_scaler: StandardScaler
    target_mean_w: float
    target_scale_w: float
    best_epoch: int
    best_validation_mae_w: float | None

    def predict_residual_w(self, samples: MIMOSamples) -> np.ndarray:
        """Return one residual trajectory per MPC decision sample."""

        torch = _torch()
        x = self.input_scaler.transform(samples.features).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            scaled = self.model(torch.from_numpy(x)).cpu().numpy()
        return scaled * self.target_scale_w + self.target_mean_w


def build_mimo_residual_samples(
    frame: pd.DataFrame,
    *,
    future_features: Sequence[str],
    n_lags: int = 4,
    max_horizon: int | None = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    day_ahead_column: str = DEFAULT_DAY_AHEAD_COLUMN,
    site_column: str | None = None,
) -> MIMOSamples:
    """Build one fixed-horizon sample for every valid MPC decision hour.

    ``frame`` must contain solar-active rows only and a frozen day-ahead PV
    prediction.  Future covariates are flattened in lead-time order.  Actual
    PV and residual inputs are limited to completed active hours before the
    decision time.
    """

    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    required = {
        "delivery_date_local",
        "week_index",
        "hour_local",
        target_column,
        day_ahead_column,
        *future_features,
    }
    if site_column is not None:
        required.add(site_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Frame is missing MIMO columns: {missing}")

    ordered = frame.sort_values(["delivery_date_local", "hour_local"]).copy()
    if ordered[list(future_features)].isna().any().any():
        missing_features = ordered.loc[:, list(future_features)].columns[
            ordered.loc[:, list(future_features)].isna().any()
        ].tolist()
        raise ValueError(f"MIMO future features contain missing values: {missing_features}")
    if ordered[day_ahead_column].isna().any():
        raise ValueError("Frozen day-ahead prediction contains missing values")

    group_columns = ["delivery_date_local"]
    if site_column is not None:
        group_columns.insert(0, site_column)
    candidate_horizon = _maximum_remaining_active_horizon(ordered, group_columns)
    horizon = max_horizon if max_horizon is not None else candidate_horizon
    if horizon < 1:
        raise ValueError("At least one future solar-active hour is required")
    if max_horizon is not None and max_horizon < candidate_horizon:
        raise ValueError(
            f"max_horizon={max_horizon} truncates a {candidate_horizon}-hour trajectory"
        )

    feature_names = _feature_names(
        future_features,
        n_lags,
        horizon,
        day_ahead_column=day_ahead_column,
    )
    features: list[np.ndarray] = []
    residual_targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actual_targets: list[np.ndarray] = []
    day_ahead_targets: list[np.ndarray] = []
    lead_targets: list[np.ndarray] = []
    week_indices: list[int] = []
    delivery_dates: list[Any] = []
    decision_hours: list[int] = []
    site_ids: list[Any] = []

    for group_key, day in ordered.groupby(group_columns, sort=False):
        delivery_date = group_key[-1] if isinstance(group_key, tuple) else group_key
        day = day.sort_values("hour_local").copy()
        day["_residual_w"] = day[target_column] - day[day_ahead_column]

        for decision_hour in range(24):
            history = day.loc[day["hour_local"] < decision_hour].tail(n_lags)
            future = day.loc[day["hour_local"] > decision_hour]
            if history.empty or future.empty:
                continue
            if len(future) > horizon:
                raise AssertionError("MIMO future trajectory is longer than its fixed horizon")

            x = _make_feature_vector(
                history=history,
                future=future,
                decision_hour=decision_hour,
                future_features=future_features,
                n_lags=n_lags,
                horizon=horizon,
                target_column=target_column,
                day_ahead_column=day_ahead_column,
            )
            residual, mask, actual, day_ahead, lead = _make_targets(
                future,
                decision_hour=decision_hour,
                horizon=horizon,
                target_column=target_column,
                day_ahead_column=day_ahead_column,
            )
            features.append(x)
            residual_targets.append(residual)
            masks.append(mask)
            actual_targets.append(actual)
            day_ahead_targets.append(day_ahead)
            lead_targets.append(lead)
            week_indices.append(int(day["week_index"].iloc[0]))
            delivery_dates.append(delivery_date)
            decision_hours.append(decision_hour)
            if site_column is not None:
                site_ids.append(day[site_column].iloc[0])

    if not features:
        raise ValueError("No valid MIMO MPC samples were created")
    return MIMOSamples(
        features=np.vstack(features).astype(np.float32),
        targets_residual_w=np.vstack(residual_targets).astype(np.float32),
        target_mask=np.vstack(masks).astype(bool),
        target_pv_w=np.vstack(actual_targets).astype(np.float32),
        day_ahead_pv_w=np.vstack(day_ahead_targets).astype(np.float32),
        lead_hours=np.vstack(lead_targets).astype(np.int16),
        week_index=np.asarray(week_indices, dtype=np.int16),
        delivery_date_local=np.asarray(delivery_dates),
        decision_hour_local=np.asarray(decision_hours, dtype=np.int8),
        feature_names=feature_names,
        site_id=np.asarray(site_ids) if site_column is not None else None,
    )


def fit_mimo_residual_mlp(
    train: MIMOSamples,
    validation: MIMOSamples | None,
    *,
    config: MIMOMLPTrainingConfig = MIMOMLPTrainingConfig(),
) -> MIMOMLPFit:
    """Fit a masked Huber-loss MIMO MLP.

    The validation set is optional.  With one, early stopping selects the
    epoch by masked validation MAE in watts.  Without one, the model trains for
    exactly ``max_epochs``; this is useful when refitting the selected epoch on
    all outer-training weeks before evaluation.
    """

    if train.horizon != (validation.horizon if validation is not None else train.horizon):
        raise ValueError("Train and validation horizons must match")
    _validate_config(config)
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as functional

    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    input_scaler = StandardScaler().fit(train.features)
    x_train = input_scaler.transform(train.features).astype(np.float32)
    x_validation = (
        input_scaler.transform(validation.features).astype(np.float32)
        if validation is not None
        else None
    )

    observed_residuals = train.targets_residual_w[train.target_mask]
    target_mean_w = float(observed_residuals.mean())
    target_scale_w = float(observed_residuals.std())
    if target_scale_w <= np.finfo(float).eps:
        target_scale_w = 1.0

    y_train = ((train.targets_residual_w - target_mean_w) / target_scale_w).astype(np.float32)
    y_validation = (
        ((validation.targets_residual_w - target_mean_w) / target_scale_w).astype(np.float32)
        if validation is not None
        else None
    )

    model = _MIMOResidualMLP(
        input_dim=x_train.shape[1],
        output_dim=train.horizon,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)
    mask_train_tensor = torch.from_numpy(train.target_mask.astype(np.float32))
    if validation is not None:
        x_validation_tensor = torch.from_numpy(x_validation)
        y_validation_tensor = torch.from_numpy(y_validation)
        mask_validation_tensor = torch.from_numpy(validation.target_mask.astype(np.float32))

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_validation_mae_w: float | None = None
    no_improvement = 0
    generator = torch.Generator().manual_seed(config.random_state)

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(len(train.features), generator=generator)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            prediction = model(x_train_tensor[batch])
            elementwise_loss = functional.smooth_l1_loss(
                prediction,
                y_train_tensor[batch],
                reduction="none",
                beta=1.0,
            )
            loss = (elementwise_loss * mask_train_tensor[batch]).sum() / (
                mask_train_tensor[batch].sum().clamp_min(1.0)
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        if validation is None:
            best_epoch = epoch
            continue

        model.eval()
        with torch.no_grad():
            validation_prediction = model(x_validation_tensor)
            validation_prediction_w = validation_prediction * target_scale_w + target_mean_w
            validation_target_w = y_validation_tensor * target_scale_w + target_mean_w
            validation_mae_w = float(
                (
                    (validation_prediction_w - validation_target_w).abs()
                    * mask_validation_tensor
                ).sum()
                / mask_validation_tensor.sum().clamp_min(1.0)
            )
        if best_validation_mae_w is None or validation_mae_w < best_validation_mae_w:
            best_validation_mae_w = validation_mae_w
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if config.patience is not None and no_improvement >= config.patience:
                break

    if validation is not None:
        assert best_state is not None
        model.load_state_dict(best_state)
    model.eval()
    return MIMOMLPFit(
        model=model,
        input_scaler=input_scaler,
        target_mean_w=target_mean_w,
        target_scale_w=target_scale_w,
        best_epoch=best_epoch,
        best_validation_mae_w=best_validation_mae_w,
    )


def trajectory_prediction_frame(
    samples: MIMOSamples,
    predicted_residual_w: np.ndarray,
) -> pd.DataFrame:
    """Convert masked MIMO trajectories to MPC decision/target observations."""

    prediction = np.asarray(predicted_residual_w, dtype=float)
    if prediction.shape != samples.targets_residual_w.shape:
        raise ValueError("Predicted residual shape does not match MIMO sample targets")
    rows: list[dict[str, Any]] = []
    for sample_index, target_index in zip(*np.nonzero(samples.target_mask), strict=True):
        base = float(samples.day_ahead_pv_w[sample_index, target_index])
        residual = float(prediction[sample_index, target_index])
        rows.append(
            {
                "delivery_date_local": samples.delivery_date_local[sample_index],
                "week_index": int(samples.week_index[sample_index]),
                "decision_hour_local": int(samples.decision_hour_local[sample_index]),
                "lead_hours": int(samples.lead_hours[sample_index, target_index]),
                "target_pv_power_w": float(samples.target_pv_w[sample_index, target_index]),
                "day_ahead_prediction_w": base,
                "mimo_residual_prediction_w": residual,
                "mimo_prediction_w": max(base + residual, 0.0),
            }
        )
        if samples.site_id is not None:
            rows[-1]["site_id"] = samples.site_id[sample_index]
    return pd.DataFrame(rows)


def refit_config_for_selected_epoch(
    config: MIMOMLPTrainingConfig,
    selected_epoch: int,
) -> MIMOMLPTrainingConfig:
    """Return a deterministic no-validation config for outer-fold refitting."""

    if selected_epoch < 1:
        raise ValueError("selected_epoch must be positive")
    return replace(config, max_epochs=selected_epoch, patience=None)


def _maximum_remaining_active_horizon(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> int:
    maximum = 0
    for _, day in frame.groupby(list(group_columns), sort=False):
        hours = np.sort(day["hour_local"].to_numpy())
        for decision_hour in range(24):
            if np.any(hours < decision_hour):
                maximum = max(maximum, int(np.sum(hours > decision_hour)))
    return maximum


def _feature_names(
    future_features: Sequence[str],
    n_lags: int,
    horizon: int,
    *,
    day_ahead_column: str,
) -> tuple[str, ...]:
    return tuple(
        ["decision_hour_local", "n_observed_residual_lags"]
        + [
            name
            for lag in range(1, n_lags + 1)
            for name in (
                f"residual_lag_{lag}_w",
                f"pv_lag_{lag}_w",
                f"day_ahead_lag_{lag}_w",
                f"residual_lag_{lag}_available",
            )
        ]
        + [
            name
            for lead in range(1, horizon + 1)
            for name in (
                *[f"lead_{lead}_{feature}" for feature in future_features],
                f"lead_{lead}_{day_ahead_column}",
                f"lead_{lead}_available",
            )
        ]
    )


def _make_feature_vector(
    *,
    history: pd.DataFrame,
    future: pd.DataFrame,
    decision_hour: int,
    future_features: Sequence[str],
    n_lags: int,
    horizon: int,
    target_column: str,
    day_ahead_column: str,
) -> np.ndarray:
    values: list[float] = [float(decision_hour), float(len(history))]
    recent_history = history.iloc[::-1]
    for lag in range(n_lags):
        if lag < len(recent_history):
            row = recent_history.iloc[lag]
            values.extend(
                [
                    float(row["_residual_w"]),
                    float(row[target_column]),
                    float(row[day_ahead_column]),
                    1.0,
                ]
            )
        else:
            values.extend([0.0, 0.0, 0.0, 0.0])

    for lead in range(horizon):
        if lead < len(future):
            row = future.iloc[lead]
            values.extend([float(row[feature]) for feature in future_features])
            values.append(float(row[day_ahead_column]))
            values.append(1.0)
        else:
            values.extend([0.0] * (len(future_features) + 1))
            values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def _make_targets(
    future: pd.DataFrame,
    *,
    decision_hour: int,
    horizon: int,
    target_column: str,
    day_ahead_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual = np.zeros(horizon, dtype=np.float32)
    mask = np.zeros(horizon, dtype=bool)
    actual = np.zeros(horizon, dtype=np.float32)
    day_ahead = np.zeros(horizon, dtype=np.float32)
    lead = np.zeros(horizon, dtype=np.int16)
    for index, (_, row) in enumerate(future.iterrows()):
        residual[index] = float(row["_residual_w"])
        mask[index] = True
        actual[index] = float(row[target_column])
        day_ahead[index] = float(row[day_ahead_column])
        lead[index] = int(row["hour_local"] - decision_hour)
    return residual, mask, actual, day_ahead, lead


class _MIMOResidualMLP:  # dynamically subclasses nn.Module when constructed
    def __new__(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        hidden_width: int,
        hidden_layers: int,
        dropout: float,
    ) -> Any:
        torch = _torch()
        import torch.nn as nn

        if hidden_width < 1 or hidden_layers < 1:
            raise ValueError("hidden_width and hidden_layers must be positive")
        layers: list[Any] = []
        width = input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(width, hidden_width), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden_width
        layers.append(nn.Linear(width, output_dim))
        return nn.Sequential(*layers)


def _validate_config(config: MIMOMLPTrainingConfig) -> None:
    if config.hidden_width < 1 or config.hidden_layers < 1:
        raise ValueError("hidden_width and hidden_layers must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if config.batch_size < 1 or config.max_epochs < 1:
        raise ValueError("batch_size and max_epochs must be positive")
    if config.patience is not None and config.patience < 1:
        raise ValueError("patience must be positive or None")


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Install the neural training extra first: python -m pip install -e '.[neural]'"
        ) from error
    return torch


@dataclass(frozen=True)
class LSTMSequenceSamples:
    """MIMO trajectories represented as encoder and decoder sequences."""

    history: np.ndarray
    history_length: np.ndarray
    future: np.ndarray
    future_length: np.ndarray
    trajectory: MIMOSamples

    def select_weeks(self, weeks: Iterable[int]) -> "LSTMSequenceSamples":
        """Select complete weeks while preserving sequence and target alignment."""

        mask = np.isin(self.trajectory.week_index, list(weeks))
        if not mask.any():
            raise ValueError("The requested weeks contain no LSTM samples")
        return LSTMSequenceSamples(
            history=self.history[mask],
            history_length=self.history_length[mask],
            future=self.future[mask],
            future_length=self.future_length[mask],
            trajectory=self.trajectory.select_weeks(weeks),
        )


@dataclass(frozen=True)
class MIMOLSTMTrainingConfig:
    """Small encoder-decoder LSTM settings for the one-site MIMO comparison."""

    hidden_size: int = 24
    num_layers: int = 1
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 300
    patience: int | None = 30
    random_state: int = 42


@dataclass(frozen=True)
class MIMOLSTMFit:
    """Fitted encoder-decoder residual model and selection diagnostics."""

    model: Any
    history_scaler: StandardScaler
    future_scaler: StandardScaler
    target_mean_w: float
    target_scale_w: float
    best_epoch: int
    best_validation_mae_w: float | None

    def predict_residual_w(self, samples: LSTMSequenceSamples) -> np.ndarray:
        """Predict a residual vector without using any future actual PV values."""

        torch = _torch()
        history = _scale_sequence(
            samples.history,
            samples.history_length,
            self.history_scaler,
        )
        future = _scale_sequence(
            samples.future,
            samples.future_length,
            self.future_scaler,
        )
        self.model.eval()
        with torch.no_grad():
            scaled = self.model(
                torch.from_numpy(history),
                torch.from_numpy(samples.history_length.astype(np.int64)),
                torch.from_numpy(future),
                torch.from_numpy(samples.future_length.astype(np.int64)),
            ).cpu().numpy()
        return scaled * self.target_scale_w + self.target_mean_w


def build_lstm_sequence_samples(
    trajectory: MIMOSamples,
    *,
    future_features: Sequence[str],
    n_lags: int,
    day_ahead_column: str = DEFAULT_DAY_AHEAD_COLUMN,
) -> LSTMSequenceSamples:
    """Turn flattened MIMO features into chronological encoder/decoder inputs.

    The LSTM encoder reads actual PV, frozen day-ahead PV and their residual
    for completed hours in chronological order.  The decoder reads only known
    future variables: the day-ahead PV trajectory plus future weather, calendar
    and solar features.  It never receives future actual PV or target residuals.
    """

    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    index_by_name = {name: index for index, name in enumerate(trajectory.feature_names)}
    required = {
        "n_observed_residual_lags",
        *[
            name
            for lag in range(1, n_lags + 1)
            for name in (
                f"residual_lag_{lag}_w",
                f"pv_lag_{lag}_w",
                f"day_ahead_lag_{lag}_w",
            )
        ],
        *[
            f"lead_{lead}_{feature}"
            for lead in range(1, trajectory.horizon + 1)
            for feature in (*future_features, day_ahead_column)
        ],
    }
    missing = sorted(required.difference(index_by_name))
    if missing:
        raise KeyError(
            "MIMO features do not include the required LSTM inputs; rebuild "
            f"MIMO samples. Missing: {missing}"
        )

    n_samples = len(trajectory.features)
    history = np.zeros((n_samples, n_lags, 3), dtype=np.float32)
    history_length = np.zeros(n_samples, dtype=np.int64)
    future = np.zeros(
        (n_samples, trajectory.horizon, len(future_features) + 1),
        dtype=np.float32,
    )
    future_length = trajectory.target_mask.sum(axis=1).astype(np.int64)

    for sample_index in range(n_samples):
        observed_count = int(
            round(trajectory.features[sample_index, index_by_name["n_observed_residual_lags"]])
        )
        observed_count = min(max(observed_count, 1), n_lags)
        history_length[sample_index] = observed_count
        # lag 1 is newest.  The LSTM must see oldest -> newest.
        for position, lag in enumerate(range(observed_count, 0, -1)):
            history[sample_index, position] = [
                trajectory.features[
                    sample_index, index_by_name[f"residual_lag_{lag}_w"]
                ],
                trajectory.features[sample_index, index_by_name[f"pv_lag_{lag}_w"]],
                trajectory.features[
                    sample_index, index_by_name[f"day_ahead_lag_{lag}_w"]
                ],
            ]
        for lead in range(1, trajectory.horizon + 1):
            future[sample_index, lead - 1] = [
                trajectory.features[
                    sample_index, index_by_name[f"lead_{lead}_{feature}"]
                ]
                for feature in (*future_features, day_ahead_column)
            ]

    return LSTMSequenceSamples(
        history=history,
        history_length=history_length,
        future=future,
        future_length=future_length,
        trajectory=trajectory,
    )


def fit_mimo_residual_lstm(
    train: LSTMSequenceSamples,
    validation: LSTMSequenceSamples | None,
    *,
    config: MIMOLSTMTrainingConfig = MIMOLSTMTrainingConfig(),
) -> MIMOLSTMFit:
    """Fit an encoder-decoder LSTM with a masked Huber trajectory loss."""

    if validation is not None and train.trajectory.horizon != validation.trajectory.horizon:
        raise ValueError("Train and validation horizons must match")
    _validate_lstm_config(config)
    torch = _torch()
    import torch.nn.functional as functional

    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    history_scaler = _fit_sequence_scaler(train.history, train.history_length)
    future_scaler = _fit_sequence_scaler(train.future, train.future_length)
    history_train = _scale_sequence(train.history, train.history_length, history_scaler)
    future_train = _scale_sequence(train.future, train.future_length, future_scaler)
    if validation is not None:
        history_validation = _scale_sequence(
            validation.history,
            validation.history_length,
            history_scaler,
        )
        future_validation = _scale_sequence(
            validation.future,
            validation.future_length,
            future_scaler,
        )

    observed_residuals = train.trajectory.targets_residual_w[train.trajectory.target_mask]
    target_mean_w = float(observed_residuals.mean())
    target_scale_w = float(observed_residuals.std())
    if target_scale_w <= np.finfo(float).eps:
        target_scale_w = 1.0
    y_train = (
        (train.trajectory.targets_residual_w - target_mean_w) / target_scale_w
    ).astype(np.float32)
    if validation is not None:
        y_validation = (
            (validation.trajectory.targets_residual_w - target_mean_w) / target_scale_w
        ).astype(np.float32)

    model = _make_lstm_model(
        history_dim=history_train.shape[2],
        future_dim=future_train.shape[2],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history_train_tensor = torch.from_numpy(history_train)
    history_length_train_tensor = torch.from_numpy(train.history_length.astype(np.int64))
    future_train_tensor = torch.from_numpy(future_train)
    future_length_train_tensor = torch.from_numpy(train.future_length.astype(np.int64))
    y_train_tensor = torch.from_numpy(y_train)
    mask_train_tensor = torch.from_numpy(train.trajectory.target_mask.astype(np.float32))
    if validation is not None:
        history_validation_tensor = torch.from_numpy(history_validation)
        history_length_validation_tensor = torch.from_numpy(
            validation.history_length.astype(np.int64)
        )
        future_validation_tensor = torch.from_numpy(future_validation)
        future_length_validation_tensor = torch.from_numpy(
            validation.future_length.astype(np.int64)
        )
        y_validation_tensor = torch.from_numpy(y_validation)
        mask_validation_tensor = torch.from_numpy(
            validation.trajectory.target_mask.astype(np.float32)
        )

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_validation_mae_w: float | None = None
    no_improvement = 0
    generator = torch.Generator().manual_seed(config.random_state)

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(len(train.history), generator=generator)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            prediction = model(
                history_train_tensor[batch],
                history_length_train_tensor[batch],
                future_train_tensor[batch],
                future_length_train_tensor[batch],
            )
            elementwise_loss = functional.smooth_l1_loss(
                prediction,
                y_train_tensor[batch],
                reduction="none",
                beta=1.0,
            )
            loss = (elementwise_loss * mask_train_tensor[batch]).sum() / (
                mask_train_tensor[batch].sum().clamp_min(1.0)
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        if validation is None:
            best_epoch = epoch
            continue

        model.eval()
        with torch.no_grad():
            validation_prediction = model(
                history_validation_tensor,
                history_length_validation_tensor,
                future_validation_tensor,
                future_length_validation_tensor,
            )
            validation_prediction_w = validation_prediction * target_scale_w + target_mean_w
            validation_target_w = y_validation_tensor * target_scale_w + target_mean_w
            validation_mae_w = float(
                (
                    (validation_prediction_w - validation_target_w).abs()
                    * mask_validation_tensor
                ).sum()
                / mask_validation_tensor.sum().clamp_min(1.0)
            )
        if best_validation_mae_w is None or validation_mae_w < best_validation_mae_w:
            best_validation_mae_w = validation_mae_w
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if config.patience is not None and no_improvement >= config.patience:
                break

    if validation is not None:
        assert best_state is not None
        model.load_state_dict(best_state)
    model.eval()
    return MIMOLSTMFit(
        model=model,
        history_scaler=history_scaler,
        future_scaler=future_scaler,
        target_mean_w=target_mean_w,
        target_scale_w=target_scale_w,
        best_epoch=best_epoch,
        best_validation_mae_w=best_validation_mae_w,
    )


def refit_lstm_config_for_selected_epoch(
    config: MIMOLSTMTrainingConfig,
    selected_epoch: int,
) -> MIMOLSTMTrainingConfig:
    """Use an early-stopping epoch to refit an LSTM on all outer-train weeks."""

    if selected_epoch < 1:
        raise ValueError("selected_epoch must be positive")
    return replace(config, max_epochs=selected_epoch, patience=None)


def _fit_sequence_scaler(sequence: np.ndarray, lengths: np.ndarray) -> StandardScaler:
    mask = np.arange(sequence.shape[1])[None, :] < lengths[:, None]
    return StandardScaler().fit(sequence[mask])


def _scale_sequence(
    sequence: np.ndarray,
    lengths: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    mask = np.arange(sequence.shape[1])[None, :] < lengths[:, None]
    scaled = np.zeros_like(sequence, dtype=np.float32)
    scaled[mask] = scaler.transform(sequence[mask]).astype(np.float32)
    return scaled


def _make_lstm_model(
    *,
    history_dim: int,
    future_dim: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> Any:
    """Create an LSTM encoder plus future-covariate decoder and linear head."""

    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    class MIMOResidualLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent_dropout = dropout if num_layers > 1 else 0.0
            self.encoder = nn.LSTM(
                input_size=history_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.decoder = nn.LSTM(
                input_size=future_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.output_dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, 1)

        def forward(
            self,
            history: torch.Tensor,
            history_length: torch.Tensor,
            future: torch.Tensor,
            future_length: torch.Tensor,
        ) -> torch.Tensor:
            packed_history = pack_padded_sequence(
                history,
                history_length.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, cell) = self.encoder(packed_history)
            packed_future = pack_padded_sequence(
                future,
                future_length.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            decoded, _ = self.decoder(packed_future, (hidden, cell))
            decoded, _ = pad_packed_sequence(
                decoded,
                batch_first=True,
                total_length=future.shape[1],
            )
            return self.head(self.output_dropout(decoded)).squeeze(-1)

    return MIMOResidualLSTM()


def _validate_lstm_config(config: MIMOLSTMTrainingConfig) -> None:
    if config.hidden_size < 1 or config.num_layers < 1:
        raise ValueError("hidden_size and num_layers must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if config.batch_size < 1 or config.max_epochs < 1:
        raise ValueError("batch_size and max_epochs must be positive")
    if config.patience is not None and config.patience < 1:
        raise ValueError("patience must be positive or None")

"""Rolling-origin conformal calibration for central quantile intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CentralIntervalCQR:
    """Non-negative symmetric CQR corrections, optionally grouped by hour."""

    quantile_levels: tuple[float, ...]
    corrections: pd.DataFrame

    @classmethod
    def fit(
        cls,
        *,
        actual: np.ndarray,
        quantile_predictions: np.ndarray,
        groups: np.ndarray,
        quantile_levels: tuple[float, ...],
    ) -> "CentralIntervalCQR":
        levels = _validate_levels(quantile_levels)
        observations = np.asarray(actual, dtype=float)
        predictions = np.sort(np.asarray(quantile_predictions, dtype=float), axis=1)
        group_values = np.asarray(groups)
        if observations.ndim != 1 or len(observations) != len(predictions):
            raise ValueError("actual and quantile_predictions must have matching rows")
        if predictions.shape[1] != len(levels):
            raise ValueError("quantile_predictions does not match quantile_levels")
        if group_values.shape != observations.shape:
            raise ValueError("groups must contain one value per observation")
        if not np.isfinite(observations).all() or not np.isfinite(predictions).all():
            raise ValueError("calibration inputs must contain only finite values")

        rows: list[dict[str, float | int]] = []
        for group in np.unique(group_values):
            mask = group_values == group
            y = observations[mask]
            q = predictions[mask]
            for lower_index, upper_index in _central_pairs(levels):
                lower_level = levels[lower_index]
                upper_level = levels[upper_index]
                target_coverage = upper_level - lower_level
                scores = np.maximum.reduce(
                    (q[:, lower_index] - y, y - q[:, upper_index], np.zeros(len(y)))
                )
                finite_sample_level = min(
                    1.0,
                    np.ceil((len(scores) + 1) * target_coverage) / len(scores),
                )
                correction = float(
                    np.quantile(scores, finite_sample_level, method="higher")
                )
                rows.append(
                    {
                        "group": int(group),
                        "lower_quantile": float(lower_level),
                        "upper_quantile": float(upper_level),
                        "target_coverage": float(target_coverage),
                        "calibration_rows": int(len(scores)),
                        "raw_coverage": float(
                            ((y >= q[:, lower_index]) & (y <= q[:, upper_index])).mean()
                        ),
                        "score_quantile_level": float(finite_sample_level),
                        "correction": correction,
                    }
                )
        return cls(
            quantile_levels=tuple(float(value) for value in levels),
            corrections=pd.DataFrame(rows),
        )

    def transform(
        self,
        quantile_predictions: np.ndarray,
        groups: np.ndarray,
        *,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> np.ndarray:
        """Apply fitted central-interval corrections and rearrange monotonically."""

        predictions = np.sort(np.asarray(quantile_predictions, dtype=float), axis=1)
        group_values = np.asarray(groups)
        if predictions.shape[1] != len(self.quantile_levels):
            raise ValueError("quantile_predictions does not match fitted levels")
        if group_values.shape != (len(predictions),):
            raise ValueError("groups must contain one value per prediction row")
        result = predictions.copy()
        levels = np.asarray(self.quantile_levels, dtype=float)
        for row in self.corrections.itertuples(index=False):
            mask = group_values == row.group
            if not mask.any():
                continue
            lower_index = int(np.flatnonzero(np.isclose(levels, row.lower_quantile))[0])
            upper_index = int(np.flatnonzero(np.isclose(levels, row.upper_quantile))[0])
            result[mask, lower_index] -= row.correction
            result[mask, upper_index] += row.correction
        result = np.sort(result, axis=1)
        if lower_bound is not None or upper_bound is not None:
            low = -np.inf if lower_bound is None else lower_bound
            high = np.inf if upper_bound is None else upper_bound
            result = np.clip(result, low, high)
        return result


def _validate_levels(values: tuple[float, ...]) -> np.ndarray:
    levels = np.asarray(values, dtype=float)
    if levels.ndim != 1 or len(levels) < 3 or not np.all(np.diff(levels) > 0):
        raise ValueError("quantile_levels must contain at least three increasing values")
    if levels[0] <= 0 or levels[-1] >= 1 or not np.isclose(levels, 0.5).any():
        raise ValueError("quantile_levels must lie inside (0, 1) and contain 0.5")
    for level in levels[levels < 0.5]:
        if not np.isclose(levels, 1.0 - level).any():
            raise ValueError("every lower quantile must have a symmetric upper quantile")
    return levels


def _central_pairs(levels: np.ndarray) -> list[tuple[int, int]]:
    pairs = []
    for lower_index, level in enumerate(levels):
        if level >= 0.5:
            continue
        upper_index = int(np.flatnonzero(np.isclose(levels, 1.0 - level))[0])
        pairs.append((lower_index, upper_index))
    return pairs

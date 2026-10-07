"""Whole-week splits for PV model development."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

LOCAL_TZ: Final = "Europe/Berlin"
SEASONS: Final = ("winter", "spring", "summer", "autumn")


def add_week_index(
    frame: pd.DataFrame,
    *,
    time_column: str = "valid_time_utc",
    week_column: str = "week_index",
) -> pd.DataFrame:
    """Add a consecutive local-calendar week index without mutating ``frame``.

    The index is intended only for split bookkeeping. It should not be passed to
    a forecasting model as a feature.
    """

    if time_column not in frame:
        raise KeyError(f"Missing required time column: {time_column}")
    if frame[time_column].isna().any():
        raise ValueError(f"{time_column} contains missing values")

    result = frame.copy()
    local_time = pd.to_datetime(result[time_column], utc=True, errors="raise").dt.tz_convert(
        LOCAL_TZ
    )
    week_start = local_time.dt.normalize() - pd.to_timedelta(local_time.dt.dayofweek, unit="D")
    result[week_column] = pd.factorize(week_start, sort=True)[0].astype("int16")
    return result


@dataclass(frozen=True)
class WeekSplit:
    """Three non-overlapping dataframes and their assigned whole weeks."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_weeks: tuple[int, ...]
    validation_weeks: tuple[int, ...]
    test_weeks: tuple[int, ...]


@dataclass(frozen=True)
class WeekFold:
    """One cross-validation fold, represented by positional dataframe indices."""

    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_weeks: tuple[int, ...]
    test_weeks: tuple[int, ...]


class SeasonalWeekSplitter:
    """Allocate complete weeks to train, validation and test in every season.

    This is a development split for a full-year PV dataset. It preserves the
    integrity of individual weeks and avoids putting all summer or all winter
    observations into one partition. It is not a substitute for a final
    chronological backtest on a later year.
    """

    def __init__(
        self,
        *,
        train_size: float = 0.70,
        validation_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
        week_column: str = "week_index",
        time_column: str = "valid_time_utc",
    ) -> None:
        sizes = (train_size, validation_size, test_size)
        if any(size < 0.0 for size in sizes) or not np.isclose(sum(sizes), 1.0):
            raise ValueError("Split sizes must be positive and sum to 1.0")
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state
        self.week_column = week_column
        self.time_column = time_column

    def split(self, frame: pd.DataFrame) -> WeekSplit:
        """Return deterministic season-balanced partitions by complete weeks."""

        if self.week_column not in frame:
            raise KeyError(
                f"Missing {self.week_column}. Call add_week_index before splitting."
            )
        if self.time_column not in frame:
            raise KeyError(f"Missing required time column: {self.time_column}")
        if frame.empty:
            raise ValueError("Cannot split an empty dataframe")

        week_seasons = _season_by_week(frame, self.week_column, self.time_column)
        rng = np.random.default_rng(self.random_state)
        assigned: dict[str, list[int]] = {"train": [], "validation": [], "test": []}

        for season in SEASONS:
            weeks = week_seasons.loc[week_seasons["season"] == season, self.week_column].to_numpy()
            if len(weeks) == 0:
                continue
            if len(weeks) < 3:
                raise ValueError(
                    f"Season {season!r} has fewer than three weeks and cannot fill all splits"
                )
            shuffled = rng.permutation(weeks)
            train_count, validation_count, test_count = self._allocation_counts(len(shuffled))
            assigned["test"].extend(sorted(shuffled[:test_count].tolist()))
            assigned["validation"].extend(
                sorted(shuffled[test_count : test_count + validation_count].tolist())
            )
            assigned["train"].extend(sorted(shuffled[-train_count:].tolist()))

        train_weeks = tuple(sorted(assigned["train"]))
        validation_weeks = tuple(sorted(assigned["validation"]))
        test_weeks = tuple(sorted(assigned["test"]))
        return WeekSplit(
            train=frame.loc[frame[self.week_column].isin(train_weeks)].copy(),
            validation=frame.loc[frame[self.week_column].isin(validation_weeks)].copy(),
            test=frame.loc[frame[self.week_column].isin(test_weeks)].copy(),
            train_weeks=train_weeks,
            validation_weeks=validation_weeks,
            test_weeks=test_weeks,
        )

    def _allocation_counts(self, count: int) -> tuple[int, int, int]:
        test_count = max(1, round(count * self.test_size))
        validation_count = max(1, round(count * self.validation_size))
        train_count = count - test_count - validation_count
        if train_count < 1:
            raise ValueError("Not enough weeks to allocate all three splits")
        return train_count, validation_count, test_count


class SeasonalWeekKFold:
    """Season-balanced K-fold cross-validation by complete local calendar weeks.

    Every week appears in exactly one test fold. Within each season, weeks are
    randomly permuted with ``random_state`` and assigned round-robin to folds.
    Thus each fold contains examples from all four seasons when the full annual
    dataset has at least ``n_splits`` usable weeks in each season.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        random_state: int = 42,
        week_column: str = "week_index",
        time_column: str = "valid_time_utc",
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.random_state = random_state
        self.week_column = week_column
        self.time_column = time_column

    def split(self, frame: pd.DataFrame):
        """Yield ``(train_indices, test_indices)`` pairs in sklearn style."""

        for fold in self.split_with_metadata(frame):
            yield fold.train_indices, fold.test_indices

    def split_with_metadata(self, frame: pd.DataFrame):
        """Yield folds with the corresponding whole-week identifiers."""

        if self.week_column not in frame:
            raise KeyError(
                f"Missing {self.week_column}. Call add_week_index before splitting."
            )
        if self.time_column not in frame:
            raise KeyError(f"Missing required time column: {self.time_column}")
        if frame.empty:
            raise ValueError("Cannot split an empty dataframe")

        week_seasons = _season_by_week(frame, self.week_column, self.time_column)
        rng = np.random.default_rng(self.random_state)
        fold_by_week: dict[int, int] = {}
        for season_index, season in enumerate(SEASONS):
            weeks = week_seasons.loc[
                week_seasons["season"] == season, self.week_column
            ].to_numpy()
            if len(weeks) < self.n_splits:
                raise ValueError(
                    f"Season {season!r} has {len(weeks)} weeks, fewer than n_splits={self.n_splits}"
                )
            for position, week in enumerate(rng.permutation(weeks)):
                fold_by_week[int(week)] = (position + season_index) % self.n_splits

        fold_labels = frame[self.week_column].map(fold_by_week)
        if fold_labels.isna().any():
            raise ValueError("At least one week did not receive a fold assignment")
        for fold_index in range(self.n_splits):
            test_mask = fold_labels.eq(fold_index).to_numpy()
            train_indices = np.flatnonzero(~test_mask)
            test_indices = np.flatnonzero(test_mask)
            test_weeks = tuple(
                sorted(frame.iloc[test_indices][self.week_column].unique().tolist())
            )
            train_weeks = tuple(
                sorted(frame.iloc[train_indices][self.week_column].unique().tolist())
            )
            yield WeekFold(
                fold_index=fold_index,
                train_indices=train_indices,
                test_indices=test_indices,
                train_weeks=train_weeks,
                test_weeks=test_weeks,
            )

    def get_n_splits(self) -> int:
        return self.n_splits


def _season_by_week(
    frame: pd.DataFrame,
    week_column: str,
    time_column: str,
) -> pd.DataFrame:
    local_time = pd.to_datetime(frame[time_column], utc=True, errors="raise").dt.tz_convert(
        LOCAL_TZ
    )
    weeks = pd.DataFrame(
        {
            week_column: frame[week_column].to_numpy(),
            "local_time": local_time.to_numpy(),
        }
    )
    first_hour = weeks.sort_values([week_column, "local_time"]).drop_duplicates(
        week_column, keep="first"
    )
    first_hour["season"] = first_hour["local_time"].dt.month.map(_season_from_month)
    return first_hour[[week_column, "season"]].sort_values(week_column)


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"

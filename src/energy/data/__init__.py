"""Point-in-time data contracts and dataset builders."""

from energy.data.builder import BuildResult, TrainingDatasetBuilder
from energy.data.historical import HistoricalDataProvider
from energy.data.intraday_weather import (
    ECMWF_ARCHIVE_START,
    EcmwfIfsArchiveConfig,
    EcmwfIfsArchiveResult,
    EcmwfIfsMaterializationResult,
    MpcWeatherSnapshotResult,
    build_mpc_weather_snapshots,
    collect_ecmwf_ifs_single_runs,
    materialize_ecmwf_ifs_archive,
)
from energy.data.intraday_price_weather import (
    SpatialEcmwfIfsArchiveConfig,
    SpatialEcmwfIfsArchiveResult,
    SpatialEcmwfIfsMaterializationResult,
    collect_spatial_ecmwf_ifs_single_runs,
    materialize_spatial_ecmwf_ifs_archive,
)
from energy.data.splitting import (
    SeasonalWeekKFold,
    SeasonalWeekSplitter,
    WeekFold,
    WeekSplit,
    add_week_index,
)

__all__ = [
    "BuildResult",
    "HistoricalDataProvider",
    "ECMWF_ARCHIVE_START",
    "EcmwfIfsArchiveConfig",
    "EcmwfIfsArchiveResult",
    "EcmwfIfsMaterializationResult",
    "MpcWeatherSnapshotResult",
    "build_mpc_weather_snapshots",
    "collect_ecmwf_ifs_single_runs",
    "materialize_ecmwf_ifs_archive",
    "SpatialEcmwfIfsArchiveConfig",
    "SpatialEcmwfIfsArchiveResult",
    "SpatialEcmwfIfsMaterializationResult",
    "collect_spatial_ecmwf_ifs_single_runs",
    "materialize_spatial_ecmwf_ifs_archive",
    "SeasonalWeekKFold",
    "SeasonalWeekSplitter",
    "TrainingDatasetBuilder",
    "WeekSplit",
    "WeekFold",
    "add_week_index",
]

from energy.data.multisite import MultiSitePVBuildResult, build_multisite_day_ahead_pv

from __future__ import annotations

import json
import os
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor

from energy.data import HistoricalDataProvider, SeasonalWeekKFold, SeasonalWeekSplitter
from energy.modeling import CrossFittedPVExperiment, load_pv_model_frame


DATA_ROOT = Path(os.environ.get("ENERGY_TEST_DATA_ROOT", "data"))
YEAR = 2024


class TrainingDatasetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.da_pv = pd.read_parquet(
            DATA_ROOT / "features" / "day_ahead_pv" / f"day_ahead_pv_{YEAR}.parquet"
        )
        cls.da_price = pd.read_parquet(
            DATA_ROOT / "features" / "day_ahead_price" / f"day_ahead_price_{YEAR}.parquet"
        )
        cls.oracle_pv = pd.read_parquet(
            DATA_ROOT / "features" / "oracle_pv" / f"oracle_pv_{YEAR}.parquet"
        )
        cls.mpc_pv = pd.read_parquet(
            DATA_ROOT / "features" / "mpc_pv" / f"mpc_pv_{YEAR}.parquet"
        )

    def test_all_targets_are_in_requested_year(self) -> None:
        for frame in (self.da_pv, self.oracle_pv, self.da_price, self.mpc_pv):
            years = frame["delivery_date_local"].str.slice(0, 4).unique().tolist()
            self.assertEqual(years, [str(YEAR)])

    def test_day_ahead_feature_availability(self) -> None:
        for frame in (self.da_pv, self.da_price):
            as_of = pd.to_datetime(frame["as_of_utc"], utc=True)
            weather_available = pd.to_datetime(
                frame["weather_forecast_available_at_utc"], utc=True
            )
            self.assertTrue((weather_available <= as_of).all())
        pv_target_available = pd.to_datetime(
            self.da_pv["target_available_at_utc"], utc=True
        )
        self.assertTrue((pv_target_available > pd.to_datetime(self.da_pv["as_of_utc"], utc=True)).all())
        as_of = pd.to_datetime(self.da_price["as_of_utc"], utc=True)
        for column in (
            "price_lag_24h_available_at_utc",
            "price_lag_168h_available_at_utc",
        ):
            available = pd.to_datetime(self.da_price[column], utc=True)
            self.assertTrue((available <= as_of).all())
        target_available = pd.to_datetime(
            self.da_price["target_available_at_utc"], utc=True
        )
        self.assertTrue((target_available > as_of).all())

    def test_day_ahead_rows_share_one_decision_time_per_delivery_day(self) -> None:
        for frame in (self.da_pv, self.da_price):
            decision_counts = frame.groupby("delivery_date_local")["as_of_utc"].nunique()
            self.assertTrue((decision_counts == 1).all())
            self.assertTrue(set(frame["weather_forecast_lead_hours"].unique()) <= {24, 48})

    def test_day_ahead_pv_has_a_consecutive_training_week_index(self) -> None:
        weeks = sorted(self.da_pv["week_index"].unique().tolist())
        self.assertEqual(weeks, list(range(len(weeks))))

    def test_oracle_pv_has_same_target_rows_and_actual_weather(self) -> None:
        self.assertEqual(len(self.oracle_pv), len(self.da_pv))
        self.assertListEqual(
            self.oracle_pv["valid_time_utc"].tolist(),
            self.da_pv["valid_time_utc"].tolist(),
        )
        self.assertListEqual(
            self.oracle_pv["week_index"].tolist(),
            self.da_pv["week_index"].tolist(),
        )
        actual_columns = [
            column for column in self.oracle_pv if column.startswith("weather_actual_")
        ]
        self.assertFalse(self.oracle_pv[actual_columns].isna().any().any())

    def test_seasonal_week_splitter_uses_whole_weeks_and_all_seasons(self) -> None:
        split = SeasonalWeekSplitter(random_state=7).split(self.da_pv)
        week_sets = [set(split.train_weeks), set(split.validation_weeks), set(split.test_weeks)]
        self.assertFalse(week_sets[0] & week_sets[1])
        self.assertFalse(week_sets[0] & week_sets[2])
        self.assertFalse(week_sets[1] & week_sets[2])
        self.assertEqual(set.union(*week_sets), set(self.da_pv["week_index"].unique()))
        self.assertEqual(
            len(split.train) + len(split.validation) + len(split.test), len(self.da_pv)
        )
        for partition in (split.train, split.validation, split.test):
            local_months = pd.to_datetime(partition["valid_time_utc"], utc=True).dt.tz_convert(
                "Europe/Berlin"
            ).dt.month
            seasons = {
                "winter" if month in (12, 1, 2) else "spring" if month in (3, 4, 5)
                else "summer" if month in (6, 7, 8) else "autumn"
                for month in local_months
            }
            self.assertEqual(seasons, {"winter", "spring", "summer", "autumn"})

    def test_seasonal_week_kfold_uses_every_week_once_for_testing(self) -> None:
        folds = list(SeasonalWeekKFold(n_splits=5, random_state=7).split_with_metadata(self.da_pv))
        tested_weeks = [set(fold.test_weeks) for fold in folds]
        self.assertEqual(set.union(*tested_weeks), set(self.da_pv["week_index"].unique()))
        for first_index, first in enumerate(tested_weeks):
            for second in tested_weeks[first_index + 1 :]:
                self.assertFalse(first & second)

    def test_nested_crossfit_generates_one_oof_prediction_per_pv_row(self) -> None:
        project_root = DATA_ROOT.parent
        frame = load_pv_model_frame(project_root, year=YEAR)
        factory = lambda _: DummyRegressor(strategy="mean")
        result = CrossFittedPVExperiment(
            outer_splits=3,
            inner_splits=3,
            oracle_factory=factory,
            forecast_factory=factory,
        ).run(frame)
        self.assertEqual(len(result.oof_predictions), len(frame))
        self.assertEqual(sorted(result.oof_predictions["outer_fold"].unique()), [0, 1, 2])
        self.assertFalse(result.oof_predictions.isna().any().any())
        self.assertEqual(len(result.fold_metrics), 3)
        self.assertIn("baseline_forecast_prediction_w", result.oof_predictions)
        self.assertIn("baseline_mae_w", result.average_fold_metrics)
        self.assertIn("forecast_mae_w", result.average_fold_metrics)

    def test_mpc_features_do_not_use_unavailable_pv(self) -> None:
        as_of = pd.to_datetime(self.mpc_pv["as_of_utc"], utc=True)
        available = pd.to_datetime(
            self.mpc_pv["pv_last_observation_available_at_utc"], utc=True
        )
        self.assertTrue((available <= as_of).all())
        self.assertEqual(sorted(self.mpc_pv["horizon_hours"].unique().tolist()), list(range(1, 11)))

    def test_suspected_outages_are_excluded(self) -> None:
        excluded = {"2024-05-10", "2024-07-07"}
        self.assertFalse(bool(set(self.da_pv["delivery_date_local"]) & excluded))
        self.assertFalse(bool(set(self.mpc_pv["delivery_date_local"]) & excluded))

    def test_required_values_are_complete(self) -> None:
        da_pv_required = [
            column for column in self.da_pv if column.startswith("weather_forecast_")
        ]
        da_price_required = ["price_lag_24h", "price_lag_168h", "day_ahead_price_eur_per_mwh"]
        mpc_required = ["pv_last_observed_w", "pv_target_lag_24h_w", "pv_power_mean_w"]
        self.assertFalse(self.da_pv[da_pv_required].isna().any().any())
        self.assertFalse(self.da_price[da_price_required].isna().any().any())
        self.assertFalse(self.mpc_pv[mpc_required].isna().any().any())

    def test_manifest_marks_intraday_price_unavailable(self) -> None:
        manifest = json.loads(
            (DATA_ROOT / "metadata" / f"training_datasets_{YEAR}_manifest.json").read_text()
        )
        self.assertIn("intraday_price", manifest["unavailable_datasets"])

    def test_historical_provider_hides_future_day_ahead_facts(self) -> None:
        provider = HistoricalDataProvider(
            DATA_ROOT / "curated" / f"canonical_hourly_{YEAR}.parquet"
        )
        context = provider.get_day_ahead_context(
            as_of=pd.Timestamp("2024-05-31T09:00:00Z").to_pydatetime(),
            delivery_day=date(2024, 6, 1),
        )
        self.assertEqual(len(context.delivery_rows), 24)
        for column in (
            "pv_power_mean_w",
            "pv_energy_kwh",
            "day_ahead_price_eur_per_mwh",
            "weather_actual_temperature_2m",
        ):
            self.assertTrue(context.delivery_rows[column].isna().all())
        forecast_columns = [
            column
            for column in context.delivery_rows
            if column.startswith("weather_forecast_")
            and column not in {"weather_forecast_available_at_utc"}
        ]
        self.assertFalse(context.delivery_rows[forecast_columns].isna().any().any())
        forecast_available = pd.to_datetime(
            context.delivery_rows["weather_forecast_available_at_utc"], utc=True
        )
        self.assertTrue((forecast_available <= pd.Timestamp(context.as_of)).all())
        history_times = pd.to_datetime(context.history_rows["valid_time_utc"], utc=True)
        self.assertTrue((history_times <= pd.Timestamp(context.as_of)).all())

    def test_historical_provider_mpc_uses_point_in_time_boundary(self) -> None:
        provider = HistoricalDataProvider(
            DATA_ROOT / "curated" / f"canonical_hourly_{YEAR}.parquet"
        )
        context = provider.get_mpc_context(
            as_of=pd.Timestamp("2024-06-01T08:00:00Z").to_pydatetime(),
            horizon_end=pd.Timestamp("2024-06-01T18:00:00Z").to_pydatetime(),
        )
        self.assertEqual(len(context.future_rows), 10)
        self.assertTrue(context.future_rows["pv_power_mean_w"].isna().all())
        self.assertTrue(context.future_rows["weather_actual_temperature_2m"].isna().all())
        observed_available = pd.to_datetime(
            context.observed_rows["pv_available_at_utc"], utc=True
        )
        observed_values = context.observed_rows["pv_power_mean_w"].notna()
        self.assertTrue(
            (observed_available.loc[observed_values] <= pd.Timestamp(context.as_of)).all()
        )


if __name__ == "__main__":
    unittest.main()

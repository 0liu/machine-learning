#!/usr/bin/env python3

from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
import pytz
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

from scipy.stats import randint
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


class TimezoneByCity:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="geoapiExercises")
        self.tzfinder = TimezoneFinder()

    def tz_name(self, city: str):
        loc = self.geolocator.geocode(city)
        tz_name = self.tzfinder.timezone_at(lng=loc.longitude, lat=loc.latitude)
        return tz_name

    def tz(self, city: str):
        tz_name = self.tz_name(city)
        return pytz.timezone(tz_name)


class CatchJoe:
    def __init__(self, model_path: str = "") -> None:
        self._model_path = (
            Path(model_path).expanduser().resolve()
        )  # defualt to current path '.'

    def load_data(self, data_file: str) -> pd.DataFrame:
        """Load data from json file."""
        data_file_path = Path(data_file).expanduser().resolve()
        if self._model_path is None:
            self._model_path = data_file_path.parent
        with open(data_file_path, "r") as f:
            data_json_struct = json.loads(f.read())
        user_sessions = pd.DataFrame(data_json_struct)
        return user_sessions

    def prepare_data(self, user_sessions: pd.DataFrame) -> pd.DataFrame:
        """
        Clean / Impute data;
        Convert start date/time to local time, and split to year/month/day/weekday and start time;
        Sine/Cosine transform of local time;
        Split location;
        Add length_session;
        Generate sites_corpus.
        """
        # impute empty sites
        empty_sites_index = user_sessions.query("sites.str.len() == 0").index
        user_sessions.loc[empty_sites_index, "sites"] = user_sessions.loc[
            empty_sites_index
        ]["sites"].apply(lambda sites: sites + [{"site": "NONE.NONE", "length": 0}])

        # Combine date/time columns and convert from string to datetime type
        user_sessions["start_dt"] = pd.to_datetime(
            user_sessions["date"] + " " + user_sessions["time"], utc=True
        )

        # Convert to local date time
        tz_by_city = TimezoneByCity()
        timezone_tbl = {
            loc: tz_by_city.tz_name(loc.split("/")[1])
            for loc in user_sessions.location.unique()
        }
        user_sessions["local_time"] = user_sessions.apply(
            lambda row: row["start_dt"]
            .tz_convert(timezone_tbl[row["location"]])
            .tz_localize(None),
            axis=1,
        )

        # Split start date/time to year / month / day / weekday and start_hour
        user_sessions["year"] = user_sessions.local_time.dt.year
        user_sessions["month"] = user_sessions.local_time.dt.month
        user_sessions["day"] = user_sessions.local_time.dt.day
        user_sessions["weekday"] = user_sessions.local_time.dt.weekday
        user_sessions["start_hour"] = user_sessions.local_time.dt.hour

        # Sine/Cosine transform of local start time
        start_dt_normalized = (
            (user_sessions["local_time"] - user_sessions["local_time"].dt.normalize())
            / pd.Timedelta("1 second")
            / 86400
        )
        user_sessions["start_sin"] = np.sin(2 * np.pi * (start_dt_normalized))
        user_sessions["start_cos"] = np.cos(2 * np.pi * (start_dt_normalized))

        # Split location to country and city
        user_sessions[["country", "city"]] = user_sessions["location"].str.split(
            "/", expand=True
        )

        # Get total length of each user session
        user_sessions["length_session"] = user_sessions["sites"].apply(
            lambda session_sites: sum(
                site_entry["length"] for site_entry in session_sites
            )
        )

        #
        user_sessions["sites_corpus"] = user_sessions["sites"].apply(
            lambda session_sites: " ".join(
                site_entry["site"] for site_entry in session_sites
            )
        )

        # Drop off original date/time columns
        user_sessions.drop(
            ["time", "date", "start_dt", "local_time", "location"], axis=1, inplace=True
        )

        return user_sessions

    def fit_encoder(self, user_sessions: pd.DataFrame):
        cat_cols = ["browser", "os", "locale", "gender", "country", "city", "weekday"]
        num_cols = ["start_sin", "start_cos", "length_session"]
        feature_encode_pipeline = ColumnTransformer(
            [
                ("cat_encoder", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num_scaler", StandardScaler(), num_cols),
                (
                    "site_count_tfidf",
                    TfidfVectorizer(
                        token_pattern=r"(?u)\b[-\w@:%.\+~#=][-\w@:%.\+~#=]+\b",
                        ngram_range=(3, 7),
                        max_features=5000,
                    ),
                    "sites_corpus",
                ),
            ]
        )
        feature_encode_pipeline = feature_encode_pipeline.fit(user_sessions)
        joblib.dump(
            feature_encode_pipeline,
            self._model_path.joinpath("feature_encode_pipeline.pkl"),
        )
        return feature_encode_pipeline

    def fit(self, train_data_file: str):
        """
        Fit encoding pipeline and model from a training data file.
        """
        print("Load data ...")
        user_sessions = self.load_data(train_data_file)
        print("Prepare data ...")
        user_sessions = self.prepare_data(user_sessions)
        print("Fit encoder ...")
        feature_encode_pipeline = self.fit_encoder(user_sessions)
        print("Encode data ...")
        X = feature_encode_pipeline.transform(user_sessions)
        y = (
            (user_sessions["user_id"] != 0).astype(int).values
        )  # joe = 0, other user = 1

        print("Optimize model ...")
        param_distribs = {
            "n_estimators": randint(low=50, high=1000),
            "max_features": randint(low=50, high=X.shape[1]),
            "max_depth": randint(low=1, high=100),
            "min_samples_leaf": randint(low=1, high=100),
            "min_samples_split": randint(low=2, high=100),
        }
        gb_clf = GradientBoostingClassifier(random_state=42, n_iter_no_change=5)
        gb_rnd_search = RandomizedSearchCV(
            gb_clf,
            param_distributions=param_distribs,
            n_iter=30,
            scoring="f1",
            n_jobs=-1,
            refit=True,
            cv=5,
            random_state=42,
        )
        gb_rnd_search = gb_rnd_search.fit(X, y)
        model = gb_rnd_search.best_estimator_

        print("Save model ...")
        joblib.dump(model, self._model_path.joinpath("model.pkl"))
        print("Done.")
        return model

    def predict(self, test_data_file: str) -> np.array:
        """
        Predict labels from a test data file. Joe=0, other user=1.
        """
        print("Load data ...")
        user_sessions = self.load_data(test_data_file)
        print("Prepare data ...")
        user_sessions = self.prepare_data(user_sessions)
        print("Load encoder ...")
        feature_encode_pipeline = joblib.load(
            self._model_path.joinpath("feature_encode_pipeline.pkl")
        )
        print("Load model ...")
        model = joblib.load(self._model_path.joinpath("model.pkl"))
        print("Encode data ...")
        X = feature_encode_pipeline.transform(user_sessions)
        print("Predict targets ...")
        y_pred = model.predict(X)

        data_path = Path(test_data_file).expanduser().resolve().parent
        result_file = data_path.joinpath("result.csv")
        print(f"Write results to {result_file} ...")
        pd.DataFrame(y_pred).to_csv(result_file, index=False, header=False)
        print("Done.")
        return y_pred

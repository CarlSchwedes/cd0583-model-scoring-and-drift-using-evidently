from pathlib import Path
import io
import zipfile

import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor

from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset
from evidently.metrics import ValueDrift

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"


def load_data() -> pd.DataFrame:
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        raw_data = pd.read_csv(
            archive.open("hour.csv"),
            header=0,
            sep=",",
            parse_dates=["dteday"],
            index_col="dteday",
        )

    return raw_data


def build_datasets(raw_data: pd.DataFrame):
    target = "cnt"
    prediction = "prediction"
    numerical_features = ["temp", "atemp", "hum", "windspeed", "hr", "weekday"]
    categorical_features = ["season", "holiday", "workingday"]

    reference = raw_data.loc["2011-01-01 00:00:00":"2011-01-28 23:00:00"].copy()
    current = raw_data.loc["2011-01-29 00:00:00":"2011-02-28 23:00:00"].copy()

    regressor = RandomForestRegressor(
        random_state=0,
        n_estimators=50,
    )

    feature_columns = numerical_features + categorical_features

    regressor.fit(reference[feature_columns], reference[target])

    reference[prediction] = regressor.predict(reference[feature_columns])
    current[prediction] = regressor.predict(current[feature_columns])

    data_definition = DataDefinition(
        numerical_columns=numerical_features,
        categorical_columns=categorical_features,
        regression=[Regression(target=target, prediction=prediction)],
    )

    reference_dataset = Dataset.from_pandas(reference, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current, data_definition=data_definition)

    drift_definition = DataDefinition(
        numerical_columns=numerical_features
    )
    reference_drift_dataset = Dataset.from_pandas(
        reference[numerical_features], data_definition=drift_definition
    )
    current_drift_dataset = Dataset.from_pandas(
        current[numerical_features], data_definition=drift_definition
    )

    return (
        reference_dataset,
        current_dataset,
        reference_drift_dataset,
        current_drift_dataset,
    )


def save_regression_reports(reference_dataset: Dataset, current_dataset: Dataset):
    report = Report([RegressionPreset()])

    report.run(reference_dataset, None)
    report.save_html(str(STATIC_DIR / "index.html"))

    week1 = Dataset.from_pandas(
        current_dataset.as_dataframe().loc["2011-01-29 00:00:00":"2011-02-07 23:00:00"],
        data_definition=current_dataset.data_definition,
    )
    report.run(week1, reference_dataset)
    report.save_html(str(STATIC_DIR / "regression_performance_after_week1.html"))

    week2 = Dataset.from_pandas(
        current_dataset.as_dataframe().loc["2011-02-08 00:00:00":"2011-02-14 23:00:00"],
        data_definition=current_dataset.data_definition,
    )
    report.run(week2, reference_dataset)
    report.save_html(str(STATIC_DIR / "regression_performance_after_week2.html"))

    week3 = Dataset.from_pandas(
        current_dataset.as_dataframe().loc["2011-02-15 00:00:00":"2011-02-21 23:00:00"],
        data_definition=current_dataset.data_definition,
    )
    report.run(week3, reference_dataset)
    report.save_html(str(STATIC_DIR / "regression_performance_after_week3.html"))


def save_target_drift_reports(reference_dataset: Dataset, current_dataset: Dataset):
    report = Report([ValueDrift(column="cnt")])

    week1 = Dataset.from_pandas(
        current_dataset.as_dataframe().loc["2011-01-29 00:00:00":"2011-02-07 23:00:00"],
        data_definition=current_dataset.data_definition,
    )
    report.run(week1, reference_dataset)
    report.save_html(str(STATIC_DIR / "target_drift_after_week1.html"))

    week2 = Dataset.from_pandas(
        current_dataset.as_dataframe().loc["2011-02-08 00:00:00":"2011-02-14 23:00:00"],
        data_definition=current_dataset.data_definition,
    )
    report.run(week2, reference_dataset)
    report.save_html(str(STATIC_DIR / "target_drift_after_week2.html"))

    week3 = Dataset.from_pandas(
        current_dataset.as_dataframe().loc["2011-02-15 00:00:00":"2011-02-21 23:00:00"],
        data_definition=current_dataset.data_definition,
    )
    report.run(week3, reference_dataset)
    report.save_html(str(STATIC_DIR / "target_drift_after_week3.html"))


def save_data_drift_reports(reference_drift_dataset: Dataset, current_drift_dataset: Dataset):
    report = Report([DataDriftPreset()])

    week1 = Dataset.from_pandas(
        current_drift_dataset.as_dataframe().loc["2011-01-29 00:00:00":"2011-02-07 23:00:00"],
        data_definition=current_drift_dataset.data_definition,
    )
    report.run(week1, reference_drift_dataset)
    report.save_html(str(STATIC_DIR / "data_drift_dashboard_after_week1.html"))

    week2 = Dataset.from_pandas(
        current_drift_dataset.as_dataframe().loc["2011-02-08 00:00:00":"2011-02-14 23:00:00"],
        data_definition=current_drift_dataset.data_definition,
    )
    report.run(week2, reference_drift_dataset)
    report.save_html(str(STATIC_DIR / "data_drift_dashboard_after_week2.html"))


raw_data = load_data()
reference_dataset, current_dataset, reference_drift_dataset, current_drift_dataset = build_datasets(raw_data)

save_regression_reports(reference_dataset, current_dataset)
save_target_drift_reports(reference_dataset, current_dataset)
save_data_drift_reports(reference_drift_dataset, current_drift_dataset)

app = FastAPI()
app.mount("/", StaticFiles(directory="static", html=True), name="static")

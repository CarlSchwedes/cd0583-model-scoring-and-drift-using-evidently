from pathlib import Path
import io
import zipfile

import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import (
    DataDriftTab,
    NumTargetDriftTab,
    RegressionPerformanceTab,
)
from evidently.pipeline.column_mapping import ColumnMapping

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

    reference = raw_data.loc["2011-01-01":"2011-01-28"].copy()
    current = raw_data.loc["2011-01-29":"2011-02-28"].copy()

    model = RandomForestRegressor(
        random_state=0,
        n_estimators=50,
    )

    features = numerical_features + categorical_features

    model.fit(reference[features], reference[target])

    reference[prediction] = model.predict(reference[features])
    current[prediction] = model.predict(current[features])

    return {
        "target": target,
        "prediction": prediction,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "reference": reference,
        "current": current,
    }


def save_regression_reports(reference, current, column_mapping):
    dashboard = Dashboard(tabs=[RegressionPerformanceTab()])

    dashboard.calculate(reference, None, column_mapping=column_mapping)
    dashboard.save(str(STATIC_DIR / "index.html"))

    week1 = current.loc["2011-01-29":"2011-02-07"].copy()
    dashboard.calculate(reference, week1, column_mapping=column_mapping)
    dashboard.save(str(STATIC_DIR / "regression_performance_after_week1.html"))

    week2 = current.loc["2011-02-08":"2011-02-14"].copy()
    dashboard.calculate(reference, week2, column_mapping=column_mapping)
    dashboard.save(str(STATIC_DIR / "regression_performance_after_week2.html"))

    week3 = current.loc["2011-02-15":"2011-02-21"].copy()
    dashboard.calculate(reference, week3, column_mapping=column_mapping)
    dashboard.save(str(STATIC_DIR / "regression_performance_after_week3.html"))


def save_target_drift_reports(reference, current, column_mapping):
    dashboard = Dashboard(tabs=[NumTargetDriftTab()])

    week1 = current.loc["2011-01-29":"2011-02-07"].copy()
    dashboard.calculate(reference, week1, column_mapping=column_mapping)
    dashboard.save(str(STATIC_DIR / "target_drift_after_week1.html"))

    week2 = current.loc["2011-02-08":"2011-02-14"].copy()
    dashboard.calculate(reference, week2, column_mapping=column_mapping)
    dashboard.save(str(STATIC_DIR / "target_drift_after_week2.html"))

    week3 = current.loc["2011-02-15":"2011-02-21"].copy()
    dashboard.calculate(reference, week3, column_mapping=column_mapping)
    dashboard.save(str(STATIC_DIR / "target_drift_after_week3.html"))


def save_data_drift_reports(reference, current, numerical_features):
    mapping = ColumnMapping()
    mapping.numerical_features = numerical_features

    dashboard = Dashboard(tabs=[DataDriftTab()])

    week1 = current.loc["2011-01-29":"2011-02-07"].copy()
    dashboard.calculate(reference, week1, column_mapping=mapping)
    dashboard.save(str(STATIC_DIR / "data_drift_dashboard_after_week1.html"))

    week2 = current.loc["2011-02-08":"2011-02-14"].copy()
    dashboard.calculate(reference, week2, column_mapping=mapping)
    dashboard.save(str(STATIC_DIR / "data_drift_dashboard_after_week2.html"))


def main():
    raw_data = load_data()
    artifacts = build_datasets(raw_data)

    mapping = ColumnMapping()
    mapping.target = artifacts["target"]
    mapping.prediction = artifacts["prediction"]
    mapping.numerical_features = artifacts["numerical_features"]
    mapping.categorical_features = artifacts["categorical_features"]

    save_regression_reports(
        artifacts["reference"],
        artifacts["current"],
        mapping,
    )
    save_target_drift_reports(
        artifacts["reference"],
        artifacts["current"],
        mapping,
    )
    save_data_drift_reports(
        artifacts["reference"],
        artifacts["current"],
        artifacts["numerical_features"],
    )

    print(f"Reports saved to: {STATIC_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.preprocessing import BASE_CATEGORICAL_FEATURES, BASE_NUMERIC_FEATURES, POST_PUBLICATION_COLUMNS, SEED

PRIMARY_METRIC = "mae"


@dataclass
class ExperimentResult:
    model_name: str
    split: str
    mae: float
    rmse: float
    r2: float


def _metrics(y_true_log: pd.Series, y_pred_log: np.ndarray, model_name: str, split: str) -> ExperimentResult:
    predictions = np.clip(np.expm1(y_pred_log), a_min=0.0, a_max=None)
    truth = np.expm1(y_true_log.to_numpy())
    return ExperimentResult(
        model_name=model_name,
        split=split,
        mae=float(mean_absolute_error(truth, predictions)),
        rmse=float(root_mean_squared_error(truth, predictions)),
        r2=float(r2_score(truth, predictions)),
    )


def _linear_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric = [column for column in BASE_NUMERIC_FEATURES if column in feature_columns]
    categorical = [column for column in BASE_CATEGORICAL_FEATURES if column in feature_columns]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def _tree_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric = [column for column in BASE_NUMERIC_FEATURES + POST_PUBLICATION_COLUMNS if column in feature_columns]
    categorical = [column for column in BASE_CATEGORICAL_FEATURES if column in feature_columns]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def build_model_registry(feature_columns: list[str]) -> dict[str, Pipeline]:
    return {
        "dummy_median": Pipeline(steps=[("model", DummyRegressor(strategy="median"))]),
        "ridge_ohe": Pipeline(
            steps=[
                ("preprocessor", _linear_preprocessor(feature_columns)),
                ("model", Ridge(alpha=3.0)),
            ]
        ),
        "random_forest_50k": Pipeline(
            steps=[
                ("preprocessor", _tree_preprocessor(feature_columns)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=20,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", _tree_preprocessor(feature_columns)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_depth=8,
                        learning_rate=0.08,
                        max_iter=300,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def run_validation_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[pd.DataFrame, str]:
    feature_columns = X_train.columns.tolist()
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    rows: list[ExperimentResult] = []
    for model_name, pipeline in build_model_registry(feature_columns).items():
        if model_name == "random_forest_50k" and len(X_train) > 50_000:
            sample_index = X_train.sample(n=50_000, random_state=SEED).index
            fit_X = X_train.loc[sample_index]
            fit_y = y_train_log.loc[sample_index]
        else:
            fit_X = X_train
            fit_y = y_train_log

        pipeline.fit(fit_X, fit_y)
        rows.append(_metrics(y_val_log, pipeline.predict(X_val), model_name, split="validation"))

    results = pd.DataFrame(asdict(row) for row in rows).sort_values(PRIMARY_METRIC).reset_index(drop=True)
    best_model_name = str(results.iloc[0]["model_name"])
    return results, best_model_name


def fit_best_model_and_evaluate(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, ExperimentResult]:
    feature_columns = X_train.columns.tolist()
    models = build_model_registry(feature_columns)
    pipeline = models[model_name]

    X_fit = pd.concat([X_train, X_val], axis=0)
    y_fit = pd.concat([y_train, y_val], axis=0)
    y_fit_log = np.log1p(y_fit)
    y_test_log = np.log1p(y_test)

    if model_name == "random_forest_50k" and len(X_fit) > 50_000:
        sample_index = X_fit.sample(n=50_000, random_state=SEED).index
        X_fit = X_fit.loc[sample_index]
        y_fit_log = y_fit_log.loc[sample_index]

    pipeline.fit(X_fit, y_fit_log)
    result = _metrics(y_test_log, pipeline.predict(X_test), model_name, split="test")
    return pipeline, result

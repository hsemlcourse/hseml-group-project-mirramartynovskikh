from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "price"
SEED = 42
LEAKAGE_COLUMNS = ["estimated_revenue_l365d"]
POST_PUBLICATION_COLUMNS = [
    "estimated_occupancy_l365d",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "number_of_reviews_ly",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]

BASE_RAW_COLUMNS = [
    TARGET_COLUMN,
    "city",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "host_response_time",
    "host_response_rate",
    "host_acceptance_rate",
    "host_is_superhost",
    "host_listings_count",
    "host_total_listings_count",
    "host_verifications",
    "host_has_profile_pic",
    "host_identity_verified",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "has_availability",
    "name",
    "description",
    "neighbourhood_cleansed",
    "host_since",
]

OPTIONAL_RAW_COLUMNS = POST_PUBLICATION_COLUMNS + [
    "estimated_occupancy_l365d",
]

BASE_NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "host_response_rate",
    "host_acceptance_rate",
    "host_listings_count",
    "host_total_listings_count",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "amenities_count",
    "host_verifications_count",
    "bathrooms_text_num",
    "name_word_count",
    "description_word_count",
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "has_availability",
    "host_tenure_days",
]

BASE_CATEGORICAL_FEATURES = [
    "city",
    "property_type",
    "room_type",
    "host_response_time",
    "neighbourhood_cleansed",
]


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def _parse_rate(value: object) -> float:
    if pd.isna(value):
        return np.nan
    cleaned = str(value).strip().replace("%", "")
    return float(cleaned) / 100 if cleaned else np.nan


def _parse_bool(value: object) -> float:
    if pd.isna(value):
        return np.nan
    normalized = str(value).strip().lower()
    if normalized in {"t", "true", "1", "yes"}:
        return 1.0
    if normalized in {"f", "false", "0", "no"}:
        return 0.0
    return np.nan


def _count_list_like_items(value: object) -> int:
    if pd.isna(value) or not str(value).strip():
        return 0
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return 0
    if isinstance(parsed, (list, tuple, set)):
        return len(parsed)
    return 0


def _parse_bathrooms_text(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan


def _ensure_available_columns(path: str | Path, requested: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    return [column for column in requested if column in header]


def load_raw_dataset(path: str | Path, include_post_publication_features: bool = False) -> pd.DataFrame:
    requested = list(BASE_RAW_COLUMNS)
    if include_post_publication_features:
        requested.extend(OPTIONAL_RAW_COLUMNS)
    available = _ensure_available_columns(path, requested)
    return pd.read_csv(path, usecols=available, low_memory=False)


def prepare_features(
    raw_df: pd.DataFrame,
    include_post_publication_features: bool = False,
) -> tuple[pd.DataFrame, pd.Series | None]:
    df = raw_df.copy()

    for column in LEAKAGE_COLUMNS:
        if column in df.columns:
            df = df.drop(columns=column)

    for column in ("host_response_rate", "host_acceptance_rate"):
        if column in df.columns:
            df[column] = df[column].map(_parse_rate)

    for column in (
        "host_is_superhost",
        "host_has_profile_pic",
        "host_identity_verified",
        "has_availability",
    ):
        if column in df.columns:
            df[column] = df[column].map(_parse_bool)

    df["amenities_count"] = df.get("amenities", pd.Series(dtype=object)).map(_count_list_like_items)
    df["host_verifications_count"] = df.get("host_verifications", pd.Series(dtype=object)).map(
        _count_list_like_items
    )
    df["bathrooms_text_num"] = df.get("bathrooms_text", pd.Series(dtype=object)).map(_parse_bathrooms_text)
    df["name_word_count"] = df.get("name", pd.Series(dtype=object)).fillna("").str.split().str.len()
    df["description_word_count"] = df.get("description", pd.Series(dtype=object)).fillna("").str.split().str.len()

    if "host_since" in df.columns:
        host_since = pd.to_datetime(df["host_since"], errors="coerce")
        reference_date = host_since.max()
        df["host_tenure_days"] = (reference_date - host_since).dt.days
    else:
        df["host_tenure_days"] = np.nan

    numeric_features = list(BASE_NUMERIC_FEATURES)
    categorical_features = list(BASE_CATEGORICAL_FEATURES)
    if include_post_publication_features:
        numeric_features.extend(column for column in POST_PUBLICATION_COLUMNS if column in df.columns)

    feature_columns = [column for column in numeric_features + categorical_features if column in df.columns]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, y


def split_dataset(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> DatasetSplit:
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
    validation_share = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_share,
        random_state=seed,
    )
    return DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

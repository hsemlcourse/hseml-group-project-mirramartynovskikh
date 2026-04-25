import pandas as pd

from src.preprocessing import LEAKAGE_COLUMNS, prepare_features, split_dataset


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [100.0, 200.0, 150.0, 90.0],
            "city": ["A", "B", "A", "B"],
            "latitude": [1.0, 2.0, 3.0, 4.0],
            "longitude": [10.0, 11.0, 12.0, 13.0],
            "property_type": ["Flat", "House", "Flat", "House"],
            "room_type": ["Entire home/apt", "Private room", "Entire home/apt", "Private room"],
            "accommodates": [2, 4, 3, 1],
            "bathrooms": [1.0, 1.5, 1.0, 1.0],
            "bathrooms_text": ["1 bath", "1.5 baths", "1 bath", "1 bath"],
            "bedrooms": [1.0, 2.0, 1.0, 1.0],
            "beds": [1.0, 2.0, 2.0, 1.0],
            "amenities": ['["Wifi", "Kitchen"]', '["Wifi"]', "[]", '["TV", "Heating", "Wifi"]'],
            "host_response_time": ["within an hour", "within a day", None, "within a few hours"],
            "host_response_rate": ["100%", "80%", None, "50%"],
            "host_acceptance_rate": ["90%", "70%", None, "40%"],
            "host_is_superhost": ["t", "f", None, "t"],
            "host_listings_count": [1, 2, 1, 3],
            "host_total_listings_count": [1, 2, 1, 3],
            "host_verifications": ['["email", "phone"]', '["email"]', "[]", '["email", "phone", "work_email"]'],
            "host_has_profile_pic": ["t", "t", "f", "t"],
            "host_identity_verified": ["t", "f", "f", "t"],
            "availability_30": [5, 10, 3, 20],
            "availability_60": [10, 20, 6, 40],
            "availability_90": [15, 30, 9, 60],
            "availability_365": [100, 200, 150, 300],
            "has_availability": ["t", "t", "f", "t"],
            "name": ["Cozy flat", "Big house", "Studio downtown", None],
            "description": ["Nice and quiet", "Spacious place", "", None],
            "neighbourhood_cleansed": ["Center", "North", "Center", "South"],
            "host_since": ["2020-01-01", "2021-05-01", "2022-01-01", None],
            "estimated_revenue_l365d": [10_000, 15_000, 9_000, 8_000],
        }
    )


def test_prepare_features_builds_engineered_columns_and_removes_leakage() -> None:
    X, y = prepare_features(_sample_frame(), include_post_publication_features=False)

    assert y is not None
    assert "amenities_count" in X.columns
    assert "host_verifications_count" in X.columns
    assert "bathrooms_text_num" in X.columns
    assert "host_tenure_days" in X.columns
    for leakage_column in LEAKAGE_COLUMNS:
        assert leakage_column not in X.columns


def test_prepare_features_parses_boolean_and_percentage_fields() -> None:
    X, _ = prepare_features(_sample_frame(), include_post_publication_features=False)

    assert X.loc[0, "host_response_rate"] == 1.0
    assert X.loc[1, "host_acceptance_rate"] == 0.7
    assert X.loc[0, "host_is_superhost"] == 1.0
    assert X.loc[1, "host_is_superhost"] == 0.0


def test_split_dataset_produces_non_empty_partitions() -> None:
    sample = pd.concat([_sample_frame()] * 15, ignore_index=True)
    X, y = prepare_features(sample, include_post_publication_features=False)
    split = split_dataset(X, y, seed=42)

    assert len(split.X_train) > len(split.X_val) > 0
    assert len(split.X_test) > 0
    assert len(split.X_train) + len(split.X_val) + len(split.X_test) == len(X)

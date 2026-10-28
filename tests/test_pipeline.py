import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.train import (
    extract_features,
    build_preprocessor,
    build_pipeline,
    NUM_COLS,
    CAT_COLS
)


@pytest.fixture
def sample_raw_dataframe():
    """Create a sample raw dataframe mimicking 'Car details v3.csv' rows."""
    return pd.DataFrame([
        {
            'name': 'Maruti Swift Dzire VDI',
            'year': 2014,
            'selling_price': 450000,
            'km_driven': 145500,
            'fuel': 'Diesel',
            'seller_type': 'Individual',
            'transmission': 'Manual',
            'owner': 'First Owner',
            'mileage': '23.4 kmpl',
            'engine': '1248 CC',
            'max_power': '74 bhp',
            'torque': '190Nm@ 2000rpm',
            'seats': 5.0
        },
        {
            'name': 'Honda City 2017-2020 EXi',
            'year': 2016,
            'selling_price': 550000,
            'km_driven': 60000,
            'fuel': 'Petrol',
            'seller_type': 'Dealer',
            'transmission': 'Automatic',
            'owner': 'Second Owner',
            'mileage': '18.0 kmpl',
            'engine': '1497 CC',
            'max_power': '117.3 bhp',
            'torque': '145Nm@ 4600rpm',
            'seats': 5.0
        }
    ])


@pytest.fixture
def sample_processed_car():
    """Create a clean preprocessed single sample car record for inference."""
    return pd.DataFrame([{
        'km_driven': 45000,
        'fuel': 'Diesel',
        'seller_type': 'Individual',
        'transmission': 'Manual',
        'owner': 'First Owner',
        'mileage_num': 22.5,
        'engine_num': 1248.0,
        'max_power_num': 88.5,
        'seats': 5.0,
        'brand': 'Maruti',
        'car_age': 5
    }])


def test_feature_extraction(sample_raw_dataframe):
    """Test that raw string attributes are cleanly parsed into numeric features and brand."""
    cleaned = extract_features(sample_raw_dataframe)

    # Check extracted columns exist
    assert 'brand' in cleaned.columns
    assert 'mileage_num' in cleaned.columns
    assert 'engine_num' in cleaned.columns
    assert 'max_power_num' in cleaned.columns
    assert 'car_age' in cleaned.columns

    # Check unneeded raw columns are dropped
    assert 'name' not in cleaned.columns
    assert 'mileage' not in cleaned.columns
    assert 'engine' not in cleaned.columns
    assert 'max_power' not in cleaned.columns
    assert 'torque' not in cleaned.columns

    # Check values
    assert cleaned.loc[0, 'brand'] == 'Maruti'
    assert cleaned.loc[0, 'mileage_num'] == 23.4
    assert cleaned.loc[0, 'engine_num'] == 1248.0
    assert cleaned.loc[0, 'max_power_num'] == 74.0
    assert cleaned.loc[0, 'car_age'] == 2026 - 2014


def test_build_preprocessor(sample_raw_dataframe):
    """Test that ColumnTransformer processes numeric and categorical columns correctly."""
    cleaned = extract_features(sample_raw_dataframe)
    X = cleaned.drop(columns=['selling_price'])
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)
    assert transformed is not None
    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= len(NUM_COLS)


def test_pipeline_construction_and_fit(sample_raw_dataframe):
    """Test that end-to-end pipeline fits without errors and outputs positive prices."""
    cleaned = extract_features(sample_raw_dataframe)
    X = cleaned.drop(columns=['selling_price'])
    y = np.log1p(cleaned['selling_price'])

    pipeline = build_pipeline({'n_estimators': 10, 'random_state': 42})
    pipeline.fit(X, y)

    preds_log = pipeline.predict(X)
    preds_act = np.expm1(preds_log)

    assert len(preds_act) == 2
    assert all(p > 0 for p in preds_act)


def test_model_inference_on_sample_car(sample_processed_car):
    """Test model inference on single vehicle input."""
    model_paths = [Path("car_price_model.pkl"), Path("src/../car_price_model.pkl")]
    model = None
    for p in model_paths:
        if p.exists():
            model = joblib.load(p)
            break

    if model is not None:
        pred_log = model.predict(sample_processed_car)
        pred_price = np.expm1(pred_log)[0]
        assert pred_price > 10000, f"Predicted price should be reasonable, got: {pred_price}"


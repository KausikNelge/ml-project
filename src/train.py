import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

NUM_COLS = ['km_driven', 'mileage_num', 'engine_num', 'max_power_num', 'seats', 'car_age']
CAT_COLS = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']


def find_data_path():
    """Locate Car details v3.csv from current directory or project root."""
    possible_paths = [
        Path("Car details v3.csv"),
        Path(__file__).resolve().parent.parent / "Car details v3.csv",
        Path(__file__).resolve().parent / "Car details v3.csv",
    ]
    for p in possible_paths:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find 'Car details v3.csv'.")


def load_data(filepath=None):
    """Load raw dataset from CSV file."""
    if filepath is None:
        filepath = find_data_path()
    return pd.read_csv(filepath)


def extract_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numerical and categorical features from raw car records:
    - Extracts brand from name
    - Parses numerical mileage, engine CC, and max power bhp
    - Calculates vehicle age
    """
    df = df_input.copy()
    if 'name' in df.columns and 'brand' not in df.columns:
        df['brand'] = df['name'].apply(lambda x: str(x).split()[0])
    if 'mileage' in df.columns and 'mileage_num' not in df.columns:
        df['mileage_num'] = df['mileage'].astype(str).str.extract(r'([\d.]+)').astype(float)
    if 'engine' in df.columns and 'engine_num' not in df.columns:
        df['engine_num'] = df['engine'].astype(str).str.extract(r'([\d.]+)').astype(float)
    if 'max_power' in df.columns and 'max_power_num' not in df.columns:
        df['max_power_num'] = pd.to_numeric(df['max_power'].astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')
    if 'year' in df.columns and 'car_age' not in df.columns:
        current_year = 2026
        df['car_age'] = current_year - df['year']

    # Keep necessary columns
    drop_candidates = ['name', 'year', 'mileage', 'engine', 'max_power', 'torque']
    cols_to_drop = [c for c in drop_candidates if c in df.columns]
    return df.drop(columns=cols_to_drop)


def build_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer for numeric and categorical pipelines."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, NUM_COLS),
        ('cat', cat_pipeline, CAT_COLS)
    ])
    return preprocessor


def build_pipeline(rf_params=None) -> Pipeline:
    """Build full end-to-end model pipeline with preprocessor and RandomForestRegressor."""
    if rf_params is None:
        rf_params = {'n_estimators': 100, 'random_state': 42}
    preprocessor = build_preprocessor()
    model = RandomForestRegressor(**rf_params)
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])


def train_and_tune(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 3) -> tuple[Pipeline, dict]:
    """
    Tune RandomForestRegressor using GridSearchCV with cross-validation.
    Returns (best_estimator, best_params).
    """
    base_pipeline = build_pipeline()

    param_grid = {
        'model__n_estimators': [100, 150],
        'model__max_depth': [None, 20],
        'model__min_samples_split': [2, 5]
    }

    print("Initiating GridSearchCV tuning for Random Forest Regressor...")
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    print(f"GridSearchCV Best Parameters: {grid_search.best_params_}")
    print(f"GridSearchCV Best CV R2 Score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate pipeline on log-transformed target and report actual scale metrics."""
    y_pred_log = pipeline.predict(X_test)
    y_test_act = np.expm1(y_test)
    y_pred_act = np.expm1(y_pred_log)

    r2 = r2_score(y_test_act, y_pred_act)
    mae = mean_absolute_error(y_test_act, y_pred_act)

    print(f"Tuned Random Forest Test R2 Score: {r2:.4f}")
    print(f"Tuned Random Forest Test MAE:      INR {mae:,.2f}")
    return {'r2': r2, 'mae': mae}


def main():
    print("=" * 60)
    print("Used Car Price Prediction: Model Training & Tuning Pipeline")
    print("=" * 60)

    # 1. Load data
    data_path = find_data_path()
    print(f"Loading dataset from: {data_path}")
    raw_df = load_data(data_path)
    print(f"Loaded {len(raw_df)} records.")

    # 2. Extract features
    cleaned_df = extract_features(raw_df)

    # 3. Separate features and target
    X = cleaned_df.drop(columns=['selling_price'])
    y = np.log1p(cleaned_df['selling_price'])

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")

    # 5. Tune and train model with GridSearchCV
    best_pipeline, best_params = train_and_tune(X_train, y_train)

    # 6. Evaluate model
    print("\nEvaluating model on test dataset...")
    metrics = evaluate_model(best_pipeline, X_test, y_test)

    # 7. Save model artifact
    save_destinations = [
        Path("car_price_model.pkl"),
        Path(__file__).resolve().parent.parent / "car_price_model.pkl"
    ]
    saved_paths = set()
    for dest in save_destinations:
        resolved = dest.resolve()
        if resolved not in saved_paths:
            joblib.dump(best_pipeline, resolved)
            print(f"Saved trained model pipeline to: {resolved}")
            saved_paths.add(resolved)

    print("Model training, hyperparameter tuning, and persistence completed successfully!")
    return best_pipeline, metrics


if __name__ == '__main__':
    main()


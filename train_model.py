import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv('Car details v3.csv')

# Preprocessing & Feature Engineering
df['brand'] = df['name'].apply(lambda x: str(x).split()[0])
df['mileage_num'] = df['mileage'].str.extract(r'([\d.]+)').astype(float)
df['engine_num'] = df['engine'].str.extract(r'([\d.]+)').astype(float)
df['max_power_num'] = pd.to_numeric(df['max_power'].str.extract(r'([\d.]+)')[0], errors='coerce')
df['car_age'] = 2026 - df['year']

df_clean = df.drop(columns=['name', 'year', 'mileage', 'engine', 'max_power', 'torque'])

# Define X and y (Log-transformed target variable)
X = df_clean.drop(columns=['selling_price'])
y = np.log1p(df_clean['selling_price'])

num_cols = ['km_driven', 'mileage_num', 'engine_num', 'max_power_num', 'seats', 'car_age']
cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']

# Pipelines for numeric and categorical handling
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Full Model Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit Model
model_pipeline.fit(X_train, y_train)

# Evaluate Performance
y_pred_log = model_pipeline.predict(X_test)
y_test_act = np.expm1(y_test)
y_pred_act = np.expm1(y_pred_log)

print(f"Random Forest R² Score: {r2_score(y_test_act, y_pred_act):.4f}")
print(f"Random Forest MAE:      ₹{mean_absolute_error(y_test_act, y_pred_act):,.2f}")

# Save Model Artifact
joblib.dump(model_pipeline, 'car_price_model.pkl')
print("Trained model saved to 'car_price_model.pkl'.")
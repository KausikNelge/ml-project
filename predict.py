import pandas as pd
import numpy as np
import joblib

# Load saved pipeline model
model = joblib.load('car_price_model.pkl')

# Sample vehicle for price evaluation
sample_car = pd.DataFrame([{
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

# Make Prediction
pred_log = model.predict(sample_car)
pred_price = np.expm1(pred_log)[0]

print(f"Estimated Used Car Value: ₹{pred_price:,.2f}")
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="AutoValuate | Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .metric-card h2 {
        color: white !important;
        font-size: 2.3rem;
        margin: 0.3rem 0;
    }
    .metric-card p {
        color: #E0E7FF;
        margin: 0;
        font-size: 0.95rem;
    }
    .info-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        background-color: #EEF2FF;
        color: #3730A3;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_pipeline():
    """Load the trained machine learning pipeline."""
    possible_paths = [
        Path("car_price_model.pkl"),
        Path(__file__).resolve().parent.parent / "car_price_model.pkl",
        Path(__file__).resolve().parent / "car_price_model.pkl",
    ]
    for p in possible_paths:
        if p.exists():
            try:
                model = joblib.load(p)
                return model, str(p)
            except Exception as e:
                st.error(f"Error loading model from {p}: {e}")
    return None, None


# Presets for quick testing
PRESETS = {
    "Custom Vehicle": {
        "brand": "Maruti", "year": 2018, "km_driven": 45000, "fuel": "Diesel",
        "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner",
        "mileage": 22.5, "engine": 1248.0, "max_power": 88.5, "seats": 5.0
    },
    "Budget Hatchback (Maruti Swift Dzire)": {
        "brand": "Maruti", "year": 2016, "km_driven": 65000, "fuel": "Diesel",
        "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner",
        "mileage": 24.3, "engine": 1248.0, "max_power": 74.0, "seats": 5.0
    },
    "Executive Sedan (Honda City Petrol)": {
        "brand": "Honda", "year": 2019, "km_driven": 35000, "fuel": "Petrol",
        "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner",
        "mileage": 18.0, "engine": 1497.0, "max_power": 117.3, "seats": 5.0
    },
    "Compact SUV (Renault Duster)": {
        "brand": "Renault", "year": 2015, "km_driven": 80000, "fuel": "Diesel",
        "seller_type": "Individual", "transmission": "Manual", "owner": "Second Owner",
        "mileage": 19.8, "engine": 1461.0, "max_power": 108.5, "seats": 5.0
    },
    "Luxury Sedan (BMW 3 Series)": {
        "brand": "BMW", "year": 2017, "km_driven": 40000, "fuel": "Diesel",
        "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner",
        "mileage": 19.6, "engine": 1995.0, "max_power": 190.0, "seats": 5.0
    }
}

ALL_BRANDS = [
    'Ambassador', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun',
    'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar',
    'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti',
    'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot',
    'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'
]


def format_inr(number: float) -> str:
    """Format currency nicely with Indian numbering style and Lakhs."""
    lakhs = number / 100000.0
    if lakhs >= 100.0:
        crores = lakhs / 100.0
        return f"₹{crores:.2f} Crore (₹{int(number):,})"
    elif lakhs >= 1.0:
        return f"₹{lakhs:.2f} Lakh (₹{int(number):,})"
    else:
        return f"₹{int(number):,}"


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/isometric/512/car.png", width=100)
    st.sidebar.title("Configuration")
    st.sidebar.markdown("Choose a vehicle preset or customize details manually.")

    selected_preset = st.sidebar.selectbox("⚡ Quick Presets", list(PRESETS.keys()))
    preset_data = PRESETS[selected_preset]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About Model")
    st.sidebar.markdown("""
    - **Model**: Tuned Random Forest Regressor
    - **Hyperparameter Optimization**: 3-Fold GridSearchCV
    - **Target**: Log-transformed Selling Price (`log1p`)
    - **Pipeline**: Automated Imputation, Standard Scaling & One-Hot Encoding
    """)

    # Main Header
    st.markdown('<div class="main-header">🚗 AutoValuate: Used Car Price Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Accurately predict used car market valuations based on vehicle specs, age, and condition.</div>',
        unsafe_allow_html=True
    )

    # Load Model Pipeline
    model, model_path = load_trained_pipeline()
    if model is None:
        st.warning("⚠️ Trained model artifact (`car_price_model.pkl`) not found. Please train the model first by running `python src/train.py`.")

    # Input Form
    with st.container():
        st.markdown("### 📋 Enter Vehicle Specifications")

        col1, col2, col3 = st.columns(3)

        with col1:
            brand_idx = ALL_BRANDS.index(preset_data["brand"]) if preset_data["brand"] in ALL_BRANDS else 0
            brand = st.selectbox("Brand / Make", ALL_BRANDS, index=brand_idx)

            year = st.slider("Manufacturing Year", min_value=1995, max_value=2026, value=preset_data["year"], step=1)
            car_age = 2026 - year
            st.caption(f"Estimated Vehicle Age: **{car_age} years**")

            km_driven = st.number_input(
                "Kilometers Driven (km)",
                min_value=100,
                max_value=1000000,
                value=preset_data["km_driven"],
                step=1000
            )

            fuel = st.selectbox(
                "Fuel Type",
                ['Petrol', 'Diesel', 'CNG', 'LPG'],
                index=['Petrol', 'Diesel', 'CNG', 'LPG'].index(preset_data["fuel"])
            )

        with col2:
            transmission = st.selectbox(
                "Transmission Type",
                ['Manual', 'Automatic'],
                index=['Manual', 'Automatic'].index(preset_data["transmission"])
            )

            owner = st.selectbox(
                "Ownership History",
                ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
                index=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'].index(preset_data["owner"])
            )

            seller_type = st.selectbox(
                "Seller Type",
                ['Individual', 'Dealer', 'Trustmark Dealer'],
                index=['Individual', 'Dealer', 'Trustmark Dealer'].index(preset_data["seller_type"])
            )

            seats = st.selectbox(
                "Seating Capacity",
                [2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                index=[2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].index(preset_data["seats"])
            )

        with col3:
            mileage_num = st.number_input(
                "Mileage (kmpl or km/kg)",
                min_value=5.0,
                max_value=45.0,
                value=float(preset_data["mileage"]),
                step=0.5
            )

            engine_num = st.number_input(
                "Engine Displacement (CC)",
                min_value=600.0,
                max_value=5000.0,
                value=float(preset_data["engine"]),
                step=50.0
            )

            max_power_num = st.number_input(
                "Max Power (bhp)",
                min_value=30.0,
                max_value=500.0,
                value=float(preset_data["max_power"]),
                step=5.0
            )

    st.markdown("---")

    # Predict Button
    predict_clicked = st.button("🔍 Calculate Estimated Resale Value", type="primary", use_container_width=True)

    if predict_clicked:
        if model is None:
            st.error("Cannot compute valuation: Model artifact not found. Run `python src/train.py` first.")
            return

        input_data = pd.DataFrame([{
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage_num': mileage_num,
            'engine_num': engine_num,
            'max_power_num': max_power_num,
            'seats': seats,
            'brand': brand,
            'car_age': car_age
        }])

        with st.spinner("Analyzing market trends and predicting price..."):
            try:
                pred_log = model.predict(input_data)
                pred_price = float(np.expm1(pred_log)[0])

                # Confidence Range estimation (~10% bound)
                low_bound = max(pred_price * 0.90, 15000.0)
                high_bound = pred_price * 1.10

                st.markdown(f"""
                <div class="metric-card">
                    <p>ESTIMATED FAIR MARKET VALUE</p>
                    <h2>{format_inr(pred_price)}</h2>
                    <p>Estimated Range: <b>{format_inr(low_bound)}</b> – <b>{format_inr(high_bound)}</b></p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 🔍 Vehicle Evaluation Summary")
                st.markdown(f"""
                <span class="info-badge">🚘 {brand}</span>
                <span class="info-badge">📅 {year} ({car_age} yrs old)</span>
                <span class="info-badge">🛣️ {km_driven:,} km</span>
                <span class="info-badge">⛽ {fuel}</span>
                <span class="info-badge">⚙️ {transmission}</span>
                <span class="info-badge">👤 {owner}</span>
                <span class="info-badge">⚡ {max_power_num} bhp</span>
                <span class="info-badge">🏎️ {engine_num:.0f} CC</span>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error occurred during price prediction: {e}")

    # Visualizations / Exploratory Insights Tab
    st.markdown("---")
    with st.expander("📊 View Exploratory Data Analysis (EDA) & Market Insights"):
        eda_img_path = Path("eda_visualizations.png")
        if not eda_img_path.exists():
            eda_img_path = Path(__file__).resolve().parent.parent / "eda_visualizations.png"

        if eda_img_path.exists():
            st.image(str(eda_img_path), caption="Exploratory Data Analysis Dashboard", use_container_width=True)
        else:
            st.info("EDA chart not found. Run `python generate_eda.py` to generate the dashboard visualization.")


if __name__ == '__main__':
    main()


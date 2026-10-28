import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="AutoValuate | AI Used Car Valuation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fixora Modern SaaS Dark Theme CSS with Atmospheric Glow & Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Newsreader:ital,wght@1,400;1,500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* High contrast dark canvas */
    .stApp {
        background-color: #020617 !important;
        color: #f1f5f9 !important;
        overflow-x: hidden;
    }

    /* Ambient background glow */
    .ambient-glow {
        position: fixed;
        top: 20%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 650px;
        height: 320px;
        background: radial-gradient(circle, rgba(147, 51, 234, 0.22) 0%, rgba(79, 70, 229, 0.12) 60%, transparent 80%);
        filter: blur(100px);
        border-radius: 9999px;
        pointer-events: none;
        z-index: 0;
    }

    /* Secondary ambient glow */
    .ambient-glow-2 {
        position: fixed;
        bottom: 10%;
        right: 15%;
        width: 450px;
        height: 250px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.14) 0%, transparent 70%);
        filter: blur(120px);
        border-radius: 9999px;
        pointer-events: none;
        z-index: 0;
    }

    header[data-testid="stHeader"] {
        background-color: rgba(2, 6, 23, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(30, 41, 59, 0.5) !important;
    }

    /* Sidebar dark style */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b !important;
    }

    /* Navbar */
    .fixora-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0 1.75rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(30, 41, 59, 0.6);
        position: relative;
        z-index: 10;
    }
    .fixora-logo-wrap {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .fixora-logo-badge {
        width: 38px;
        height: 38px;
        border-radius: 12px;
        background: linear-gradient(135deg, #9333ea 0%, #6366f1 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        color: white;
        font-size: 1.1rem;
        box-shadow: 0 10px 15px -3px rgba(147, 51, 234, 0.4);
    }
    .fixora-logo-text {
        font-weight: 700;
        font-size: 1.25rem;
        letter-spacing: -0.025em;
        color: #f8fafc;
    }
    .fixora-nav-links {
        display: flex;
        align-items: center;
        gap: 1.75rem;
        font-size: 0.875rem;
        color: #94a3b8;
    }
    .fixora-nav-links span:hover {
        color: #ffffff;
        cursor: pointer;
    }
    .fixora-badge-live {
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        background: rgba(147, 51, 234, 0.15);
        border: 1px solid rgba(147, 51, 234, 0.4);
        color: #c084fc;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    .fixora-badge-dot {
        width: 6px;
        height: 6px;
        border-radius: 9999px;
        background-color: #a855f7;
        box-shadow: 0 0 8px #a855f7;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        max-width: 850px;
        margin: 0 auto 2.5rem auto;
        position: relative;
        z-index: 10;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.15;
        color: #ffffff;
        margin-bottom: 1.2rem;
    }
    .hero-title .italic-accent {
        font-family: 'Newsreader', serif;
        font-style: italic;
        font-weight: 400;
        color: #c084fc;
        text-shadow: 0 0 25px rgba(192, 132, 252, 0.3);
    }
    .hero-subtitle {
        font-size: 1.125rem;
        color: #94a3b8;
        line-height: 1.6;
        max-width: 640px;
        margin: 0 auto;
    }

    /* Product Preview Mockup Card */
    .mockup-card {
        border-radius: 1.25rem;
        border: 1px solid rgba(30, 41, 59, 0.9);
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        margin-bottom: 2.5rem;
        position: relative;
        z-index: 10;
    }
    .mockup-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1rem;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid rgba(30, 41, 59, 0.7);
    }
    .mockup-title-area {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .mockup-icon {
        width: 26px;
        height: 26px;
        border-radius: 8px;
        background: #9333ea;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
    }
    .mockup-dots {
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .mockup-dot {
        width: 10px;
        height: 10px;
        border-radius: 9999px;
        background-color: #334155;
    }

    /* Stat Cards */
    .stat-card {
        background: rgba(2, 6, 23, 0.65);
        border: 1px solid rgba(30, 41, 59, 0.8);
        border-radius: 0.85rem;
        padding: 1.1rem;
        text-align: left;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-bottom: 0.25rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-val-white {
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .stat-val-purple {
        font-size: 1.75rem;
        font-weight: 800;
        color: #c084fc;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .stat-val-emerald {
        font-size: 1.75rem;
        font-weight: 800;
        color: #34d399;
        letter-spacing: -0.02em;
        line-height: 1;
    }

    /* Form Container */
    .form-panel {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(30, 41, 59, 0.85);
        border-radius: 1.25rem;
        backdrop-filter: blur(16px);
        padding: 1.75rem;
        margin-bottom: 2rem;
        position: relative;
        z-index: 10;
    }
    .panel-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Streamlit Input Overrides */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: rgba(2, 6, 23, 0.75) !important;
        border: 1px solid #334155 !important;
        border-radius: 0.75rem !important;
        color: #f8fafc !important;
        transition: all 0.2s ease;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: #9333ea !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input {
        color: #f8fafc !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-color: #334155 !important;
    }
    label[data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #9333ea 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.75rem 2.25rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 10px 25px -5px rgba(147, 51, 234, 0.4) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 20px 30px -8px rgba(147, 51, 234, 0.55) !important;
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%) !important;
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.99) !important;
    }

    /* Result Metric Card */
    .valuation-result-card {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.9) 0%, rgba(15, 23, 42, 0.95) 50%, rgba(17, 24, 39, 0.9) 100%);
        border: 1px solid rgba(168, 85, 247, 0.4);
        border-radius: 1.25rem;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0 2rem 0;
        box-shadow: 0 20px 40px -10px rgba(147, 51, 234, 0.25), 0 0 0 1px rgba(147, 51, 234, 0.2);
        position: relative;
        overflow: hidden;
    }
    .valuation-result-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        height: 1px;
        background: linear-gradient(90deg, transparent, #c084fc, transparent);
    }
    .result-kicker {
        color: #c084fc;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .result-price {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        margin: 0.25rem 0 0.5rem 0;
        text-shadow: 0 0 30px rgba(192, 132, 252, 0.3);
    }
    .result-range {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.25rem;
    }
    .result-range b {
        color: #e2e8f0;
    }

    /* Badges */
    .spec-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.8rem;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #e2e8f0;
        margin: 0.2rem;
    }

    /* Chart Simulation Area */
    .chart-box {
        background: rgba(2, 6, 23, 0.4);
        border: 1px solid rgba(30, 41, 59, 0.6);
        border-radius: 0.75rem;
        padding: 1.25rem;
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        height: 130px;
        margin-top: 1rem;
    }
    .chart-bar-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.4rem;
        flex: 1;
    }
    .chart-bar {
        width: 38px;
        border-radius: 6px 6px 0 0;
        transition: all 0.3s ease;
    }
    .chart-bar:hover {
        filter: brightness(1.2);
    }
    .chart-bar-label {
        font-size: 0.7rem;
        color: #64748b;
    }
</style>

<!-- Atmospheric background glow overlays -->
<div class="ambient-glow"></div>
<div class="ambient-glow-2"></div>
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


# Presets for quick evaluation
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
    # 1. Navbar Header
    st.markdown("""
    <div class="fixora-navbar">
        <div class="fixora-logo-wrap">
            <div class="fixora-logo-badge">F</div>
            <span class="fixora-logo-text">Fixora <span style="font-size:0.85rem; color:#94a3b8; font-weight:400;">/ AutoValuate</span></span>
        </div>
        <div class="fixora-nav-links">
            <span class="fixora-badge-live">
                <span class="fixora-badge-dot"></span>
                Tuned RF v2.0 Live
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Hero Section (Replicating Fixora Hero)
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">
            Elevate Your <span class="italic-accent">Vehicle</span> Valuation with Fixora
        </h1>
        <p class="hero-subtitle">
            Streamline, Optimize, and Scale Your Used Car Price Estimation with Our High-Precision Random Forest Machine Learning Pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Product Preview Mockup (Matching SaaS Dashboard Mockup)
    st.markdown("""
    <div class="mockup-card">
        <div class="mockup-header">
            <div class="mockup-title-area">
                <div class="mockup-icon">F</div>
                <span style="font-size:0.875rem; font-weight:500; color:#e2e8f0;">Fixora Valuation Intelligence</span>
            </div>
            <div class="mockup-dots">
                <div class="mockup-dot"></div>
                <div class="mockup-dot"></div>
                <div class="mockup-dot"></div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.25rem;">
            <div class="stat-card">
                <div class="stat-label">Trained Vehicles</div>
                <div class="stat-val-white">8,128</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Model Accuracy (R²)</div>
                <div class="stat-val-purple">97.23%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Abs Error (MAE)</div>
                <div class="stat-val-emerald">±₹66.3k</div>
            </div>
        </div>
        <div class="chart-box">
            <div class="chart-bar-wrap">
                <div class="chart-bar" style="height: 40px; background: rgba(147, 51, 234, 0.35);"></div>
                <span class="chart-bar-label">&lt;2 yrs</span>
            </div>
            <div class="chart-bar-wrap">
                <div class="chart-bar" style="height: 70px; background: rgba(147, 51, 234, 0.55);"></div>
                <span class="chart-bar-label">4 yrs</span>
            </div>
            <div class="chart-bar-wrap">
                <div class="chart-bar" style="height: 55px; background: rgba(147, 51, 234, 0.45);"></div>
                <span class="chart-bar-label">6 yrs</span>
            </div>
            <div class="chart-bar-wrap">
                <div class="chart-bar" style="height: 95px; background: #9333ea; box-shadow: 0 0 15px rgba(147, 51, 234, 0.6);"></div>
                <span class="chart-bar-label">Optimal</span>
            </div>
            <div class="chart-bar-wrap">
                <div class="chart-bar" style="height: 75px; background: rgba(147, 51, 234, 0.7);"></div>
                <span class="chart-bar-label">8 yrs</span>
            </div>
            <div class="chart-bar-wrap">
                <div class="chart-bar" style="height: 105px; background: rgba(147, 51, 234, 0.9);"></div>
                <span class="chart-bar-label">10+ yrs</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Model Pipeline
    model, model_path = load_trained_pipeline()

    # Preset Selector in a clean row
    preset_col1, preset_col2 = st.columns([1, 2])
    with preset_col1:
        st.markdown("<p style='color:#94a3b8; font-size:0.875rem; font-weight:600; margin-bottom:0.25rem;'>⚡ Quick Preset Profiles</p>", unsafe_allow_html=True)
        selected_preset = st.selectbox(
            "Quick Preset Profiles",
            list(PRESETS.keys()),
            label_visibility="collapsed"
        )
        preset_data = PRESETS[selected_preset]

    with preset_col2:
        st.markdown(
            f"<p style='color:#64748b; font-size:0.85rem; padding-top:1.4rem;'>Selected preset config: <b style='color:#c084fc;'>{preset_data['brand']}</b> ({preset_data['year']}) · {preset_data['km_driven']:,} km · {preset_data['fuel']}</p>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 4. Valuation Input Form
    with st.container():
        st.markdown('<div class="form-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🚘 Vehicle Profile & Powertrain Specifications</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            brand_idx = ALL_BRANDS.index(preset_data["brand"]) if preset_data["brand"] in ALL_BRANDS else 0
            brand = st.selectbox("Vehicle Brand / Make", ALL_BRANDS, index=brand_idx)

            year = st.slider("Manufacturing Year", min_value=1995, max_value=2026, value=preset_data["year"], step=1)
            car_age = 2026 - year
            st.markdown(f"<span style='color:#a855f7; font-size:0.8rem; font-weight:600;'>Estimated Age: {car_age} years</span>", unsafe_allow_html=True)

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
                "Ownership Record",
                ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
                index=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'].index(preset_data["owner"])
            )

            seller_type = st.selectbox(
                "Seller Category",
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
                "Fuel Economy Mileage (kmpl)",
                min_value=5.0,
                max_value=45.0,
                value=float(preset_data["mileage"]),
                step=0.5
            )

            engine_num = st.number_input(
                "Displacement Engine (CC)",
                min_value=600.0,
                max_value=5000.0,
                value=float(preset_data["engine"]),
                step=50.0
            )

            max_power_num = st.number_input(
                "Max Engine Power (bhp)",
                min_value=30.0,
                max_value=500.0,
                value=float(preset_data["max_power"]),
                step=5.0
            )

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Predict Button
        predict_clicked = st.button("Calculate Market Valuation", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. Valuation Output
    if predict_clicked:
        if model is None:
            st.error("Model artifact not found. Please run python src/train.py first.")
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

        with st.spinner("Analyzing valuation model and market benchmarks..."):
            try:
                pred_log = model.predict(input_data)
                pred_price = float(np.expm1(pred_log)[0])

                low_bound = max(pred_price * 0.90, 15000.0)
                high_bound = pred_price * 1.10

                st.markdown(f"""
                <div class="valuation-result-card">
                    <div class="result-kicker">Estimated Fair Market Valuation</div>
                    <div class="result-price">{format_inr(pred_price)}</div>
                    <div class="result-range">
                        Projected Range: <b>{format_inr(low_bound)}</b> – <b>{format_inr(high_bound)}</b>
                    </div>
                    <div>
                        <span class="spec-pill">🚘 {brand}</span>
                        <span class="spec-pill">📅 {year} ({car_age} yrs old)</span>
                        <span class="spec-pill">🛣️ {km_driven:,} km</span>
                        <span class="spec-pill">⛽ {fuel}</span>
                        <span class="spec-pill">⚙️ {transmission}</span>
                        <span class="spec-pill">👤 {owner}</span>
                        <span class="spec-pill">⚡ {max_power_num} bhp</span>
                        <span class="spec-pill">🏎️ {engine_num:.0f} CC</span>
                        <span class="spec-pill">💺 {seats:.0f} Seats</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # 6. EDA Visualizations Expandable Section
    with st.expander("📊 View Exploratory Data Insights & Distribution Analysis"):
        eda_img_path = Path("eda_visualizations.png")
        if not eda_img_path.exists():
            eda_img_path = Path(__file__).resolve().parent.parent / "eda_visualizations.png"

        if eda_img_path.exists():
            st.image(str(eda_img_path), caption="Fixora Fleet EDA Analysis Dashboard", use_container_width=True)
        else:
            st.info("EDA chart not found. Run python generate_eda.py to generate visual metrics.")


if __name__ == '__main__':
    main()

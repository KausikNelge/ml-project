import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Fixora - Fleet Management & Valuation Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixora Dark Theme CSS with Atmospheric Glow & Sidebar Styling
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

    /* Fixora Sidebar Brand Header */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 1.25rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(30, 41, 59, 0.8);
    }
    .sidebar-logo {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #9333ea 0%, #6366f1 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        color: white;
        font-size: 1.1rem;
        box-shadow: 0 8px 16px -4px rgba(147, 51, 234, 0.5);
    }
    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .sidebar-sub {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #c084fc;
        letter-spacing: 0.08em;
    }

    /* Metric Cards */
    .kpi-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(30, 41, 59, 0.85);
        border-radius: 1rem;
        padding: 1.25rem;
        backdrop-filter: blur(12px);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }
    .kpi-val {
        font-size: 2rem;
        font-weight: 800;
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

    /* Result Metric Card */
    .valuation-result-card {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.9) 0%, rgba(15, 23, 42, 0.95) 50%, rgba(17, 24, 39, 0.9) 100%);
        border: 1px solid rgba(168, 85, 247, 0.4);
        border-radius: 1.25rem;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0 2rem 0;
        box-shadow: 0 20px 40px -10px rgba(147, 51, 234, 0.25);
        position: relative;
        overflow: hidden;
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
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        margin: 0.25rem 0 0.5rem 0;
        text-shadow: 0 0 30px rgba(192, 132, 252, 0.3);
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
</style>
<div class="ambient-glow"></div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_pipeline():
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


@st.cache_data
def load_sample_dataset():
    possible_paths = [
        Path("Car details v3.csv"),
        Path(__file__).resolve().parent.parent / "Car details v3.csv",
    ]
    for p in possible_paths:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None


PRESETS = {
    "Maruti Swift Dzire": {
        "brand": "Maruti", "year": 2016, "km_driven": 65000, "fuel": "Diesel",
        "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner",
        "mileage": 24.3, "engine": 1248.0, "max_power": 74.0, "seats": 5.0
    },
    "Honda City Petrol": {
        "brand": "Honda", "year": 2019, "km_driven": 35000, "fuel": "Petrol",
        "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner",
        "mileage": 18.0, "engine": 1497.0, "max_power": 117.3, "seats": 5.0
    },
    "Renault Duster Diesel": {
        "brand": "Renault", "year": 2015, "km_driven": 80000, "fuel": "Diesel",
        "seller_type": "Individual", "transmission": "Manual", "owner": "Second Owner",
        "mileage": 19.8, "engine": 1461.0, "max_power": 108.5, "seats": 5.0
    },
    "BMW 3 Series Luxury": {
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
    lakhs = number / 100000.0
    if lakhs >= 100.0:
        crores = lakhs / 100.0
        return f"₹{crores:.2f} Crore (₹{int(number):,})"
    elif lakhs >= 1.0:
        return f"₹{lakhs:.2f} Lakh (₹{int(number):,})"
    else:
        return f"₹{int(number):,}"


def main():
    # Sidebar Navigation matching Fixora Dashboard
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-logo">F</div>
            <div>
                <div class="sidebar-title">Fixora</div>
                <div class="sidebar-sub">Fleet Intelligence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        selected_tab = st.radio(
            "Navigation",
            ["📊 Executive Overview", "🚗 Fleet Inventory", "⚡ Price Appraisal Studio", "📈 Market Intelligence"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("""
        <div style="padding: 0.75rem; border-radius: 0.75rem; background: rgba(2, 6, 23, 0.6); border: 1px solid rgba(30, 41, 59, 0.8);">
            <p style="font-size: 0.75rem; color: #94a3b8; margin: 0; font-weight: 600;">VALUATION ENGINE</p>
            <p style="font-size: 0.85rem; color: #34d399; margin: 0.2rem 0; font-weight: 700;">● Fixora Core v2.4 Live</p>
            <p style="font-size: 0.7rem; color: #64748b; margin: 0;">High Liquidity Index · 8,128 Assets</p>
        </div>
        """, unsafe_allow_html=True)

    model, _ = load_trained_pipeline()

    # TAB 1: EXECUTIVE OVERVIEW
    if selected_tab == "📊 Executive Overview":
        st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.4rem; font-weight: 800; color: #ffffff; letter-spacing: -0.03em; margin-bottom: 0.3rem;">
                Elevate Your <span style="font-family:'Newsreader', serif; font-style:italic; font-weight:400; color:#c084fc;">Fleet</span> Management with Fixora
            </h1>
            <p style="color: #94a3b8; font-size: 1rem;">Optimize trade-in acquisition pricing, accelerate inventory turnover velocity, and protect dealership gross margins.</p>
        </div>
        """, unsafe_allow_html=True)

        # Business KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Managed Fleet Assets</div>
                <div class="kpi-val" style="color: #ffffff;">8,128</div>
                <span style="font-size: 0.7rem; color: #34d399;">↑ Across 32 Automotive Brands</span>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Portfolio Valuation</div>
                <div class="kpi-val" style="color: #c084fc;">₹52.8 Cr</div>
                <span style="font-size: 0.7rem; color: #a855f7;">Total Asset Capitalization</span>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Average Unit Price</div>
                <div class="kpi-val" style="color: #34d399;">₹6.49 L</div>
                <span style="font-size: 0.7rem; color: #94a3b8;">Commercial & Retail Mix</span>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Gross Margin Potential</div>
                <div class="kpi-val" style="color: #818cf8;">+12.5%</div>
                <span style="font-size: 0.7rem; color: #94a3b8;">₹75k – ₹95k Per Unit Spread</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Dashboard 2-column view
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown("### ⚡ Commercial Resale Appraisal Simulator")
            st.markdown("<p style='color:#94a3b8; font-size:0.85rem;'>Select a profile preset to benchmark acquisition and retail values immediately.</p>", unsafe_allow_html=True)
            preset_choice = st.selectbox("Vehicle Profile Preset", list(PRESETS.keys()))
            p_data = PRESETS[preset_choice]

            if model:
                sample_df = pd.DataFrame([{
                    'km_driven': p_data['km_driven'],
                    'fuel': p_data['fuel'],
                    'seller_type': p_data['seller_type'],
                    'transmission': p_data['transmission'],
                    'owner': p_data['owner'],
                    'mileage_num': p_data['mileage'],
                    'engine_num': p_data['engine'],
                    'max_power_num': p_data['max_power'],
                    'seats': p_data['seats'],
                    'brand': p_data['brand'],
                    'car_age': 2026 - p_data['year']
                }])
                est_val = float(np.expm1(model.predict(sample_df))[0])
                buy_in = est_val * 0.90
                retail = est_val * 1.10
                margin_low = retail - est_val
                margin_high = retail - buy_in

                st.markdown(f"""
                <div class="valuation-result-card" style="padding: 1.5rem; margin-top: 1rem;">
                    <div class="result-kicker">Estimated Fair Market Value (FMV)</div>
                    <div class="result-price" style="font-size: 2.4rem;">{format_inr(est_val)}</div>
                    <div style="display:flex; justify-content:space-around; background:rgba(2,6,23,0.5); padding:0.75rem; border-radius:0.75rem; border:1px solid rgba(30,41,59,0.7); margin-top:1rem;">
                        <div>
                            <span style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Target Buy-In</span>
                            <p style="font-size:1.1rem; font-weight:700; color:#34d399; margin:0;">{format_inr(buy_in)}</p>
                        </div>
                        <div>
                            <span style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Target Retail</span>
                            <p style="font-size:1.1rem; font-weight:700; color:#c084fc; margin:0;">{format_inr(retail)}</p>
                        </div>
                        <div>
                            <span style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Margin Spread</span>
                            <p style="font-size:1.1rem; font-weight:700; color:#818cf8; margin:0;">+{format_inr(margin_low)}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            st.markdown("### 📊 Value Retention Trends")
            st.markdown("<p style='color:#94a3b8; font-size:0.85rem;'>Historical trade-in residual value across vehicle operational lifecycle.</p>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background: rgba(2, 6, 23, 0.5); border: 1px solid rgba(30, 41, 59, 0.7); border-radius: 1rem; padding: 1.5rem; height: 210px; display: flex; align-items: flex-end; justify-content: space-around;">
                <div style="display:flex; flex-direction:column; align-items:center; gap:0.3rem;">
                    <div style="width:34px; height:60px; background:rgba(147,51,234,0.35); border-radius:6px 6px 0 0;"></div>
                    <span style="font-size:0.7rem; color:#94a3b8;">&lt;2 yrs</span>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; gap:0.3rem;">
                    <div style="width:34px; height:105px; background:rgba(147,51,234,0.55); border-radius:6px 6px 0 0;"></div>
                    <span style="font-size:0.7rem; color:#94a3b8;">4 yrs</span>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; gap:0.3rem;">
                    <div style="width:34px; height:150px; background:#9333ea; border-radius:6px 6px 0 0; box-shadow:0 0 15px rgba(147,51,234,0.6);"></div>
                    <span style="font-size:0.7rem; color:#c084fc; font-weight:700;">Optimal</span>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; gap:0.3rem;">
                    <div style="width:34px; height:115px; background:rgba(147,51,234,0.75); border-radius:6px 6px 0 0;"></div>
                    <span style="font-size:0.7rem; color:#94a3b8;">8 yrs</span>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; gap:0.3rem;">
                    <div style="width:34px; height:140px; background:rgba(147,51,234,0.9); border-radius:6px 6px 0 0;"></div>
                    <span style="font-size:0.7rem; color:#94a3b8;">10+ yrs</span>
                </div>
            </div>
            <p style="font-size: 0.75rem; color: #94a3b8; text-align: center; margin-top: 0.75rem;">Peak trade liquidity observed in 4–6 year fleet tier.</p>
            """, unsafe_allow_html=True)

    # TAB 2: FLEET INVENTORY
    elif selected_tab == "🚗 Fleet Inventory":
        st.markdown("## 🚗 Commercial Fleet Inventory")
        st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>Browse, search, and evaluate vehicles with real-time acquisition and retail price guidance.</p>", unsafe_allow_html=True)

        df = load_sample_dataset()
        if df is not None:
            search = st.text_input("🔍 Search by vehicle name", "")
            if search:
                df_filtered = df[df['name'].str.contains(search, case=False, na=False)]
            else:
                df_filtered = df

            st.dataframe(
                df_filtered[['name', 'year', 'selling_price', 'km_driven', 'fuel', 'transmission', 'mileage', 'engine', 'max_power']].head(25),
                use_container_width=True
            )
            st.caption(f"Showing top 25 matches out of {len(df_filtered):,} active records.")

    # TAB 3: PRICE APPRAISAL STUDIO
    elif selected_tab == "⚡ Price Appraisal Studio":
        st.markdown("## ⚡ Commercial Vehicle Valuation Studio")
        st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>Calibrated on over 8,100 market transactions for precise acquisition and retail pricing guidance.</p>", unsafe_allow_html=True)

        p_name = st.selectbox("Quick Vehicle Profile Preset", ["Custom Vehicle"] + list(PRESETS.keys()))
        p_val = PRESETS.get(p_name, PRESETS["Maruti Swift Dzire"])

        with st.container():
            st.markdown('<div class="form-panel">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            with col1:
                b_idx = ALL_BRANDS.index(p_val["brand"]) if p_val["brand"] in ALL_BRANDS else 0
                brand = st.selectbox("Brand / Make", ALL_BRANDS, index=b_idx)
                year = st.slider("Manufacturing Year", 1995, 2026, p_val["year"])
                car_age = 2026 - year
                st.caption(f"Vehicle Age: **{car_age} years**")
                km_driven = st.number_input("Kilometers Driven", 100, 1000000, p_val["km_driven"], step=2500)
                fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG'], index=['Petrol', 'Diesel', 'CNG', 'LPG'].index(p_val["fuel"]))

            with col2:
                transmission = st.selectbox("Transmission", ['Manual', 'Automatic'], index=['Manual', 'Automatic'].index(p_val["transmission"]))
                owner = st.selectbox("Ownership History", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], index=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'].index(p_val["owner"]))
                seller_type = st.selectbox("Seller Category", ['Individual', 'Dealer', 'Trustmark Dealer'], index=['Individual', 'Dealer', 'Trustmark Dealer'].index(p_val["seller_type"]))
                seats = st.selectbox("Seating Capacity", [2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=[2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].index(p_val["seats"]))

            with col3:
                mileage_num = st.number_input("Fuel Economy Mileage (kmpl)", 5.0, 45.0, float(p_val["mileage"]), step=0.5)
                engine_num = st.number_input("Displacement Engine (CC)", 600.0, 5000.0, float(p_val["engine"]), step=50.0)
                max_power_num = st.number_input("Max Power (bhp)", 30.0, 500.0, float(p_val["max_power"]), step=5.0)

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            calc_btn = st.button("Generate Commercial Valuation", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)

        if calc_btn and model:
            inp = pd.DataFrame([{
                'km_driven': km_driven, 'fuel': fuel, 'seller_type': seller_type,
                'transmission': transmission, 'owner': owner, 'mileage_num': mileage_num,
                'engine_num': engine_num, 'max_power_num': max_power_num, 'seats': seats,
                'brand': brand, 'car_age': car_age
            }])
            pred = float(np.expm1(model.predict(inp))[0])
            buy_in = pred * 0.90
            retail = pred * 1.10
            margin_low = retail - pred
            margin_high = retail - buy_in

            st.markdown(f"""
            <div class="valuation-result-card">
                <div class="result-kicker">Estimated Fair Market Value (FMV)</div>
                <div class="result-price">{format_inr(pred)}</div>
                
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:1rem; max-width:650px; margin:1.5rem auto; text-align:left;">
                    <div style="background:rgba(2,6,23,0.7); border:1px solid rgba(30,41,59,0.8); border-radius:0.75rem; padding:0.85rem;">
                        <span style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Target Buy-In Price</span>
                        <p style="font-size:1.25rem; font-weight:800; color:#34d399; margin:0.2rem 0 0 0;">{format_inr(buy_in)}</p>
                    </div>
                    <div style="background:rgba(2,6,23,0.7); border:1px solid rgba(30,41,59,0.8); border-radius:0.75rem; padding:0.85rem;">
                        <span style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Target Retail Listing</span>
                        <p style="font-size:1.25rem; font-weight:800; color:#c084fc; margin:0.2rem 0 0 0;">{format_inr(retail)}</p>
                    </div>
                    <div style="background:rgba(2,6,23,0.7); border:1px solid rgba(30,41,59,0.8); border-radius:0.75rem; padding:0.85rem;">
                        <span style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Gross Margin Spread</span>
                        <p style="font-size:1.25rem; font-weight:800; color:#818cf8; margin:0.2rem 0 0 0;">+{format_inr(margin_low)}</p>
                    </div>
                </div>

                <div>
                    <span class="spec-pill">🚘 {brand}</span>
                    <span class="spec-pill">📅 {year} ({car_age}y)</span>
                    <span class="spec-pill">🛣️ {km_driven:,} km</span>
                    <span class="spec-pill">⚡ {max_power_num} bhp</span>
                    <span class="spec-pill">🏎️ {engine_num:.0f} CC</span>
                    <span class="spec-pill">⚙️ {transmission}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # TAB 4: MARKET INTELLIGENCE
    elif selected_tab == "📈 Market Intelligence":
        st.markdown("## 📈 Market Intelligence & Residual Value Trends")
        st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>Capital recovery, residual curves, and demand trends observed across 8,128 vehicle transactions.</p>", unsafe_allow_html=True)
        eda_img = Path("eda_visualizations.png")
        if eda_img.exists():
            st.image(str(eda_img), caption="Automotive Feature Correlation & Market Price Distribution Analysis", use_container_width=True)
        else:
            st.info("Run `python generate_eda.py` to generate the market analysis visualization.")


if __name__ == '__main__':
    main()

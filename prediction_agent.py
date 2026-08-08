import streamlit as st
import pandas as pd
import pickle


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# ULTRA-MODERN DARK THEME & CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main App background */
    .stApp {
        background-color: #0b0f19;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    /* Typography Header Styling */
    .main-title {
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .main-subtitle {
        font-size: 15px;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 24px;
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-top: 5px;
    }

    .sidebar-subtitle {
        text-align: center;
        color: #64748b !important;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 20px;
    }

    /* Feature Label Cards in Sidebar */
    .feature-title {
        font-size: 13px;
        font-weight: 700;
        color: #e2e8f0 !important;
        margin-top: 14px;
    }

    .feature-description {
        font-size: 11px;
        color: #64748b !important;
        margin-bottom: 6px;
        line-height: 1.3;
    }

    /* Custom Input Controls */
    [data-testid="stSidebar"] div[data-baseweb="input"], 
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }

    /* Glassmorphism Containers */
    [data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 20px;
    }

    /* Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        padding: 12px 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 20px -4px rgba(2, 132, 199, 0.4) !important;
    }

    div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

    /* Metric Display Box */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 14px !important;
        padding: 20px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 34px !important;
        font-weight: 800 !important;
    }

    /* Text Color Fixes */
    h1, h2, h3, h4, h5, h6, p, span {
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# DEFAULT VALUES & CURRENCY CONVERSION DICTIONARY
# =========================================================

defaults = {
    "area": 3000,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "yes",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 2,
    "prefarea": "yes",
    "furnishingstatus": "furnished",
    "currency": "PKR"
}

EXCHANGE_RATES = {
    "PKR": {"symbol": "Rs.", "rate": 1.0},
    "INR": {"symbol": "₹", "rate": 0.30},
    "USD": {"symbol": "$", "rate": 0.0036},
    "EUR": {"symbol": "€", "rate": 0.0033}
}


# =========================================================
# RESET FUNCTION
# =========================================================

def reset_form():
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state["prediction"] = None


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None


# =========================================================
# LOAD MODEL & ENCODERS
# =========================================================

@st.cache_resource
def load_artifacts():
    with open("linear_regression_model.pkl", "rb") as f:
        mdl = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    return mdl, enc

try:
    model, encoders = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🏠 House Predictor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">Enter property specifications</div>',
        unsafe_allow_html=True
    )

    # Reset Button
    st.button(
        "🔄 Reset All Inputs",
        on_click=reset_form,
        use_container_width=True,
        type="secondary"
    )

   
    
    st.markdown("---")

    # Feature Inputs
    st.markdown('<div class="feature-title">📐 Area</div>', unsafe_allow_html=True)
    area = st.number_input("Area (sq ft)", min_value=100, step=100, key="area")

    st.markdown('<div class="feature-title">🛏️ Bedrooms</div>', unsafe_allow_html=True)
    bedrooms = st.number_input("Bedrooms", min_value=1, step=1, key="bedrooms")

    st.markdown('<div class="feature-title">🛁 Bathrooms</div>', unsafe_allow_html=True)
    bathrooms = st.number_input("Bathrooms", min_value=1, step=1, key="bathrooms")

    st.markdown('<div class="feature-title">🏢 Stories</div>', unsafe_allow_html=True)
    stories = st.number_input("Stories", min_value=1, step=1, key="stories")

    st.markdown('<div class="feature-title">🛣️ Main Road</div>', unsafe_allow_html=True)
    mainroad = st.selectbox("Main Road", ["yes", "no"], key="mainroad")

    st.markdown('<div class="feature-title">🛏️ Guest Room</div>', unsafe_allow_html=True)
    guestroom = st.selectbox("Guest Room", ["yes", "no"], key="guestroom")

    st.markdown('<div class="feature-title">🏚️ Basement</div>', unsafe_allow_html=True)
    basement = st.selectbox("Basement", ["yes", "no"], key="basement")

    st.markdown('<div class="feature-title">🔥 Hot Water Heating</div>', unsafe_allow_html=True)
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"], key="hotwaterheating")

    st.markdown('<div class="feature-title">❄️ Air Conditioning</div>', unsafe_allow_html=True)
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"], key="airconditioning")

    st.markdown('<div class="feature-title">🚗 Parking</div>', unsafe_allow_html=True)
    parking = st.number_input("Parking Spaces", min_value=0, step=1, key="parking")

    st.markdown('<div class="feature-title">📍 Preferred Area</div>', unsafe_allow_html=True)
    prefarea = st.selectbox("Preferred Area", ["yes", "no"], key="prefarea")

    st.markdown('<div class="feature-title">🛋️ Furnishing Status</div>', unsafe_allow_html=True)
    furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"], key="furnishingstatus")


# =========================================================
# MAIN PAGE HEADER & INTRODUCTION
# =========================================================

st.markdown('<div class="main-title">🏠 House Price Predictor</div>', unsafe_allow_html=True)


with st.container(border=True):
    st.subheader("📋 System Overview")
    st.write(
        "Enter house details from the sidebar. "
        "Categorical values are transformed using saved label encoders. "
        "The trained Linear Regression model predicts the valuation in real time."
    )


# =========================================================
# SMART INPUT VALIDATIONS (Dynamic Alerts)
# =========================================================

if area > 8000 and bedrooms <= 2:
    st.warning("⚠️ **Unusual Configuration Detected:** Area is extremely large (>8,000 sq ft) for 2 or fewer bedrooms.")
elif bedrooms > 5 and bathrooms == 1:
    st.warning("⚠️ **Unusual Configuration Detected:** 5+ bedrooms with only 1 bathroom is atypical for standard valuations.")


# =========================================================
# PREDICT BUTTON & PROCESSING
# =========================================================

st.write("")

predict_button = st.button(
    "🔮 Predict House Price",
    type="primary",
    use_container_width=True
)

if predict_button:

    sample_df = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishingstatus]
    })

    # ENCODING
    sample_df["mainroad"] = encoders["mainroad"].transform(sample_df["mainroad"])
    sample_df["guestroom"] = encoders["guestroom"].transform(sample_df["guestroom"])
    sample_df["basement"] = encoders["basement"].transform(sample_df["basement"])
    sample_df["hotwaterheating"] = encoders["hotwaterheating"].transform(sample_df["hotwaterheating"])
    sample_df["airconditioning"] = encoders["airconditioning"].transform(sample_df["airconditioning"])
    sample_df["prefarea"] = encoders["prefarea"].transform(sample_df["prefarea"])
    sample_df["furnishingstatus"] = encoders["furnishingstatus"].transform(sample_df["furnishingstatus"])

    # PREDICTION
    raw_prediction = model.predict(sample_df)[0]
    st.session_state["prediction"] = float(raw_prediction)


# =========================================================
# SHOW PREDICTION & REPORT SECTION
# =========================================================

if st.session_state["prediction"] is not None:

    st.write("")
    st.subheader("🎯 Prediction Result")

    # Currency Conversion Calculation
    curr_data = EXCHANGE_RATES[selected_currency]
    converted_price = st.session_state["prediction"] * curr_data["rate"]
    symbol = curr_data["symbol"]

    # Top Metric Display Box
    st.metric(
        label=f"🏠 Estimated Price ({selected_currency})",
        value=f"{symbol} {converted_price:,.2f}"
    )

    st.write("")

    # Summary Table Dataframe
    summary_df = pd.DataFrame({
        "Feature": [
            "Area (sq ft)", "Bedrooms", "Bathrooms", "Stories", 
            "Main Road", "Guest Room", "Basement", "Hot Water Heating", 
            "Air Conditioning", "Parking", "Preferred Area", "Furnishing Status",
            f"Predicted Price ({selected_currency})"
        ],
        "Value": [
            f"{area}", bedrooms, bathrooms, stories, 
            mainroad, guestroom, basement, hotwaterheating, 
            airconditioning, parking, prefarea, furnishingstatus,
            f"{symbol} {converted_price:,.2f}"
        ]
    })

    col_exp, col_dl = st.columns([3, 1])

    with col_exp:
        with st.expander("📊 View Input Summary"):
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with col_dl:
        csv_data = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Report (CSV)",
            data=csv_data,
            file_name="house_price_prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )
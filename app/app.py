from pathlib import Path
import sys
import base64

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Needed for joblib model deserialization if the pipeline contains this custom transformer
from src.feature_engineering import ClinicalFeatureEngineer  # noqa: F401
from src.auth import ensure_db, create_user, authenticate_user
from src.explanations import generate_role_based_explanation

# =========================================================
# FORCE PROJECT ROOT INTO PYTHON PATH
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Pediatric BMT Survival Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = PROJECT_ROOT
MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "model_columns.pkl"
INFO_PATH = BASE_DIR / "models" / "model_info.pkl"
CSS_PATH = BASE_DIR / "app" / "styles" / "main.css"

# =========================================================
# SESSION STATE
# =========================================================
if "user" not in st.session_state:
    st.session_state.user = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

# =========================================================
# INIT DB
# =========================================================
ensure_db()

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_artifacts():
    missing_files = [str(p) for p in [MODEL_PATH, COLUMNS_PATH, INFO_PATH] if not p.exists()]
    if missing_files:
        raise FileNotFoundError("Missing required model files:\n" + "\n".join(missing_files))

    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    model_info = joblib.load(INFO_PATH)
    return model, model_columns, model_info


try:
    model, model_columns, model_info = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# =========================================================
# CSS / STYLING
# =========================================================
def load_css():
    # Load external CSS first if present
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Force a readable theme on top
    st.markdown(
        """
        <style>
        .stApp {
            color: #0f2740 !important;
        }

        .main {
            background: transparent !important;
            color: #0f2740 !important;
        }

        [data-testid="stHeader"] {
            background: rgba(215, 230, 240, 0.72) !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #d7e6f0 0%, #c7d9e8 100%) !important;
            border-right: 1px solid #b7cede !important;
        }

        section[data-testid="stSidebar"] * {
            color: #0f2740 !important;
        }

        .hero-box {
            background: rgba(241, 247, 251, 0.97) !important;
            border: 1px solid #b7cede !important;
            border-radius: 22px !important;
            padding: 1.4rem !important;
            margin-bottom: 1.1rem !important;
            box-shadow: 0 10px 28px rgba(16, 42, 67, 0.10) !important;
            color: #102a43 !important;
        }

        .mission-sign {
            font-size: 2.2rem !important;
            margin-bottom: 0.35rem !important;
        }

        .hero-subtitle {
            font-size: 1.02rem !important;
            margin-top: 0.35rem !important;
            color: #2f4d68 !important;
        }

        .hero-badges {
            display: flex !important;
            gap: 0.6rem !important;
            flex-wrap: wrap !important;
            margin-top: 0.9rem !important;
        }

        .hero-badge {
            background: #edf5fb !important;
            border: 1px solid #bfd3e1 !important;
            border-radius: 999px !important;
            padding: 0.42rem 0.85rem !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            color: #21435d !important;
        }

        .role-card, .metric-card {
            background: rgba(236, 244, 249, 0.97) !important;
            border: 1px solid #bfd3e1 !important;
            border-radius: 18px !important;
            padding: 1rem !important;
            margin-bottom: 1rem !important;
            color: #102a43 !important;
            box-shadow: 0 6px 18px rgba(16, 42, 67, 0.08) !important;
            min-height: 170px !important;
        }

        h1, h2, h3, h4, h5, h6, p, label, div, span {
            color: #102a43 !important;
        }

        /* Closed select boxes */
        div[data-baseweb="select"] > div {
            background: #111827 !important;
            color: #ffffff !important;
            border: 1px solid #374151 !important;
            border-radius: 12px !important;
            min-height: 46px !important;
            box-shadow: none !important;
        }

        div[data-baseweb="select"] > div > div {
            color: #ffffff !important;
        }

        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] p,
        div[data-baseweb="select"] div {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        div[data-baseweb="select"] svg {
            fill: #ffffff !important;
        }

        /* Open dropdown menu + all options */
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        div[role="listbox"] {
            background: #0b1220 !important;
            color: #ffffff !important;
        }

        ul[role="listbox"] {
            background: #0b1220 !important;
            border: 1px solid #374151 !important;
            border-radius: 12px !important;
            color: #ffffff !important;
        }

        ul[role="listbox"] *,
        div[role="option"],
        div[role="option"] *,
        li[role="option"],
        li[role="option"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        ul[role="listbox"] li,
        div[role="option"],
        li[role="option"] {
            background: #0b1220 !important;
        }

        ul[role="listbox"] li:hover,
        div[role="option"]:hover,
        li[role="option"]:hover {
            background: #1f2937 !important;
            color: #ffffff !important;
        }

        ul[role="listbox"] li[aria-selected="true"],
        div[role="option"][aria-selected="true"],
        li[role="option"][aria-selected="true"] {
            background: #374151 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Number inputs remain light and readable */
        div[data-testid="stNumberInput"] {
            background: #f7fbff !important;
            border: 1px solid #9fbcd3 !important;
            border-radius: 12px !important;
            padding: 2px !important;
        }

        div[data-testid="stNumberInput"] input {
            background: #ffffff !important;
            color: #102a43 !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }

        div[data-testid="stNumberInput"] button {
            background: #e8f2f8 !important;
            color: #2f6f9f !important;
            border: none !important;
            border-radius: 8px !important;
        }

        div[data-testid="stNumberInput"] button:hover {
            background: #d9e9f4 !important;
            color: #174d78 !important;
        }

        /* Text inputs */
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {
            background: #f7fbff !important;
            color: #102a43 !important;
            border: 1px solid #9fbcd3 !important;
            border-radius: 12px !important;
        }

        /* Buttons */
        .stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: #2f6f9f !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }

        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #21597f !important;
            color: #ffffff !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(236, 244, 249, 0.88) !important;
            border-radius: 10px 10px 0 0 !important;
            color: #102a43 !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(201, 220, 235, 0.96) !important;
            color: #0b2239 !important;
            font-weight: 700 !important;
        }

        /* Tables / metrics */
        div[data-testid="stDataFrame"] {
            background: rgba(247, 251, 255, 0.96) !important;
            border: 1px solid #bfd3e1 !important;
            border-radius: 12px !important;
        }

        div[data-testid="stMetric"] {
            background: rgba(247, 251, 255, 0.96) !important;
            border: 1px solid #bfd3e1 !important;
            border-radius: 12px !important;
            padding: 0.6rem !important;
        }

        /* Code / json / pre */
        pre, code, .stCodeBlock, [data-testid="stCodeBlock"] {
            background: #f7fbff !important;
            color: #102a43 !important;
            border: 1px solid #bfd3e1 !important;
            border-radius: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


load_css()

# =========================================================
# OPTIONAL MEDICAL BACKGROUND
# =========================================================
def set_background_from_asset():
    possible_files = [
        BASE_DIR / "app" / "assets" / "background.jpg",
        BASE_DIR / "app" / "assets" / "background.png",
        BASE_DIR / "app" / "assets" / "doctor_bg.jpg",
        BASE_DIR / "app" / "assets" / "doctor_bg.png",
    ]

    selected_file = next((path for path in possible_files if path.exists()), None)

    if selected_file:
        with open(selected_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        ext = "jpg" if selected_file.suffix.lower() == ".jpg" else "png"

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image:
                    linear-gradient(rgba(220, 235, 245, 0.88), rgba(184, 209, 227, 0.92)),
                    url("data:image/{ext};base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #dcebf5 0%, #c9dceb 55%, #b8d1e3 100%);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


set_background_from_asset()

# =========================================================
# HELPERS
# =========================================================
def classify_survival_risk(survival_probability: float) -> str:
    if survival_probability >= 0.80:
        return "Very Low Risk"
    if survival_probability >= 0.65:
        return "Lower Risk"
    if survival_probability >= 0.45:
        return "Intermediate Risk"
    if survival_probability >= 0.25:
        return "Concerning Risk"
    return "High Risk"


def risk_color(survival_probability: float) -> str:
    if survival_probability >= 0.80:
        return "#1b9e77"
    if survival_probability >= 0.65:
        return "#4daf4a"
    if survival_probability >= 0.45:
        return "#ffb000"
    if survival_probability >= 0.25:
        return "#ff7f0e"
    return "#d62728"


def plot_probability_gauge(survival_probability: float):
    percent = survival_probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percent,
            number={"suffix": "%"},
            title={"text": "Predicted Survival Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color(survival_probability)},
                "steps": [
                    {"range": [0, 25], "color": "#fde2e2"},
                    {"range": [25, 45], "color": "#fff0d6"},
                    {"range": [45, 65], "color": "#fff7cc"},
                    {"range": [65, 80], "color": "#e6f5d6"},
                    {"range": [80, 100], "color": "#d9f3e5"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def get_feature_names_from_pipeline(model_obj):
    try:
        base_estimator = model_obj.estimator if hasattr(model_obj, "estimator") else model_obj
        preprocessor = base_estimator.named_steps["preprocessor"]
        return preprocessor.get_feature_names_out()
    except Exception:
        return None


def clean_feature_name(name: str) -> str:
    pretty_feature_names = {
        "Recipientgender": "Recipient Gender",
        "Stemcellsource": "Stem Cell Source",
        "Donorage": "Donor Age",
        "Gendermatch": "Gender Match",
        "DonorABO": "Donor ABO",
        "RecipientABO": "Recipient ABO",
        "RecipientRh": "Recipient Rh",
        "CMVstatus": "CMV Serostatus",
        "DonorCMV": "Donor CMV",
        "RecipientCMV": "Recipient CMV",
        "Disease": "Disease Type",
        "Riskgroup": "Risk Group",
        "Diseasegroup": "Disease Group",
        "HLAmatch": "HLA Match",
        "HLAgrI": "HLA Group I",
        "Recipientage": "Recipient Age",
        "CD34kgx10d6": "CD34+ Cell Dose (10^6/kg)",
        "Rbodymass": "Recipient Body Mass (kg)",
        "age_gap": "Age Gap",
        "donor_older": "Donor Older Than Recipient",
        "donor_age_ratio": "Donor/Recipient Age Ratio",
        "abo_mismatch": "ABO Mismatch",
        "cmv_mismatch": "CMV Mismatch",
        "hla_perfect_match": "Perfect HLA Match",
        "hla_severe_mismatch": "Severe HLA Mismatch",
        "cd34_bodymass_interaction": "CD34/Body Mass Interaction",
        "recipient_age_group_engineered": "Recipient Age Group",
        "donor_age_group_engineered": "Donor Age Group",
    }

    if name.startswith("num__"):
        raw = name.replace("num__", "")
        return pretty_feature_names.get(raw, raw)

    if name.startswith("cat__"):
        raw = name.replace("cat__", "")
        for original in pretty_feature_names:
            prefix = f"{original}_"
            if raw.startswith(prefix):
                suffix = raw.replace(prefix, "")
                return f"{pretty_feature_names.get(original, original)} = {suffix}"
        return pretty_feature_names.get(raw, raw)

    return pretty_feature_names.get(name, name)


def get_shap_values(model_obj, patient_data):
    try:
        base_estimator = model_obj.estimator if hasattr(model_obj, "estimator") else model_obj
        preprocessor = base_estimator.named_steps["preprocessor"]
        classifier = base_estimator.named_steps["classifier"]

        if "feature_engineering" in base_estimator.named_steps:
            engineered = base_estimator.named_steps["feature_engineering"].transform(patient_data)
        else:
            engineered = patient_data

        transformed = preprocessor.transform(engineered)
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(transformed)

        if isinstance(shap_vals, list):
            patient_shap = shap_vals[1][0]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            patient_shap = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 2:
            patient_shap = shap_vals[0]
        else:
            patient_shap = np.array(shap_vals)[0]

        return np.array(patient_shap)
    except Exception:
        return None


def explain_top_effects(model_obj, patient_data, top_n=8):
    shap_values = get_shap_values(model_obj, patient_data)
    feature_names = get_feature_names_from_pipeline(model_obj)

    if shap_values is None or feature_names is None:
        return []

    impacts = []
    for name, value in zip(feature_names, shap_values):
        impacts.append((clean_feature_name(name), float(value)))

    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    return impacts[:top_n]


def plot_shap_bar(top_effects):
    labels = [item[0] for item in top_effects]
    values = [item[1] for item in top_effects]
    colors = ["green" if v > 0 else "red" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP Value")
    ax.set_title("Top Feature Contributions")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

# =========================================================
# INPUT MAPPINGS
# =========================================================
recipient_gender_map = {"Female": "0", "Male": "1"}
stemcell_source_map = {"Bone Marrow": "0", "Peripheral Blood": "1"}
gender_match_map = {"Other": "0", "Female to Male": "1"}
abo_map = {"O": "0", "A": "1", "B": "-1", "AB": "2"}
recipient_rh_map = {"Negative": "0", "Positive": "1"}
cmv_binary_map = {"Absent": "0", "Present": "1"}
risk_group_map = {"Low Risk": "0", "High Risk": "1"}
disease_group_map = {"Non-malignant": "0", "Malignant": "1"}
hla_match_map = {"10/10": "0", "9/10": "1", "8/10": "2", "7/10": "3"}
hla_group_map = {
    "HLA matched": "0",
    "One antigen difference": "1",
    "One allele difference": "2",
    "Only DRB1 difference": "3",
    "Two differences (type 1)": "4",
    "Two differences (type 2)": "5",
}

disease_options = ["ALL", "AML", "chronic", "nonmalignant", "lymphoma"]
cmvstatus_options = [
    "Donor-/Recipient-",
    "Donor-/Recipient+",
    "Donor+/Recipient-",
    "Donor+/Recipient+",
]

# =========================================================
# AUTH UI
# =========================================================
def show_auth_screen():
    st.markdown(
        """
        <div class="hero-box">
            <div class="mission-sign">🧒🦴🩸</div>
            <h1>Pediatric BMT Survival Predictor</h1>
            <p class="hero-subtitle">
                Explainable AI support for pediatric bone marrow transplant evaluation,
                combining prediction, transparency, and role-based clinical communication.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">Pediatric Transplant Support</span>
                <span class="hero-badge">SHAP Explainability</span>
                <span class="hero-badge">Clinical Decision Support</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="role-card">
                <h3>🦴 Bone Marrow Focus</h3>
                <p>
                    Designed for pediatric bone marrow transplant prognosis support
                    using donor, recipient, and transplant-related variables.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="role-card">
                <h3>📊 Transparent Predictions</h3>
                <p>
                    Uses explainable machine learning and SHAP-based interpretation
                    to make predictions clearer and easier to understand.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="role-card">
                <h3>👩‍⚕️ Role-Based Access</h3>
                <p>
                    Adapts communication for doctors, nurses, and general users
                    to match each level of clinical interpretation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    mode = st.radio("Choose access mode", ["Login", "Create Account"], horizontal=True)

    if mode == "Login":
        st.subheader("Login")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            user = authenticate_user(email, password)
            if user:
                st.session_state.user = user
                st.success(f"Welcome back, {user['full_name']} ({user['role']}).")
                st.rerun()
            else:
                st.error("Invalid email or password.")
    else:
        st.subheader("Create Account")
        with st.form("signup_form"):
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            role = st.selectbox(
                "Role",
                ["doctor", "nurse", "public"],
                format_func=lambda x: {
                    "doctor": "Doctor",
                    "nurse": "Nurse",
                    "public": "General User",
                }[x],
            )
            submitted = st.form_submit_button("Create Account")

        if submitted:
            result = create_user(full_name, email, password, role)
            if result["success"]:
                st.success(result["message"] + " You can now log in.")
            else:
                st.error(result["message"])

# =========================================================
# MAIN APP UI
# =========================================================
def show_sidebar_inputs():
    st.sidebar.header("Patient Inputs")

    st.sidebar.subheader("Recipient Information")
    recipient_gender = st.sidebar.selectbox("Recipient Gender", list(recipient_gender_map.keys()))
    recipient_age = st.sidebar.number_input("Recipient Age", min_value=0.0, max_value=25.0, value=8.0, step=1.0)
    recipient_abo = st.sidebar.selectbox("Recipient ABO", list(abo_map.keys()))
    recipient_rh = st.sidebar.selectbox("Recipient Rh", list(recipient_rh_map.keys()))
    recipient_cmv = st.sidebar.selectbox("Recipient CMV", list(cmv_binary_map.keys()))
    recipient_body_mass = st.sidebar.number_input("Recipient Body Mass (kg)", min_value=1.0, max_value=150.0, value=25.0, step=1.0)

    st.sidebar.subheader("Donor Information")
    donor_age = st.sidebar.number_input("Donor Age", min_value=0.0, max_value=80.0, value=28.0, step=1.0)
    donor_abo = st.sidebar.selectbox("Donor ABO", list(abo_map.keys()))
    donor_cmv = st.sidebar.selectbox("Donor CMV", list(cmv_binary_map.keys()))

    st.sidebar.subheader("Disease Information")
    disease = st.sidebar.selectbox("Disease Type", disease_options)
    risk_group = st.sidebar.selectbox("Risk Group", list(risk_group_map.keys()))
    disease_group = st.sidebar.selectbox("Disease Group", list(disease_group_map.keys()))

    st.sidebar.subheader("Transplant Information")
    stemcell_source = st.sidebar.selectbox("Stem Cell Source", list(stemcell_source_map.keys()))
    gender_match = st.sidebar.selectbox("Gender Match", list(gender_match_map.keys()))
    cmv_status = st.sidebar.selectbox("CMV Serological Compatibility", cmvstatus_options)
    hla_match = st.sidebar.selectbox("HLA Match", list(hla_match_map.keys()))
    hla_group = st.sidebar.selectbox("HLA Group I", list(hla_group_map.keys()))
    cd34_dose = st.sidebar.number_input("CD34+ Cell Dose (10^6/kg)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)

    st.sidebar.markdown("---")
    predict_now = st.sidebar.button("Predict Outcome", use_container_width=True)

    inputs = {
        "Recipientgender": recipient_gender_map[recipient_gender],
        "Stemcellsource": stemcell_source_map[stemcell_source],
        "Donorage": donor_age,
        "Gendermatch": gender_match_map[gender_match],
        "DonorABO": abo_map[donor_abo],
        "RecipientABO": abo_map[recipient_abo],
        "RecipientRh": recipient_rh_map[recipient_rh],
        "CMVstatus": cmv_status,
        "DonorCMV": cmv_binary_map[donor_cmv],
        "RecipientCMV": cmv_binary_map[recipient_cmv],
        "Disease": disease,
        "Riskgroup": risk_group_map[risk_group],
        "Diseasegroup": disease_group_map[disease_group],
        "HLAmatch": hla_match_map[hla_match],
        "HLAgrI": hla_group_map[hla_group],
        "Recipientage": recipient_age,
        "CD34kgx10d6": cd34_dose,
        "Rbodymass": recipient_body_mass,
    }

    return inputs, predict_now


def build_patient_dataframe(inputs: dict) -> pd.DataFrame:
    patient_data = pd.DataFrame([inputs])
    missing_cols = [col for col in model_columns if col not in patient_data.columns]
    if missing_cols:
        raise ValueError("Model/interface mismatch. Missing columns required by model: " + ", ".join(missing_cols))
    return patient_data[model_columns]


def show_main_app():
    user = st.session_state.user
    role = user["role"]

    st.markdown(
        f"""
        <div class="hero-box">
            <div class="mission-sign">🧒🦴🩸</div>
            <h1>Pediatric BMT Survival Predictor</h1>
            <p class="hero-subtitle">
                <b>Logged in as:</b> {user['full_name']} |
                <b>Role:</b> {role.title()}
            </p>
            <div class="hero-badges">
                <span class="hero-badge">Mission: Pediatric Bone Marrow Transplant Support</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    topbar_col1, topbar_col2 = st.columns([8, 1])
    with topbar_col2:
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

    inputs, predict_now = show_sidebar_inputs()

    if not predict_now:
        st.info("Fill in the clinical data in the sidebar, then click 'Predict Outcome'.")
        return

    try:
        patient_data = build_patient_dataframe(inputs)
    except Exception as e:
        st.error(f"Input/model mismatch: {e}")
        return

    try:
        proba = model.predict_proba(patient_data)[0]
        prediction = int(model.predict(patient_data)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    if len(proba) != 2:
        st.error("Unexpected probability output from model.")
        return

    death_probability = float(proba[1])
    survival_probability = 1.0 - death_probability
    risk_label = classify_survival_risk(survival_probability)

    top_effects = explain_top_effects(model, patient_data, top_n=8)

    role_based_text = generate_role_based_explanation(
        role=role,
        survival_probability=survival_probability,
        death_probability=death_probability,
        risk_label=risk_label,
        top_effects=top_effects,
    )

    tabs = st.tabs(["Prediction", "Explanation", "Technical", "About"])

    with tabs[0]:
        col1, col2 = st.columns([1, 1.25])

        with col1:
            st.markdown("### Patient Summary")
            summary_df = pd.DataFrame({"Feature": patient_data.columns, "Value": patient_data.iloc[0].values})
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### Prediction Result")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Survival Probability", f"{survival_probability:.1%}")
            with m2:
                st.metric("Death Probability", f"{death_probability:.1%}")
            with m3:
                st.metric("Risk Level", risk_label)

            st.progress(min(max(survival_probability, 0.0), 1.0))
            st.plotly_chart(plot_probability_gauge(survival_probability), use_container_width=True)

            if prediction == 0:
                st.success("Model output leans toward survival.")
            else:
                st.error("Model output leans toward death risk.")

            if role == "doctor":
                st.info(
                    "For clinicians: class interpretation is based on target coding "
                    "`0 = alive`, `1 = dead`, and displayed survival is computed as `1 - P(dead)`.")

    with tabs[1]:
        st.markdown("### Role-Based Explanation")
        st.write(role_based_text)

        if top_effects:
            st.markdown("### Top Influential Factors")
            fig = plot_shap_bar(top_effects)
            st.pyplot(fig)
        else:
            st.warning("SHAP explanation is not available for this model configuration.")

    with tabs[2]:
        if role == "doctor":
            st.markdown("### Scientific / Technical View")
            st.write("This section is visible only to doctors.")

            tech_info_df = pd.DataFrame(
                [
                    ["Best Model", model_info.get("best_model_name")],
                    ["Test Accuracy", model_info.get("test_accuracy")],
                    ["Test ROC-AUC", model_info.get("test_roc_auc")],
                    ["Final CV ROC-AUC Mean", model_info.get("final_cv_roc_auc_mean")],
                    ["Final CV ROC-AUC Std", model_info.get("final_cv_roc_auc_std")],
                    ["Target Meaning", str(model_info.get("target_meaning"))],
                ],
                columns=["Metric", "Value"],
            )
            st.dataframe(tech_info_df, use_container_width=True, hide_index=True)

            if top_effects:
                tech_df = pd.DataFrame(top_effects, columns=["Feature", "SHAP Value"])
                st.dataframe(tech_df, use_container_width=True, hide_index=True)

        elif role == "nurse":
            st.markdown("### Practical Clinical Notes")
            st.write(
                "This model combines transplant compatibility, donor-recipient characteristics, "
                "disease-related features, and graft information to estimate prognosis supportively."
            )
            st.write(
                "Use this result as an aid for attention and monitoring priorities, not as a replacement "
                "for physician judgment."
            )
        else:
            st.markdown("### How to Read This Result")
            st.write(
                "The percentage shown is a computer-based estimate built from medical data. "
                "It is not a guarantee and should always be discussed with healthcare professionals."
            )

    with tabs[3]:
        st.markdown("### About This Platform")
        st.write(
            "This application predicts pediatric bone marrow transplant survival using a machine learning model "
            "trained on pre-transplant and transplant-time variables."
        )
        st.write(
            "The interface adapts the explanation level according to the user profile: doctor, nurse, or public."
        )
        st.warning(
            "This tool is for educational and decision-support purposes only. "
            "It must not replace physician expertise, clinical protocols, or medical guidelines."
        )

# =========================================================
# APP ROUTER
# =========================================================
if st.session_state.user is None:
    show_auth_screen()
else:
    show_main_app()

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

# Force project root into Python path before loading local modules / pickle classes
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import ClinicalFeatureEngineer  # noqa: F401
from src.auth import ensure_db, create_user, authenticate_user
from src.explanations import generate_role_based_explanation


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
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Force the final theme after external CSS so the override always wins.
    st.markdown(
        """
        <style>
        :root {
            --bmt-bg-1: #b7cfdf;
            --bmt-bg-2: #9cb8ca;
            --bmt-bg-3: #7f9fb3;
            --bmt-surface: rgba(241, 247, 251, 0.92);
            --bmt-surface-2: rgba(229, 238, 245, 0.92);
            --bmt-border: rgba(67, 102, 128, 0.26);
            --bmt-text: #102a43;
            --bmt-text-soft: #486581;
            --bmt-primary: #165d79;
            --bmt-secondary: #2f855a;
            --bmt-danger: #c53030;
            --bmt-shadow: 0 16px 40px rgba(16, 42, 67, 0.12);
        }

        html, body, [class*="css"] {
            color: var(--bmt-text);
        }

        .stApp {
            color: var(--bmt-text) !important;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stAppViewContainer"] {
            background: transparent !important;
        }

        [data-testid="stHeader"] {
            background: rgba(201, 221, 234, 0.52) !important;
            backdrop-filter: blur(8px);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(220, 235, 245, 0.92) 0%, rgba(188, 210, 225, 0.96) 100%) !important;
            border-right: 1px solid rgba(55, 84, 109, 0.10);
        }

        [data-testid="stSidebar"] * {
            color: var(--bmt-text) !important;
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(244, 249, 253, 0.94) 0%, rgba(227, 237, 245, 0.92) 100%);
            border: 1px solid var(--bmt-border);
            border-radius: 28px;
            padding: 1.35rem 1.35rem 1.2rem 1.35rem;
            box-shadow: var(--bmt-shadow);
            margin-bottom: 1.1rem;
        }

        .hero-box {
            background: linear-gradient(135deg, rgba(241, 247, 251, 0.96) 0%, rgba(229, 238, 245, 0.94) 100%) !important;
            border: 1px solid var(--bmt-border) !important;
            border-radius: 24px !important;
            padding: 1.4rem !important;
            margin-bottom: 1rem !important;
            box-shadow: var(--bmt-shadow) !important;
            color: var(--bmt-text) !important;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 96px 1fr;
            gap: 1rem;
            align-items: center;
        }

        .hero-icon {
            width: 92px;
            height: 92px;
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #dbeaf3 0%, #bad2e1 100%);
            border: 1px solid rgba(22, 93, 121, 0.18);
            box-shadow: inset 0 2px 10px rgba(255,255,255,0.45), 0 12px 28px rgba(22, 93, 121, 0.10);
            font-size: 2.35rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            color: var(--bmt-text);
            margin: 0;
            line-height: 1.05;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: var(--bmt-text-soft);
            margin: 0.35rem 0 0 0;
            line-height: 1.55;
        }

        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.95rem;
        }

        .hero-pill {
            background: rgba(22, 93, 121, 0.08);
            color: var(--bmt-primary);
            border: 1px solid rgba(22, 93, 121, 0.12);
            padding: 0.42rem 0.8rem;
            border-radius: 999px;
            font-size: 0.88rem;
            font-weight: 600;
        }

        .accent-line {
            height: 5px;
            width: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #1f6f8b 0%, #3a8fb7 45%, #48bb78 100%);
            margin-top: 1rem;
        }

        .feature-card {
            background: linear-gradient(135deg, rgba(244, 249, 253, 0.88) 0%, rgba(232, 240, 246, 0.92) 100%);
            border: 1px solid var(--bmt-border);
            border-radius: 20px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 10px 26px rgba(16, 42, 67, 0.08);
            min-height: 150px;
        }

        .feature-card h4 {
            margin: 0 0 0.45rem 0;
            color: var(--bmt-text) !important;
            font-size: 1.02rem;
            font-weight: 700;
        }

        .feature-card p {
            margin: 0;
            color: var(--bmt-text-soft) !important;
            line-height: 1.55;
            font-size: 0.93rem;
        }

        .mini-stat {
            background: rgba(255,255,255,0.45);
            border: 1px solid rgba(22, 93, 121, 0.12);
            border-radius: 16px;
            padding: 0.75rem 0.85rem;
        }

        .mini-stat .value {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--bmt-primary);
            line-height: 1;
        }

        .mini-stat .label {
            font-size: 0.82rem;
            color: var(--bmt-text-soft);
            margin-top: 0.25rem;
        }

        .role-card, .metric-card {
            background: linear-gradient(135deg, rgba(241, 247, 251, 0.96) 0%, rgba(229, 238, 245, 0.92) 100%) !important;
            border: 1px solid var(--bmt-border) !important;
            border-radius: 18px !important;
            padding: 1rem !important;
            margin-bottom: 1rem !important;
            color: var(--bmt-text) !important;
            box-shadow: 0 12px 28px rgba(16, 42, 67, 0.08) !important;
        }

        .section-card {
            background: linear-gradient(135deg, rgba(242, 248, 252, 0.94) 0%, rgba(232, 240, 246, 0.92) 100%);
            border: 1px solid var(--bmt-border);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(16, 42, 67, 0.08);
        }

        .section-card h3,
        .section-card p,
        .section-card li {
            color: var(--bmt-text) !important;
        }

        .focus-banner {
            background: linear-gradient(90deg, rgba(22, 93, 121, 0.12) 0%, rgba(72, 187, 120, 0.12) 100%);
            border: 1px solid rgba(22, 93, 121, 0.16);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            color: var(--bmt-text);
            margin-bottom: 0.8rem;
            font-size: 0.95rem;
        }

        .focus-banner strong {
            color: var(--bmt-primary);
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        textarea {
            background: rgba(248, 252, 255, 0.94) !important;
            color: var(--bmt-text) !important;
            border: 1px solid rgba(67, 102, 128, 0.22) !important;
            border-radius: 14px !important;
        }

        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stMultiSelect label,
        .stRadio label,
        .stMarkdown,
        .stAlert,
        .stSubheader,
        h1, h2, h3, h4, h5, h6, p, li, span, div {
            color: var(--bmt-text) !important;
        }

        .stRadio > div {
            gap: 0.8rem;
        }

        .stRadio [role="radiogroup"] {
            background: rgba(244, 249, 253, 0.74);
            border: 1px solid rgba(67, 102, 128, 0.18);
            border-radius: 14px;
            padding: 0.45rem 0.75rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(234, 243, 248, 0.82) !important;
            border-radius: 14px 14px 0 0 !important;
            border: 1px solid rgba(67, 102, 128, 0.14) !important;
            padding: 0.6rem 1rem !important;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(205, 224, 236, 0.98) !important;
            color: #0b2239 !important;
            font-weight: 700 !important;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, #1f6f8b 0%, #18556f 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 0.58rem 1rem !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 22px rgba(22, 93, 121, 0.22);
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: linear-gradient(135deg, #18556f 0%, #124559 100%) !important;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(244, 249, 253, 0.96) 0%, rgba(231, 240, 246, 0.94) 100%);
            border: 1px solid var(--bmt-border);
            padding: 1rem;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(16, 42, 67, 0.08);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--bmt-text-soft) !important;
        }

        .stDataFrame,
        div[data-testid="stAlert"] {
            background: rgba(244, 249, 253, 0.86) !important;
            border-radius: 18px !important;
            border: 1px solid rgba(67, 102, 128, 0.12) !important;
        }

        .footer-note {
            color: var(--bmt-text-soft);
            font-size: 0.88rem;
            margin-top: 0.5rem;
        }

        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: 1fr;
                text-align: center;
            }

            .hero-icon {
                margin: 0 auto;
            }

            .hero-pills {
                justify-content: center;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


load_css()


# =========================================================
# MEDICAL BACKGROUND
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
                    linear-gradient(rgba(222, 235, 244, 0.86), rgba(170, 194, 209, 0.88)),
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
                background: linear-gradient(135deg, #b7cfdf 0%, #9cb8ca 52%, #7f9fb3 100%) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


set_background_from_asset()


# =========================================================
# HELPERS
# =========================================================
def render_project_signature(user=None):
    user_line = ""
    if user is not None:
        user_line = (
            f"<p class='hero-subtitle'><b>Logged in as:</b> {user['full_name']} "
            f"| <b>Role:</b> {user['role'].title()}</p>"
        )

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-grid">
                <div class="hero-icon">🦴🩸</div>
                <div>
                    <h1 class="hero-title">Pediatric BMT Survival Predictor</h1>
                    <p class="hero-subtitle">
                        Explainable AI support for pediatric <b>bone marrow transplant</b> evaluation,
                        combining prediction, transparency, and role-based clinical communication.
                    </p>
                    {user_line}
                    <div class="hero-pills">
                        <span class="hero-pill">🧬 Bone Marrow Transplant</span>
                        <span class="hero-pill">📊 SHAP Explainability</span>
                        <span class="hero-pill">🩺 Clinical Decision Support</span>
                    </div>
                </div>
            </div>
            <div class="accent-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_highlights():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🦴 Bone Marrow Focus</h4>
                <p>
                    Designed around pediatric bone marrow transplant evaluation to make the purpose of the platform
                    instantly clear when a visitor lands on the site.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h4>📈 Transparent Predictions</h4>
                <p>
                    The interface highlights prediction confidence and SHAP-based contributing factors so results
                    remain readable and explainable.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h4>👩‍⚕️ Multi-Role Experience</h4>
                <p>
                    Doctors, nurses, and general users each receive an interpretation adapted to their level of use.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_focus_banner():
    st.markdown(
        """
        <div class="focus-banner">
            <strong>BMT Insight:</strong> This application estimates pediatric bone marrow transplant outcome support
            using clinical compatibility, disease context, and graft-related information.
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def get_feature_names_from_pipeline(model):
    try:
        base_estimator = model.estimator if hasattr(model, "estimator") else model
        preprocessor = base_estimator.named_steps["preprocessor"]
        return preprocessor.get_feature_names_out()
    except Exception:
        return None


PRETTY_FEATURE_NAMES = {
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


def clean_feature_name(name: str) -> str:
    if name.startswith("num__"):
        raw = name.replace("num__", "")
        return PRETTY_FEATURE_NAMES.get(raw, raw)

    if name.startswith("cat__"):
        raw = name.replace("cat__", "")
        for original in PRETTY_FEATURE_NAMES:
            prefix = f"{original}_"
            if raw.startswith(prefix):
                suffix = raw.replace(prefix, "")
                return f"{PRETTY_FEATURE_NAMES.get(original, original)} = {suffix}"
        return PRETTY_FEATURE_NAMES.get(raw, raw)

    return PRETTY_FEATURE_NAMES.get(name, name)


def get_shap_values(model, patient_data):
    try:
        base_estimator = model.estimator if hasattr(model, "estimator") else model
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


def explain_top_effects(model, patient_data, top_n=8):
    shap_values = get_shap_values(model, patient_data)
    feature_names = get_feature_names_from_pipeline(model)

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
    fig.patch.set_alpha(0)
    ax.set_facecolor((1, 1, 1, 0.65))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP Value")
    ax.set_title("Top Feature Contributions")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# =========================================================
# INPUT MAPPINGS
# IMPORTANT:
# keep these exactly as used during training
# =========================================================
recipient_gender_map = {
    "Female": "0",
    "Male": "1",
}

stemcell_source_map = {
    "Bone Marrow": "0",
    "Peripheral Blood": "1",
}

gender_match_map = {
    "Other": "0",
    "Female to Male": "1",
}

abo_map = {
    "O": "0",
    "A": "1",
    "B": "-1",
    "AB": "2",
}

recipient_rh_map = {
    "Negative": "0",
    "Positive": "1",
}

cmv_binary_map = {
    "Absent": "0",
    "Present": "1",
}

risk_group_map = {
    "Low Risk": "0",
    "High Risk": "1",
}

disease_group_map = {
    "Non-malignant": "0",
    "Malignant": "1",
}

hla_match_map = {
    "10/10": "0",
    "9/10": "1",
    "8/10": "2",
    "7/10": "3",
}

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
    render_project_signature()
    render_focus_banner()

    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.markdown(
            """
            <div class="mini-stat">
                <div class="value">BMT</div>
                <div class="label">Bone marrow transplant focus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_col2:
        st.markdown(
            """
            <div class="mini-stat">
                <div class="value">SHAP</div>
                <div class="label">Explainable model behavior</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_col3:
        st.markdown(
            """
            <div class="mini-stat">
                <div class="value">AI</div>
                <div class="label">Clinical decision-support interface</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    render_auth_highlights()
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    mode = st.radio(
        "Choose access mode",
        ["Login", "Create Account"],
        horizontal=True,
    )

    if mode == "Login":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<p class='footer-note'>This interface is intended for educational and decision-support use, not as a replacement for medical judgment.</p>",
        unsafe_allow_html=True,
    )


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
    recipient_body_mass = st.sidebar.number_input(
        "Recipient Body Mass (kg)", min_value=1.0, max_value=150.0, value=25.0, step=1.0
    )

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
        raise ValueError(
            "Model/interface mismatch. Missing columns required by model: " + ", ".join(missing_cols)
        )

    return patient_data[model_columns]


def show_main_app():
    user = st.session_state.user
    role = user["role"]

    render_project_signature(user=user)
    render_focus_banner()

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

    # IMPORTANT: assumes target coding is 0 = alive, 1 = dead
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
            summary_df = pd.DataFrame({
                "Feature": [clean_feature_name(col) for col in patient_data.columns],
                "Value": patient_data.iloc[0].values,
            })
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

            st.json({
                "best_model_name": model_info.get("best_model_name"),
                "test_accuracy": model_info.get("test_accuracy"),
                "test_roc_auc": model_info.get("test_roc_auc"),
                "final_cv_roc_auc_mean": model_info.get("final_cv_roc_auc_mean"),
                "final_cv_roc_auc_std": model_info.get("final_cv_roc_auc_std"),
                "target_meaning": model_info.get("target_meaning"),
            })

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

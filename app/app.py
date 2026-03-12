import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Pediatric BMT Survival Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩺 Pediatric BMT Survival Predictor")
st.caption(
    "Pre-transplant clinical decision-support tool for pediatric bone marrow transplantation."
)


# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main-card {
    background-color: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.small-note {
    color: #6b7280;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/rf_model.pkl")
    model_columns = joblib.load("models/model_columns.pkl")
    return model, model_columns


model, model_columns = load_artifacts()


# =========================
# MAPPINGS
# =========================
recipient_gender_map = {
    "Female": "0",
    "Male": "1"
}

stemcell_source_map = {
    "Bone Marrow": "0",
    "Peripheral Blood": "1"
}

gender_match_map = {
    "Other": "0",
    "Female to Male": "1"
}

abo_map = {
    "O": "0",
    "A": "1",
    "B": "-1",
    "AB": "2"
}

recipient_rh_map = {
    "Negative": "0",
    "Positive": "1"
}

cmv_binary_map = {
    "Absent": "0",
    "Present": "1"
}

risk_group_map = {
    "Low Risk": "0",
    "High Risk": "1"
}

disease_group_map = {
    "Non-malignant": "0",
    "Malignant": "1"
}

hla_match_map = {
    "10/10": "0",
    "9/10": "1",
    "8/10": "2",
    "7/10": "3"
}

hla_group_map = {
    "HLA matched": "0",
    "One antigen difference": "1",
    "One allele difference": "2",
    "Only DRB1 difference": "3",
    "Two differences (type 1)": "4",
    "Two differences (type 2)": "5"
}

disease_options = ["ALL", "AML", "chronic", "nonmalignant", "lymphoma"]

cmvstatus_options = [
    "Donor-/Recipient-",
    "Donor-/Recipient+",
    "Donor+/Recipient-",
    "Donor+/Recipient+"
]

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
    "Rbodymass": "Recipient Body Mass (kg)"
}


# =========================
# HELPERS
# =========================
def classify_risk(probability: float) -> str:
    if probability < 0.40:
        return "High Risk"
    elif probability < 0.70:
        return "Moderate Risk"
    return "Lower Risk"


def risk_color(probability: float) -> str:
    if probability < 0.40:
        return "#d62728"
    elif probability < 0.70:
        return "#ff7f0e"
    return "#2ca02c"


def build_patient_dataframe(inputs: dict) -> pd.DataFrame:
    patient_data = pd.DataFrame([inputs])
    patient_data = patient_data[model_columns]
    return patient_data


def plot_probability_gauge(probability: float):
    percent = probability * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        number={"suffix": "%"},
        title={"text": "Predicted Survival Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": risk_color(probability)},
            "steps": [
                {"range": [0, 40], "color": "#fde2e2"},
                {"range": [40, 70], "color": "#fff0d6"},
                {"range": [70, 100], "color": "#dff5e3"},
            ]
        }
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def get_feature_names_from_pipeline(model):
    preprocessor = model.named_steps["preprocessor"]
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return None


def clean_feature_name(name: str) -> str:
    # Removes pipeline prefixes for display
    if name.startswith("num__"):
        raw = name.replace("num__", "")
        return pretty_feature_names.get(raw, raw)

    if name.startswith("cat__"):
        raw = name.replace("cat__", "")
        # One-hot encoded features look like: cat__Disease_ALL
        for original in pretty_feature_names:
            prefix = f"{original}_"
            if raw.startswith(prefix):
                suffix = raw.replace(prefix, "")
                return f"{pretty_feature_names.get(original, original)} = {suffix}"
        return raw

    return name


def get_shap_values(model, patient_data):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(patient_data)
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


def explain_top_effects(model, patient_data, top_n=8):
    shap_values = get_shap_values(model, patient_data)
    feature_names = get_feature_names_from_pipeline(model)

    if feature_names is None:
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


def get_example_patient(example_name: str) -> dict:
    presets = {
        "Custom": {
            "Recipientgender": "0",
            "Stemcellsource": "0",
            "Donorage": 28.0,
            "Gendermatch": "0",
            "DonorABO": "1",
            "RecipientABO": "1",
            "RecipientRh": "1",
            "CMVstatus": "Donor-/Recipient-",
            "DonorCMV": "0",
            "RecipientCMV": "0",
            "Disease": "ALL",
            "Riskgroup": "0",
            "Diseasegroup": "1",
            "HLAmatch": "0",
            "HLAgrI": "0",
            "Recipientage": 8.0,
            "CD34kgx10d6": 5.0,
            "Rbodymass": 25.0
        },
        "Lower Risk Example": {
            "Recipientgender": "0",
            "Stemcellsource": "1",
            "Donorage": 24.0,
            "Gendermatch": "0",
            "DonorABO": "1",
            "RecipientABO": "1",
            "RecipientRh": "1",
            "CMVstatus": "Donor-/Recipient-",
            "DonorCMV": "0",
            "RecipientCMV": "0",
            "Disease": "ALL",
            "Riskgroup": "0",
            "Diseasegroup": "1",
            "HLAmatch": "0",
            "HLAgrI": "0",
            "Recipientage": 6.0,
            "CD34kgx10d6": 8.0,
            "Rbodymass": 22.0
        },
        "Moderate Risk Example": {
            "Recipientgender": "1",
            "Stemcellsource": "0",
            "Donorage": 36.0,
            "Gendermatch": "1",
            "DonorABO": "0",
            "RecipientABO": "1",
            "RecipientRh": "1",
            "CMVstatus": "Donor+/Recipient-",
            "DonorCMV": "1",
            "RecipientCMV": "0",
            "Disease": "AML",
            "Riskgroup": "1",
            "Diseasegroup": "1",
            "HLAmatch": "1",
            "HLAgrI": "2",
            "Recipientage": 11.0,
            "CD34kgx10d6": 4.5,
            "Rbodymass": 30.0
        },
        "High Risk Example": {
            "Recipientgender": "1",
            "Stemcellsource": "0",
            "Donorage": 45.0,
            "Gendermatch": "1",
            "DonorABO": "-1",
            "RecipientABO": "2",
            "RecipientRh": "0",
            "CMVstatus": "Donor+/Recipient+",
            "DonorCMV": "1",
            "RecipientCMV": "1",
            "Disease": "AML",
            "Riskgroup": "1",
            "Diseasegroup": "1",
            "HLAmatch": "3",
            "HLAgrI": "5",
            "Recipientage": 16.0,
            "CD34kgx10d6": 2.5,
            "Rbodymass": 42.0
        }
    }
    return presets.get(example_name, presets["Custom"])


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Patient Inputs")

example_choice = st.sidebar.selectbox(
    "Load Example Patient",
    ["Custom", "Lower Risk Example", "Moderate Risk Example", "High Risk Example"]
)

preset = get_example_patient(example_choice)

st.sidebar.subheader("Recipient Information")
recipient_gender = st.sidebar.selectbox(
    "Recipient Gender",
    list(recipient_gender_map.keys()),
    index=list(recipient_gender_map.values()).index(preset["Recipientgender"])
)
recipient_age = st.sidebar.number_input(
    "Recipient Age",
    min_value=0.0,
    max_value=25.0,
    value=float(preset["Recipientage"]),
    step=1.0
)
recipient_abo = st.sidebar.selectbox(
    "Recipient ABO",
    list(abo_map.keys()),
    index=list(abo_map.values()).index(preset["RecipientABO"])
)
recipient_rh = st.sidebar.selectbox(
    "Recipient Rh",
    list(recipient_rh_map.keys()),
    index=list(recipient_rh_map.values()).index(preset["RecipientRh"])
)
recipient_cmv = st.sidebar.selectbox(
    "Recipient CMV",
    list(cmv_binary_map.keys()),
    index=list(cmv_binary_map.values()).index(preset["RecipientCMV"])
)
recipient_body_mass = st.sidebar.number_input(
    "Recipient Body Mass (kg)",
    min_value=1.0,
    max_value=150.0,
    value=float(preset["Rbodymass"]),
    step=1.0
)

st.sidebar.subheader("Donor Information")
donor_age = st.sidebar.number_input(
    "Donor Age",
    min_value=0.0,
    max_value=80.0,
    value=float(preset["Donorage"]),
    step=1.0
)
donor_abo = st.sidebar.selectbox(
    "Donor ABO",
    list(abo_map.keys()),
    index=list(abo_map.values()).index(preset["DonorABO"])
)
donor_cmv = st.sidebar.selectbox(
    "Donor CMV",
    list(cmv_binary_map.keys()),
    index=list(cmv_binary_map.values()).index(preset["DonorCMV"])
)

st.sidebar.subheader("Disease Information")
disease = st.sidebar.selectbox(
    "Disease Type",
    disease_options,
    index=disease_options.index(preset["Disease"])
)
risk_group = st.sidebar.selectbox(
    "Risk Group",
    list(risk_group_map.keys()),
    index=list(risk_group_map.values()).index(preset["Riskgroup"])
)
disease_group = st.sidebar.selectbox(
    "Disease Group",
    list(disease_group_map.keys()),
    index=list(disease_group_map.values()).index(preset["Diseasegroup"])
)

st.sidebar.subheader("Transplant Information")
stemcell_source = st.sidebar.selectbox(
    "Stem Cell Source",
    list(stemcell_source_map.keys()),
    index=list(stemcell_source_map.values()).index(preset["Stemcellsource"])
)
gender_match = st.sidebar.selectbox(
    "Gender Match",
    list(gender_match_map.keys()),
    index=list(gender_match_map.values()).index(preset["Gendermatch"])
)
cmv_status = st.sidebar.selectbox(
    "CMV Serological Compatibility",
    cmvstatus_options,
    index=cmvstatus_options.index(preset["CMVstatus"])
)
hla_match = st.sidebar.selectbox(
    "HLA Match",
    list(hla_match_map.keys()),
    index=list(hla_match_map.values()).index(preset["HLAmatch"])
)
hla_group = st.sidebar.selectbox(
    "HLA Group I",
    list(hla_group_map.keys()),
    index=list(hla_group_map.values()).index(preset["HLAgrI"])
)
cd34_dose = st.sidebar.number_input(
    "CD34+ Cell Dose (10^6/kg)",
    min_value=0.0,
    max_value=30.0,
    value=float(preset["CD34kgx10d6"]),
    step=0.1
)

live_mode = st.sidebar.toggle("Live Prediction Update", value=True)
show_shap_values = st.sidebar.toggle("Show Raw SHAP Values", value=False)

predict_now = True
if not live_mode:
    predict_now = st.sidebar.button("Predict Survival", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application is intended for educational decision support only."
)


# =========================
# MAIN
# =========================
patient_inputs = {
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
    "Rbodymass": recipient_body_mass
}

patient_data = build_patient_dataframe(patient_inputs)

if predict_now:
    probability = float(model.predict_proba(patient_data)[0][1])
    prediction = int(model.predict(patient_data)[0])
    risk_label = classify_risk(probability)

    tab1, tab2, tab3 = st.tabs([
        "Prediction",
        "Explanation",
        "About"
    ])

    with tab1:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("### Patient Summary")
            summary_df = pd.DataFrame({
                "Feature": [pretty_feature_names.get(col, col) for col in patient_data.columns],
                "Value": patient_data.iloc[0].values
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### Prediction Result")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Predicted Survival Probability", f"{probability:.1%}")
            with m2:
                st.metric("Clinical Risk Level", risk_label)

            st.progress(min(max(probability, 0.0), 1.0))
            st.plotly_chart(plot_probability_gauge(probability), use_container_width=True)

            if prediction == 1:
                st.success("Model output suggests survival is more likely.")
            else:
                st.error("Model output suggests lower likelihood of survival.")

            st.caption(
                "Interpret this result as model-based decision support, not as a definitive prognosis."
            )

    with tab2:
        st.markdown("### Model Explanation")
        st.write(
            "This section shows the strongest local contributions for this patient using SHAP. "
            "Positive values push the prediction toward survival, while negative values push it away."
        )

        try:
            top_effects = explain_top_effects(model, patient_data)

            if top_effects:
                fig = plot_shap_bar(top_effects)
                st.pyplot(fig)

                top_df = pd.DataFrame(top_effects, columns=["Feature", "SHAP Value"])
                if show_shap_values:
                    st.dataframe(top_df, use_container_width=True, hide_index=True)
            else:
                st.info("No SHAP explanation could be generated.")
        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {e}")

    with tab3:
        st.markdown("### About This Application")
        st.write(
            "This application predicts pediatric bone marrow transplant survival using "
            "pre-transplant clinical and transplant-time variables."
        )

        st.markdown("### Core Modeling Principle")
        st.write(
            "Only features available before or at the time of transplantation are used. "
            "Post-transplant events are excluded to avoid data leakage."
        )

        st.markdown("### Disclaimer")
        st.warning(
            "This tool is for educational and decision-support purposes only. "
            "It must not replace physician expertise, clinical protocols, or medical guidelines."
        )
else:
    st.info("Configure the patient data in the sidebar, then click the prediction button.")

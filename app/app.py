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
st.title("TEST MODIF TOTO 999")
st.set_page_config(
    page_title="Pediatric BMT Success Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.sub-text {
    color: #555;
    font-size: 1rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f7f9fc;
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #e6eaf1;
    margin-bottom: 1rem;
}
.small-note {
    font-size: 0.9rem;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL & FEATURES
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("app/rf_model.pkl")
    model_columns = joblib.load("app/model_columns.pkl")
    return model, model_columns

model, model_columns = load_artifacts()

# =========================
# HELPERS
# =========================
def build_patient_dataframe(relapse, plt_recovery, cd34):
    patient_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    if "Relapse" in model_columns:
        patient_data["Relapse"] = relapse
    if "PLTrecovery" in model_columns:
        patient_data["PLTrecovery"] = plt_recovery
    if "CD34kgx10d6" in model_columns:
        patient_data["CD34kgx10d6"] = cd34
    return patient_data

def classify_risk(probability):
    if probability < 0.40:
        return "High Risk"
    elif probability < 0.70:
        return "Moderate Risk"
    return "Lower Risk"

def risk_color(probability):
    if probability < 0.40:
        return "#d62728"   # red
    elif probability < 0.70:
        return "#ff7f0e"   # orange
    return "#2ca02c"       # green

def get_example_patient(example_name):
    presets = {
        "Custom": {"Relapse": 0, "PLTrecovery": 20.0, "CD34kgx10d6": 5.0},
        "Low Risk Example": {"Relapse": 0, "PLTrecovery": 12.0, "CD34kgx10d6": 8.5},
        "Moderate Risk Example": {"Relapse": 1, "PLTrecovery": 22.0, "CD34kgx10d6": 5.0},
        "High Risk Example": {"Relapse": 1, "PLTrecovery": 35.0, "CD34kgx10d6": 2.5},
    }
    return presets.get(example_name, presets["Custom"])

def explain_top_effects(patient_shap, model_columns, patient_data):
    impacts = {}
    mapping = {
        "Relapse": "Relapse History",
        "PLTrecovery": "Platelet Recovery Time",
        "CD34kgx10d6": "CD34+ Cell Dose"
    }

    for raw_name, pretty_name in mapping.items():
        if raw_name in model_columns:
            idx = model_columns.index(raw_name)
            impacts[pretty_name] = {
                "shap": float(patient_shap[idx]),
                "value": float(patient_data.iloc[0][raw_name]) if raw_name != "Relapse" else int(patient_data.iloc[0][raw_name])
            }

    sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]["shap"]), reverse=True)
    return sorted_impacts

def get_shap_values(model, patient_data):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(patient_data)

    # Robust extraction depending on SHAP version / output format
    if isinstance(shap_vals, list):
        # binary classification old-style: [class0, class1]
        patient_shap = shap_vals[1][0]
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        # shape like (n_samples, n_features, n_classes)
        patient_shap = shap_vals[0, :, 1]
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 2:
        # shape like (n_samples, n_features)
        patient_shap = shap_vals[0]
    else:
        patient_shap = np.array(shap_vals)[0]

    return explainer, np.array(patient_shap)

def plot_probability_gauge(probability):
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

def plot_shap_bar(top_effects):
    labels = [item[0] for item in top_effects]
    values = [item[1]["shap"] for item in top_effects]
    colors = ["green" if v > 0 else "red" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP Value")
    ax.set_title("Feature Impact for This Patient")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def generate_interpretation(top_effects):
    positive = [name for name, meta in top_effects if meta["shap"] > 0]
    negative = [name for name, meta in top_effects if meta["shap"] < 0]

    lines = []
    if positive:
        lines.append("**Positive contributors increasing predicted survival:** " + ", ".join(positive))
    if negative:
        lines.append("**Negative contributors decreasing predicted survival:** " + ", ".join(negative))
    if not lines:
        lines.append("No dominant local contributors were detected for this patient.")
    return lines

# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">🩺 Pediatric BMT Success Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">AI-assisted clinical decision support for pediatric bone marrow transplant outcome prediction. '
    'This interface emphasizes prediction, interpretability, and scenario exploration.</div>',
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Patient Inputs")

example_choice = st.sidebar.selectbox(
    "Load Example Patient",
    ["Custom", "Low Risk Example", "Moderate Risk Example", "High Risk Example"]
)

preset = get_example_patient(example_choice)

st.sidebar.markdown("### Core Clinical Features")

relapse = st.sidebar.selectbox(
    "Relapse History",
    options=[0, 1],
    index=0 if preset["Relapse"] == 0 else 1,
    help="0 = No relapse history, 1 = History of relapse"
)

plt_recovery = st.sidebar.slider(
    "Platelet Recovery Time (Days)",
    min_value=0.0,
    max_value=60.0,
    value=float(preset["PLTrecovery"]),
    step=1.0,
    help="Longer recovery time may indicate delayed hematologic recovery."
)

cd34 = st.sidebar.slider(
    "CD34+ Cell Dose (×10^6/kg)",
    min_value=0.0,
    max_value=20.0,
    value=float(preset["CD34kgx10d6"]),
    step=0.1,
    help="Higher CD34+ dose may support engraftment, depending on clinical context."
)

live_mode = st.sidebar.toggle("Live Prediction Update", value=True)
show_comparison = st.sidebar.toggle("Enable Scenario Comparison", value=True)
show_raw_values = st.sidebar.toggle("Show Raw SHAP Values", value=False)

predict_now = True
if not live_mode:
    predict_now = st.sidebar.button("Predict Transplant Success", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "This tool supports clinical interpretation but does not replace physician judgment."
)

# =========================
# MAIN PREDICTION
# =========================
patient_data = build_patient_dataframe(relapse, plt_recovery, cd34)

if predict_now:
    prediction = model.predict(patient_data)[0]
    probability = float(model.predict_proba(patient_data)[0][1])
    risk_label = classify_risk(probability)

    try:
        explainer, patient_shap = get_shap_values(model, patient_data)
        top_effects = explain_top_effects(patient_shap, model_columns, patient_data)
    except Exception as e:
        explainer, patient_shap, top_effects = None, None, []
        st.warning(f"SHAP explanation could not be generated for this prediction. Details: {e}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Prediction",
        "Explanation",
        "Compare Scenarios",
        "About"
    ])

    # =========================
    # TAB 1 - PREDICTION
    # =========================
    with tab1:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("### Patient Summary")
            display_df = pd.DataFrame({
                "Feature": ["Relapse History", "Platelet Recovery Time", "CD34+ Cell Dose"],
                "Value": [relapse, plt_recovery, cd34]
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("### Clinical Interpretation")
            if probability >= 0.70:
                st.success("This patient falls into the **lower predicted risk** group.")
            elif probability >= 0.40:
                st.warning("This patient falls into the **moderate predicted risk** group.")
            else:
                st.error("This patient falls into the **high predicted risk** group.")

            st.caption(
                "Interpret this output as decision support, not as a definitive prognosis."
            )

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
                st.success(f"Model classification: Survival more likely ({probability:.1%})")
            else:
                st.error(f"Model classification: Lower likelihood of survival ({probability:.1%})")

    # =========================
    # TAB 2 - EXPLANATION
    # =========================
    with tab2:
        st.markdown("### Why did the model make this prediction?")
        st.write(
            "This section shows the local explanation for this specific patient using SHAP "
            "(SHapley Additive Explanations). Positive values push the prediction toward survival; "
            "negative values push it away."
        )

        if top_effects:
            for line in generate_interpretation(top_effects):
                st.markdown(line)

            fig = plot_shap_bar(top_effects[:3])
            st.pyplot(fig)

            cards = st.columns(min(3, len(top_effects[:3])))
            for i, (name, meta) in enumerate(top_effects[:3]):
                direction = "↑ increases survival" if meta["shap"] > 0 else "↓ decreases survival"
                with cards[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <b>{name}</b><br>
                        Patient value: {meta["value"]}<br>
                        Effect: {direction}
                    </div>
                    """, unsafe_allow_html=True)

            if show_raw_values:
                raw_df = pd.DataFrame([
                    {
                        "Feature": name,
                        "Patient Value": meta["value"],
                        "SHAP Value": round(meta["shap"], 5)
                    }
                    for name, meta in top_effects
                ])
                st.markdown("### Raw SHAP Values")
                st.dataframe(raw_df, use_container_width=True, hide_index=True)
        else:
            st.info("No SHAP explanation is currently available for this patient.")

    # =========================
    # TAB 3 - COMPARE SCENARIOS
    # =========================
    with tab3:
        st.markdown("### Scenario Comparison")

        if show_comparison:
            st.write(
                "Compare the current patient with an alternative scenario to understand how "
                "small changes in clinical variables affect prediction."
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Current Patient")
                st.write(f"- Relapse: {relapse}")
                st.write(f"- Platelet Recovery: {plt_recovery} days")
                st.write(f"- CD34+ Dose: {cd34}")
                st.write(f"- Survival Probability: **{probability:.1%}**")

            with c2:
                st.markdown("#### Alternative Scenario")
                alt_relapse = st.selectbox("Alternative Relapse History", [0, 1], index=0, key="alt_relapse")
                alt_plt = st.slider("Alternative Platelet Recovery Time", 0.0, 60.0, float(plt_recovery), 1.0, key="alt_plt")
                alt_cd34 = st.slider("Alternative CD34+ Dose", 0.0, 20.0, float(cd34), 0.1, key="alt_cd34")

                alt_patient = build_patient_dataframe(alt_relapse, alt_plt, alt_cd34)
                alt_prob = float(model.predict_proba(alt_patient)[0][1])
                alt_risk = classify_risk(alt_prob)

                st.write(f"- Relapse: {alt_relapse}")
                st.write(f"- Platelet Recovery: {alt_plt} days")
                st.write(f"- CD34+ Dose: {alt_cd34}")
                st.write(f"- Survival Probability: **{alt_prob:.1%}**")
                st.write(f"- Risk Level: **{alt_risk}**")

            st.markdown("#### Comparison Summary")
            delta = alt_prob - probability
            if delta > 0:
                st.success(f"The alternative scenario improves predicted survival by {delta:.1%}.")
            elif delta < 0:
                st.error(f"The alternative scenario reduces predicted survival by {abs(delta):.1%}.")
            else:
                st.info("Both scenarios produce the same predicted survival probability.")

            compare_df = pd.DataFrame({
                "Scenario": ["Current Patient", "Alternative Scenario"],
                "Survival Probability": [probability * 100, alt_prob * 100]
            })

            compare_fig = go.Figure()
            compare_fig.add_bar(x=compare_df["Scenario"], y=compare_df["Survival Probability"])
            compare_fig.update_layout(
                title="Scenario Comparison",
                yaxis_title="Survival Probability (%)",
                height=350
            )
            st.plotly_chart(compare_fig, use_container_width=True)
        else:
            st.info("Enable scenario comparison from the sidebar to use this feature.")

    # =========================
    # TAB 4 - ABOUT
    # =========================
    with tab4:
        st.markdown("### About This Application")
        st.write(
            "This decision-support application predicts pediatric bone marrow transplant outcomes "
            "using a machine learning model and provides local explainability through SHAP."
        )

        st.markdown("### Core Features Used in This Interface")
        st.write("- **Relapse**: Whether the patient has a prior relapse history.")
        st.write("- **PLTrecovery**: Platelet recovery time, reflecting post-transplant recovery dynamics.")
        st.write("- **CD34kgx10d6**: CD34+ infused cell dose per kilogram.")

        st.markdown("### Important Disclaimer")
        st.warning(
            "This application is intended for educational and decision-support purposes only. "
            "It should not replace physician expertise, medical guidelines, or full clinical evaluation."
        )

else:
    st.info("Configure the patient data in the sidebar, then click the prediction button.")

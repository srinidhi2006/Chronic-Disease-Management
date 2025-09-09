# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
from PIL import Image

import streamlit.components.v1 as components  # For advanced HTML components if needed

# ------------------- CSS Styling -------------------
st.markdown("""
<style>
/* Your existing CSS code here... */
body {
    background: linear-gradient(135deg, #232526 0%, #414345 100%);
    font-family: 'Segoe UI', sans-serif;
    color: #f3f4f6;
    min-height: 100vh;
}
.card {
    background: linear-gradient(135deg, #374151 0%, #4b5563 50%, #6b7280 100%);
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 28px;
    box-shadow: 0 6px 18px rgba(156, 163, 175, 0.25);
    transition: transform 0.2s cubic-bezier(.77,.2,.22,1), box-shadow 0.2s;
    border: 2px solid  #9ca3af;
}
.card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 12px 32px rgba(59, 130, 246, 0.20), 0 2px 12px rgba(88, 56, 255, 0.15);
    border-color: #60a5fa;
}
h1, h2, h3 {
    color: #f9fafb;
    margin-bottom: 16px;
    text-shadow: 0 3px 12px #2563eb90;
    letter-spacing: 1px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #374151 0%, #4b5563 50%, #6b7280 100%);
    color: #fafafa;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 2px 18px rgba(59,130,246,0.20);
}
.stSelectbox>div>div {
    background: linear-gradient(90deg, #9333ea 0%, #3b82f6 100%);
    border-radius: 12px;
    padding: 7px;
    color: #fff;
    font-weight: 600;
    overflow: hidden;
    min-height: 35px;
    box-shadow: 0 1px 8px rgba(59,130,246,0.12);
    height: auto;
    overflow: hidden;
}
.stSelectbox>div>div:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(147,51,234,0.45);
}
.risk-high {
    background: linear-gradient(90deg, #fca5a5 5%, #f87171 60%, #fecaca 100%);
    border-left: 6px solid #ef4444;
    padding: 18px;
    margin-bottom: 19px;
    color: #7f1d1d;
    box-shadow: 0 2px 8px #fca5a528;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.risk-low {
    background: linear-gradient(90deg, #bfdbfe 5%, #60a5fa 60%, #93c5fd 100%);
    border-left: 6px solid #3b82f6;
    padding: 18px;
    margin-bottom: 19px;
    color: #1e40af;
    box-shadow: 0 2px 8px #60a5fa28;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.risk-low {
    background: linear-gradient(90deg, #2563eb 5%, #1e3a8a 60%, #22d3ee 100%);
    border-left: 6px solid #22d3ee;
    padding: 18px;
    margin-bottom: 19px;
    color: #e0f7fa;
    box-shadow: 0 2px 10px #22d3ee28;
    font-weight: 600;
    letter-spacing: 0.5px;
}
::-webkit-scrollbar {
    width: 10px;
    background: #232526;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #9333ea, #3b82f6, #22d3ee);
    border-radius: 10px;
}
input,
select,
textarea {
    background: linear-gradient(90deg, #22223b 0%, #393e46 100%);
    color: #fafafa;
    border-radius: 8px;
    border: 1.5px solid #60a5fa;
    padding: 8px;
    box-shadow: 0 1px 8px rgba(59,130,246,0.09);
}
</style>
""", unsafe_allow_html=True)


# ------------------- Sidebar -------------------
st.sidebar.title("Patient Selection")


# ------------------- Load model -------------------
@st.cache_resource(show_spinner=False)
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()


# ------------------- Load dataset -------------------
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("diabetic_data.csv", index_col=0)
df = load_data()


# ------------------- Features used by model -------------------
model_features = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 
                  'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 
                  'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 
                  'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 
                  'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 
                  'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
                  'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 
                  'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 
                  'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                  'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed']


df_model = df[model_features].copy()

categorical_cols = df_model.select_dtypes(include=['object', 'category']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le


# ------------------- Patient Selection -------------------
patient_ids = df.index.tolist()
patient_id = st.sidebar.selectbox("Select Patient ID", patient_ids)


# ------------------- SHAP Explainer -------------------
explainer = shap.Explainer(model, df_model)


# ------------------- Main Dashboard -------------------
st.title("Patient Risk Prediction Dashboard")


if patient_id:
    patient_data_numeric = df_model.loc[[patient_id]]
    shap_values_patient = explainer(patient_data_numeric)
    prediction = model.predict_proba(patient_data_numeric)[0,1]

    # Collect high risk features
    high_risk_features = []
    for feature, value in zip(model_features, shap_values_patient.values[0]):
        if value > 0:
            high_risk_features.append(feature)

    # ------- Patient Summary Card -------
    st.markdown("<div class='card'><h2>Patient Summary</h2></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    def get_risk_level(prob):
        if prob < 0.3:
            return ("Low", "🟢")
        elif prob < 0.6:
            return ("Moderate", "🟠")
        else:
            return ("High", "🔴")

    risk_label, risk_emoji = get_risk_level(prediction)

    col1.metric("Readmission Risk Probability", f"{prediction:.1%}")
    col2.metric("Patient Age", df.loc[patient_id, "age"] if "age" in df.columns else "N/A")

    gender = df.loc[patient_id, "gender"] if "gender" in df.columns else "N/A"
    col3.metric("Gender", gender)

    st.markdown(f"### Risk Level: {risk_emoji} {risk_label} Risk")

    summary = f"This patient has a {risk_label.lower()} risk of readmission with a probability of {prediction:.1%}."
    if risk_label == "High":
        summary += " Close monitoring and proactive management are strongly recommended."
    elif risk_label == "Moderate":
        summary += " Consider interventions to reduce risk and schedule regular follow-ups."
    else:
        summary += " Standard care and routine monitoring are appropriate."

    st.write(summary)

    num_prior_inpatient = df.loc[patient_id]["number_inpatient"] if "number_inpatient" in df.columns else "N/A"
    st.markdown(f"**Previous Inpatient Admissions:** {num_prior_inpatient}")

    # ------- Patient Details -------
    st.markdown("<div class='card'><h3>Patient Details</h3></div>", unsafe_allow_html=True)
    st.write(df.loc[patient_id])

    # ------- SHAP local explanation plot -------
    st.markdown("<div class='card'><h3>Local Patient Explanation (SHAP)</h3></div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12,5))
    shap.plots.waterfall(shap_values_patient[0], max_display=12, show=False)
    st.pyplot(fig)

    # (Continue with further SHAP explanations or recommended actions as needed)
    # ------- SHAP Explanations -------
    st.markdown("<div class='card'><h3>Key Factors Affecting Risk</h3></div>", unsafe_allow_html=True)
    st.write("Impact of each attribute on the patient's predicted readmission risk:")
    high_risk_features = []

    explanations_dict = {
        "race": "Race can influence health outcomes due to genetic, socioeconomic, and access-to-care factors.",
        "gender": "Gender differences may impact disease prevalence, treatment response, and readmission likelihood.",
        "age": "Older patients often have higher readmission risk due to comorbid conditions and frailty.",
        "admission_type_id": "Emergency admissions usually indicate acute worsening of health, increasing readmission risk.",
        "discharge_disposition_id": "Discharge to rehabilitation or home care affects follow-up and monitoring, impacting readmission.",
        "admission_source_id": "How a patient was admitted, such as transfer or emergency, reflects health severity and resource use.",
        "time_in_hospital": "Longer hospital stays may indicate severe illness or complications, raising readmission risk.",
        "num_lab_procedures": "More lab procedures can reflect complex diagnostic evaluation and patient acuity.",
        "num_procedures": "Number of procedures signifies treatment intensity which can relate to readmission chances.",
        "num_medications": "Higher medication counts may indicate multiple conditions contributing to risk.",
        "number_outpatient": "Frequent outpatient visits might suggest ongoing disease management needs.",
        "number_emergency": "Emergency visits before admission can signal instability leading to readmission.",
        "number_inpatient": "Previous inpatient admissions show health complexity and risk of recurring hospitalization.",
        "diag_1": "Primary diagnosis drives treatment focus and affects likelihood of complications.",
        "diag_2": "Secondary diagnoses provide context on comorbidities elevating risk.",
        "diag_3": "Additional diagnoses add further information on patient health burden.",
        "number_diagnoses": "More diagnoses correlate with higher complexity and readmission chances.",
        "max_glu_serum": "Highest glucose serum levels can indicate poor diabetes control influencing outcomes.",
        "A1Cresult": "A1C results give long-term blood sugar control status linked to complications.",
        "metformin": "Use of metformin may reflect diabetes management impacting risk.",
        "repaglinide": "Repaglinide treatment dose and adherence impact blood sugar and readmission.",
        "nateglinide": "This medication’s use indicates treatment regimen complexity.",
        "chlorpropamide": "Use of sulfonylureas like chlorpropamide impacts risk via side effect profiles.",
        "glimepiride": "Glimepiride's effect on glucose control can influence readmission likelihood.",
        "acetohexamide": "Presence of this drug indicates specific diabetes treatment approach.",
        "glipizide": "Glipizide use may modulate diabetes management risks.",
        "glyburide": "This medication can impact hypoglycemia risk and readmission.",
        "tolbutamide": "Tolbutamide use reflects older treatment regimens with specific risk profiles.",
        "pioglitazone": "Pioglitazone affects insulin sensitivity influencing readmission risk.",
        "rosiglitazone": "Its use carries specific cardiovascular risk considerations.",
        "acarbose": "Acarbose impacts post-meal glucose spikes and management outcomes.",
        "miglitol": "Miglitol's effect on carbohydrate absorption relates to control and risk.",
        "troglitazone": "Historical medication with legacy risk concerns affecting patient history.",
        "tolazamide": "Use of this drug indicates specific diabetes treatment considerations.",
        "examide": "Presence shows a particular therapeutic approach with associated risks.",
        "citoglipton": "This diabetes drug’s use reflects treatment complexity.",
        "insulin": "Insulin therapy intensity and control effectiveness modify readmission probabilities.",
        "glyburide-metformin": "Combination treatment signifies advanced management with mixed impact on risk.",
        "glipizide-metformin": "Combination therapy affects both control and side effect risk profiles.",
        "glimepiride-pioglitazone": "Dual therapy reflects multifactorial treatment and risk.",
        "metformin-rosiglitazone": "Combination impacts multiple physiological pathways influencing outcomes.",
        "metformin-pioglitazone": "Use reflects tailored management with specific risk trade-offs.",
        "change": "Recent treatment or condition changes may destabilize patient leading to readmission.",
        "diabetesMed": "Overall diabetes medication regimen complexity influences control and risk."
    }

    high_risk_features = []
    for feature, value in zip(model_features, shap_values_patient.values[0]):
        if value > 0:
            high_risk_features.append(feature)  # Track features increasing risk
        color_class = "risk-high" if value > 0 else "risk-low"
        meaning = "increases" if value > 0 else "decreases"
        explanation = explanations_dict.get(feature, "This feature influences readmission risk.")
        st.markdown(f"""
        <div class='{color_class}'>
        <b>{feature}:</b> {meaning} the risk of readmission.<br>
        {explanation}
        </div>
        """, unsafe_allow_html=True)

    # ------------------- Recommended Actions -------------------
    st.markdown("<div class='card'><h3>Recommended Next Steps</h3></div>", unsafe_allow_html=True)
    actions = []

    if "glimepiride" in high_risk_features:
        actions += [
            "Review Glimepiride dosage and adherence carefully, as improper use may increase readmission risk.",
            "Monitor blood sugar levels frequently to assess medication effectiveness."
        ]
    if "change" in high_risk_features:
        actions += [
            "Closely monitor any recent treatment or medication changes which may destabilize condition.",
            "Keep a detailed log of symptoms and responses to new therapies."
        ]
    if "discharge_disposition_id" in high_risk_features:
        actions += [
            "Ensure comprehensive discharge planning including clear instructions on follow-up and medication.",
            "Coordinate with home care and support services as needed."
        ]
    if "admission_type_id" in high_risk_features:
        actions += [
            "Provide additional monitoring and care for emergency admissions, anticipating potential complications.",
            "Educate patients and caregivers on signs of deterioration and when to seek help."
        ]
    if "age" in high_risk_features:
        actions += [
            "Schedule age-appropriate screenings and more frequent check-ins to address comorbidities.",
            "Personalize treatment regimens considering frailty, medication tolerance, and cognition."
        ]
    if "num_lab_procedures" in high_risk_features and df.loc[patient_id]["num_lab_procedures"] > 40:
        actions.append("Review recent lab test results thoroughly and schedule timely follow-ups.")
    if "num_procedures" in high_risk_features and df.loc[patient_id]["num_procedures"] > 5:
        actions.append("Coordinate care across specialties due to complex procedures.")
    if "num_medications" in high_risk_features and df.loc[patient_id]["num_medications"] > 10:
        actions.append("Conduct medication reconciliation to prevent adverse interactions.")
    if "number_outpatient" in high_risk_features and df.loc[patient_id]["number_outpatient"] > 5:
        actions.append("Enhance outpatient education and monitoring for chronic condition stability.")
    if "number_emergency" in high_risk_features and df.loc[patient_id]["number_emergency"] > 1:
        actions.append("Develop emergency care plan and educate patient on early warning signs.")
    if "max_glu_serum" in high_risk_features and df.loc[patient_id]["max_glu_serum"] in [" >200", ">300"]:
        actions.append("Intensify glycemic control interventions with diet and medication adjustments.")
    if "A1Cresult" in high_risk_features and df.loc[patient_id]["A1Cresult"] in [">8", ">9"]:
        actions.append("Personalize lifestyle and medication plan to improve A1C control.")
    if "diabetesMed" in high_risk_features and df.loc[patient_id]["diabetesMed"] == 1:
        actions.append("Support medication adherence and monitor side effects.")

    actions += [
        "Encourage diabetes self-management education and support (diet, exercise, glucose monitoring).",
        "Promote adherence to prescribed medication regimens to improve glycemic control.",
        "Screen regularly for complications such as retinopathy, nephropathy, and neuropathy.",
        "Control cardiovascular risk factors like hypertension and hyperlipidemia.",
        "Coordinate multidisciplinary care involving physicians, nurses, dietitians, and social workers."
    ]

    st.write("Recommended steps based on patient's risk factors:")
    for step in actions:
        st.markdown(f"- {step}")


    # ------------------- Global SHAP (Sample for speed) -------------------
    st.markdown("<hr>")
    st.markdown("<div class='card'><h3>Global Feature Importance</h3></div>", unsafe_allow_html=True)
    st.write("Features most influential across a representative sample of patients:")
    sample_df = df_model.sample(n=min(500, len(df_model)), random_state=42)
    shap_values_sample = explainer(sample_df)
    fig, ax = plt.subplots(figsize=(12,5))
    shap.summary_plot(shap_values_sample, sample_df, show=False)
    st.pyplot(fig)


# ------------------- Confusion Matrix & ROC Curve Side by Side -------------------
st.markdown("<hr>")

st.markdown("<div class='card'><h3>Model Evaluation</h3></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
    cm_img = Image.open("confusion_matrix.png")
    st.image(cm_img, use_container_width=True)

with col2:
    st.markdown("<h4>ROC Curve</h4>", unsafe_allow_html=True)
    roc_img = Image.open("roc_curve.png")
    st.image(roc_img, use_container_width=True)

st.markdown("""
<div class='card'>
<h4>Detailed Explanation:</h4>
<ul>
<li><b>Confusion Matrix:</b> This table helps us understand how well the model predicts patient readmission.
    <ul>
        <li><b>True Negatives (5005):</b> Patients who were correctly predicted <i>not</i> to be readmitted.</li>
        <li><b>False Positives (322):</b> Patients who were wrongly predicted to be readmitted but actually were not (Type I error).</li>
        <li><b>False Negatives (585):</b> Patients who were wrongly predicted <i>not</i> to be readmitted but actually were (Type II error).</li>
        <li><b>True Positives (88):</b> Patients who were correctly predicted <i>to be</i> readmitted.</li>
    </ul>
    <p>This matrix helps identify types of correct and incorrect predictions that guide how reliable and safe the model is in practice.</p>
</li>
<li><b>ROC Curve (Receiver Operating Characteristic):</b> This graph shows how well the model distinguishes between patients who are readmitted and those who are not.
    <ul>
        <li><b>True Positive Rate (Y-axis):</b> The proportion of actual readmissions correctly identified by the model (also called sensitivity).</li>
        <li><b>False Positive Rate (X-axis):</b> The proportion of patients not readmitted but incorrectly predicted as readmitted.</li>
        <li><b>AUC (Area Under Curve - 0.59):</b> The overall ability of the model to discriminate—the closer to 1, the better; 0.5 is no better than random guessing.</li>
    </ul>
    <p>A model with AUC of 0.59 has a modest ability to distinguish readmitted patients from others, better than random but with room for improvement.</p>
</li>
</ul>
</div>
""", unsafe_allow_html=True)

"""
Page 1: Patient Overview & Selection
Select a patient and view their demographics, vitals, and lab summary.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.ingestion import (
    get_patient_list, load_mimic_labs, load_mimic_vitals,
    load_mimic_patients, get_patient_full_data
)

st.set_page_config(page_title="Patient Overview", page_icon="👤", layout="wide")

st.markdown(
    '<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;'
    'padding:0.75rem 1rem;font-size:0.85rem;color:#856404;margin-bottom:1rem;">'
    '⚠️ <b>DECISION-SUPPORT ONLY</b> — Not a clinical diagnosis.</div>',
    unsafe_allow_html=True
)

st.title("👤 Patient Overview")
st.markdown("Select a patient to view their ICU data and begin analysis.")

# Load patient list
@st.cache_data
def load_patients():
    return get_patient_list()

patients = load_patients()

if not patients:
    st.error("No patients found. Please ensure MIMIC-IV data is in the data/mimic/ directory.")
    st.stop()

# Patient selector
patient_options = {
    f"Patient {p['patient_id']} — {p.get('age', '?')}yo {p.get('gender', '?')} — {p.get('diagnosis', 'N/A')}": p
    for p in patients
}

selected_label = st.selectbox("Select Patient", list(patient_options.keys()))
selected_patient = patient_options[selected_label]
patient_id = selected_patient["patient_id"]

# Store in session state for other pages
st.session_state["selected_patient"] = selected_patient
st.session_state["patient_id"] = patient_id

st.markdown("---")

# Demographics Card
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Patient ID", patient_id)
with col2:
    st.metric("Age", f"{selected_patient.get('age', 'N/A')} years")
with col3:
    st.metric("Gender", selected_patient.get("gender", "N/A"))
with col4:
    st.metric("Primary Diagnosis", selected_patient.get("diagnosis", "N/A"))

st.markdown("---")

# Load full patient data (no cache — ensures fresh data per patient selection)
def load_full_data(pid):
    return get_patient_full_data(pid)

with st.spinner("Loading patient data..."):
    full_data = load_full_data(patient_id)
    # Clear any previous patient's data from other pages
    if st.session_state.get("_last_pid") != patient_id:
        for key in ["report_results", "notes", "handoff", "scores", "trends"]:
            st.session_state.pop(key, None)
        st.session_state["_last_pid"] = patient_id

st.session_state["full_data"] = full_data

# Vitals Summary
st.subheader("📊 Vital Signs")
vitals_df = full_data.get("vitals")
if vitals_df is not None and not vitals_df.empty:
    # Show latest vitals
    latest_cols = st.columns(6)
    vital_names = ["Heart Rate", "Systolic BP", "Respiratory Rate", "SpO2", "Temperature", "GCS"]
    vital_keys = ["heart_rate", "sbp", "respiratory_rate", "spo2", "temperature", "gcs"]
    vital_units = ["bpm", "mmHg", "/min", "%", "°C", "/15"]

    for i, (name, key, unit) in enumerate(zip(vital_names, vital_keys, vital_units)):
        with latest_cols[i]:
            if key in vitals_df.columns:
                valid = vitals_df[key].dropna()
                if not valid.empty:
                    latest_val = valid.iloc[-1]
                    prev_val = valid.iloc[-2] if len(valid) > 1 else None
                    delta = None
                    if prev_val is not None:
                        delta = f"{latest_val - prev_val:+.1f}"
                    st.metric(name, f"{latest_val:.1f} {unit}", delta=delta)
                else:
                    st.metric(name, "N/A")
            else:
                st.metric(name, "N/A")

    st.markdown("**Recent Vital Signs (last 10 entries)**")
    display_cols = [c for c in ["charttime", "heart_rate", "sbp", "dbp",
                                "respiratory_rate", "spo2", "temperature", "gcs"]
                    if c in vitals_df.columns]
    st.dataframe(vitals_df[display_cols].tail(10), use_container_width=True)
else:
    st.info("No vital signs data available for this patient.")

st.markdown("---")

# Labs Summary
st.subheader("🧪 Laboratory Results")
labs_df = full_data.get("labs")
if labs_df is not None and not labs_df.empty:
    # Key labs summary
    key_labs = {
        "WBC": "white blood",
        "Lactate": "lactate",
        "Creatinine": "creatinine",
        "Platelets": "platelet",
        "Bilirubin": "bilirubin",
        "BUN": "urea nitrogen",
    }
    lab_cols = st.columns(6)

    for i, (lab_name, search_term) in enumerate(key_labs.items()):
        with lab_cols[i]:
            lab_data = labs_df[labs_df["label"].str.contains(search_term, case=False, na=False)]
            if not lab_data.empty:
                latest = lab_data.iloc[-1]
                val = latest.get("valuenum", None)
                if pd.notna(val):
                    unit = latest.get("valueuom", "")
                    is_abnormal = False
                    ref_low = latest.get("ref_range_lower")
                    ref_high = latest.get("ref_range_upper")
                    if pd.notna(ref_low) and val < ref_low:
                        is_abnormal = True
                    if pd.notna(ref_high) and val > ref_high:
                        is_abnormal = True

                    display_val = f"{val:.1f} {unit}"
                    if is_abnormal:
                        st.metric(lab_name, display_val, delta="ABNORMAL", delta_color="inverse")
                    else:
                        st.metric(lab_name, display_val)
                else:
                    st.metric(lab_name, "N/A")
            else:
                st.metric(lab_name, "N/A")

    st.markdown("**Recent Lab Results (last 20)**")
    display_cols = [c for c in ["charttime", "label", "valuenum", "valueuom",
                                "ref_range_lower", "ref_range_upper"]
                    if c in labs_df.columns]
    st.dataframe(labs_df[display_cols].tail(20), use_container_width=True)
else:
    st.info("No lab data available for this patient.")

st.markdown("---")

# Diagnoses
st.subheader("📋 Diagnoses")
diagnoses = full_data.get("diagnoses")
if diagnoses is not None and not diagnoses.empty:
    st.dataframe(diagnoses, use_container_width=True)
else:
    st.info("No diagnosis data available.")

st.markdown("---")
st.caption("Data source: MIMIC-IV Clinical Database Demo v2.2 (PhysioNet)")
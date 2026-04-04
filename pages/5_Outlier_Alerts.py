"""
Page 5: Outlier Detection & Lab Error Screening
Statistical analysis to flag probable lab errors vs real clinical changes.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.utils.outlier_detector import detect_outlier, detect_outliers_in_series

st.set_page_config(page_title="Outlier Alerts", page_icon="⚠️", layout="wide")

st.markdown(
    '<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;'
    'padding:0.75rem 1rem;font-size:0.85rem;color:#856404;margin-bottom:1rem;">'
    '⚠️ <b>DECISION-SUPPORT ONLY</b> — Outlier flags suggest probable lab errors, '
    'not confirmed diagnoses. Always verify with a redraw.</div>',
    unsafe_allow_html=True
)

st.title("⚠️ Outlier Detection & Lab Error Screening")
st.markdown(
    "Identifies lab values that are statistically anomalous compared to the patient's "
    "recent 72-hour history. Outliers may indicate **lab collection errors** rather than "
    "true clinical changes."
)

if "full_data" not in st.session_state:
    st.warning("Please select a patient on the **Patient Overview** page first.")
    st.stop()

full_data = st.session_state["full_data"]
patient_info = st.session_state.get("selected_patient", {})
labs_df = full_data.get("labs")

if labs_df is None or labs_df.empty:
    st.warning("No lab data available for this patient.")
    st.stop()

st.markdown(f"**Patient {patient_info.get('patient_id', 'N/A')}** — "
            f"{patient_info.get('age', '?')}yo {patient_info.get('gender', '?')}")

st.markdown("---")

# Z-score threshold selector
z_threshold = st.slider(
    "Z-Score Threshold for Outlier Detection",
    min_value=1.5, max_value=5.0, value=3.0, step=0.5,
    help="Values beyond this many standard deviations from the recent mean are flagged."
)

st.markdown("---")

# Analyze each key lab
key_labs = {
    "WBC": {"unit": "K/uL", "normal_low": 4.0, "normal_high": 11.0},
    "Lactate": {"unit": "mmol/L", "normal_low": 0.5, "normal_high": 2.0},
    "Creatinine": {"unit": "mg/dL", "normal_low": 0.7, "normal_high": 1.3},
    "Platelets": {"unit": "K/uL", "normal_low": 150, "normal_high": 400},
    "Bilirubin": {"unit": "mg/dL", "normal_low": 0.1, "normal_high": 1.2},
    "BUN": {"unit": "mg/dL", "normal_low": 7, "normal_high": 20}
}

outlier_count = 0
total_analyzed = 0

for lab_name, lab_info in key_labs.items():
    lab_data = labs_df[labs_df["label"].str.contains(lab_name, case=False, na=False)]

    if lab_data.empty or "valuenum" not in lab_data.columns:
        continue

    valid_data = lab_data.dropna(subset=["valuenum"])
    if len(valid_data) < 3:
        continue

    total_analyzed += 1
    values = valid_data["valuenum"].tolist()
    time_col = "charttime" if "charttime" in valid_data.columns else valid_data.columns[0]
    timestamps = valid_data[time_col].tolist()

    # Check latest value for outlier
    latest_val = values[-1]
    history = values[:-1]

    if len(history) >= 2:
        result = detect_outlier(history, latest_val, z_threshold)
    else:
        result = {"is_outlier": False, "z_score": 0, "mean": np.mean(values), "std": np.std(values)}

    is_outlier = result.get("is_outlier", False)
    if is_outlier:
        outlier_count += 1

    # Create chart
    with st.expander(
        f"{'🔴 OUTLIER' if is_outlier else '🟢 Normal'} — {lab_name} "
        f"(Latest: {latest_val:.1f} {lab_info['unit']}, Z-score: {result.get('z_score', 0):.1f})",
        expanded=is_outlier
    ):
        col_chart, col_stats = st.columns([3, 1])

        with col_chart:
            fig = go.Figure()

            # Plot all values
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode="lines+markers",
                    name=lab_name,
                    line=dict(color="#0066cc", width=2),
                    marker=dict(size=8)
                )
            )

            # Highlight outlier
            if is_outlier:
                fig.add_trace(
                    go.Scatter(
                        x=[len(values) - 1],
                        y=[latest_val],
                        mode="markers",
                        name="OUTLIER",
                        marker=dict(color="red", size=15, symbol="x"),
                    )
                )

            # Normal range band
            fig.add_hrect(
                y0=lab_info["normal_low"],
                y1=lab_info["normal_high"],
                fillcolor="green",
                opacity=0.1,
                line_width=0,
                annotation_text="Normal Range"
            )

            # Mean line
            mean_val = result.get("mean", np.mean(values))
            fig.add_hline(
                y=mean_val, line_dash="dash", line_color="gray",
                annotation_text=f"Mean: {mean_val:.1f}"
            )

            # Std band
            std_val = result.get("std", np.std(values))
            if std_val > 0:
                fig.add_hrect(
                    y0=mean_val - z_threshold * std_val,
                    y1=mean_val + z_threshold * std_val,
                    fillcolor="yellow",
                    opacity=0.05,
                    line_width=1,
                    line_color="orange",
                    annotation_text=f"±{z_threshold}σ"
                )

            fig.update_layout(
                title=f"{lab_name} ({lab_info['unit']})",
                xaxis_title="Measurement #",
                yaxis_title=f"{lab_name} ({lab_info['unit']})",
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            st.metric("Latest Value", f"{latest_val:.1f}")
            st.metric("Z-Score", f"{result.get('z_score', 0):.2f}")
            st.metric("72hr Mean", f"{mean_val:.1f}")
            st.metric("72hr Std", f"{std_val:.2f}")
            st.metric("Normal Range",
                       f"{lab_info['normal_low']}-{lab_info['normal_high']}")

        if is_outlier:
            st.error(
                f"**⚠️ OUTLIER DETECTED:** {lab_name} value of {latest_val:.1f} "
                f"is {abs(result.get('z_score', 0)):.1f} standard deviations from the "
                f"72-hour mean of {mean_val:.1f}. "
                f"\n\n**Recommendation:** This may represent a lab collection or processing error. "
                f"**Recommend redraw to confirm** before updating the clinical assessment. "
                f"Do NOT alter the diagnosis based on this single value until confirmed."
            )

st.markdown("---")

# Summary
st.markdown("### 📊 Analysis Summary")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.metric("Labs Analyzed", total_analyzed)
with col_s2:
    st.metric("Outliers Detected", outlier_count)
with col_s3:
    st.metric("Z-Threshold", f"±{z_threshold}σ")

if outlier_count > 0:
    st.warning(
        f"**{outlier_count} outlier(s) detected.** These values deviate significantly "
        f"from the patient's recent lab history and may indicate lab errors. "
        f"The Chief Synthesis Agent will NOT update the diagnosis based on these values "
        f"until a confirmed redraw is received."
    )
else:
    st.success("All recent lab values are within expected statistical range. No outliers detected.")

st.markdown("---")
st.markdown(
    "**Methodology:** Outlier detection uses Z-score analysis against the patient's "
    "own 72-hour lab history. This approach detects values that are anomalous *for this "
    "specific patient*, not just outside population norms. A sudden spike that contradicts "
    "3 days of consistent data is flagged as a probable lab error per the HC01 problem statement."
)

st.markdown("---")
st.caption("Statistical outlier detection per HC01 requirement | Z-score threshold: ±3.0σ default")

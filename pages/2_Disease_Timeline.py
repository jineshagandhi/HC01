"""
Page 2: Disease Progression Timeline
Interactive Plotly timeline showing vitals, labs, scores, and alerts over time.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agents.temporal_mapper import (
    calculate_scores_over_time, detect_all_trends, build_timeline
)
from backend.orchestrator import _normalize_vitals_for_agents, _normalize_labs_for_agents

st.set_page_config(page_title="Disease Timeline", page_icon="📊", layout="wide")

st.markdown(
    '<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;'
    'padding:0.75rem 1rem;font-size:0.85rem;color:#856404;margin-bottom:1rem;">'
    '⚠️ <b>DECISION-SUPPORT ONLY</b> — Not a clinical diagnosis.</div>',
    unsafe_allow_html=True
)

st.title("📊 Disease Progression Timeline")

# Check if patient is selected
if "full_data" not in st.session_state:
    st.warning("Please select a patient on the **Patient Overview** page first.")
    st.stop()

full_data = st.session_state["full_data"]
patient_info = st.session_state.get("selected_patient", {})
vitals_df = full_data.get("vitals")
labs_df = full_data.get("labs")

if vitals_df is None or vitals_df.empty:
    st.warning("No vitals data available for this patient.")
    st.stop()

st.markdown(f"**Patient {patient_info.get('patient_id', 'N/A')}** — "
            f"{patient_info.get('age', '?')}yo {patient_info.get('gender', '?')} — "
            f"{patient_info.get('diagnosis', 'N/A')}")

st.markdown("---")

# Normalize data for agents (MIMIC columns -> agent-expected columns)
vitals_norm = _normalize_vitals_for_agents(vitals_df)
labs_norm = _normalize_labs_for_agents(labs_df)

# Calculate scores
with st.spinner("Agent 2 (Temporal Mapper) analyzing trends..."):
    scores = calculate_scores_over_time(vitals_norm, labs_norm)
    trends = detect_all_trends(labs_norm)

# Store for other pages
st.session_state["scores"] = scores
st.session_state["trends"] = trends

# === SOFA/qSOFA Score Timeline ===
st.subheader("🎯 SOFA & qSOFA Score Progression")

if scores:
    score_df = pd.DataFrame(scores)
    if "timestamp" in score_df.columns:
        fig_scores = make_subplots(
            rows=2, cols=1,
            subplot_titles=("SOFA Score (0-24)", "qSOFA Score (0-3)"),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )

        # SOFA Score
        fig_scores.add_trace(
            go.Scatter(
                x=score_df["timestamp"],
                y=score_df["sofa_total"],
                mode="lines+markers",
                name="SOFA Total",
                line=dict(color="#dc3545", width=3),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(220,53,69,0.1)"
            ),
            row=1, col=1
        )

        # SOFA threshold line
        fig_scores.add_hline(
            y=2, line_dash="dash", line_color="orange",
            annotation_text="Sepsis Threshold (SOFA ≥ 2)",
            row=1, col=1
        )

        # qSOFA Score
        fig_scores.add_trace(
            go.Scatter(
                x=score_df["timestamp"],
                y=score_df["qsofa_total"],
                mode="lines+markers",
                name="qSOFA Total",
                line=dict(color="#fd7e14", width=3),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(253,126,20,0.1)"
            ),
            row=2, col=1
        )

        # qSOFA threshold
        fig_scores.add_hline(
            y=2, line_dash="dash", line_color="red",
            annotation_text="High Risk (qSOFA ≥ 2)",
            row=2, col=1
        )

        fig_scores.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig_scores, use_container_width=True)

        # Score summary
        if len(scores) >= 2:
            first_sofa = scores[0].get("sofa_total", 0)
            last_sofa = scores[-1].get("sofa_total", 0)
            sofa_change = last_sofa - first_sofa
            direction = "↑ Worsening" if sofa_change > 0 else "↓ Improving" if sofa_change < 0 else "→ Stable"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latest SOFA", last_sofa, delta=f"{sofa_change:+d} ({direction})",
                          delta_color="inverse" if sofa_change > 0 else "normal")
            with col2:
                st.metric("Latest qSOFA", scores[-1].get("qsofa_total", 0))
            with col3:
                st.metric("Latest SIRS", scores[-1].get("sirs_total", 0))
else:
    st.info("Insufficient data to calculate clinical scores.")

st.markdown("---")

# === Vital Signs Timeline ===
st.subheader("💓 Vital Signs Over Time")

if vitals_df is not None and not vitals_df.empty:
    time_col = "charttime" if "charttime" in vitals_df.columns else vitals_df.columns[0]

    vital_options = {
        "Heart Rate": "heart_rate",
        "Systolic BP": "sbp",
        "Diastolic BP": "dbp",
        "Respiratory Rate": "respiratory_rate",
        "SpO2": "spo2",
        "Temperature": "temperature"
    }

    selected_vitals = st.multiselect(
        "Select vitals to display",
        list(vital_options.keys()),
        default=["Heart Rate", "Systolic BP", "Respiratory Rate"]
    )

    if selected_vitals:
        fig_vitals = go.Figure()
        colors = ["#dc3545", "#0066cc", "#28a745", "#fd7e14", "#6f42c1", "#e83e8c"]

        for i, vital_name in enumerate(selected_vitals):
            col_name = vital_options[vital_name]
            if col_name in vitals_df.columns:
                valid_data = vitals_df.dropna(subset=[col_name])
                if not valid_data.empty:
                    fig_vitals.add_trace(
                        go.Scatter(
                            x=valid_data[time_col],
                            y=valid_data[col_name],
                            mode="lines+markers",
                            name=vital_name,
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=5)
                        )
                    )

        fig_vitals.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_vitals, use_container_width=True)

st.markdown("---")

# === Lab Trends ===
st.subheader("🧪 Key Lab Trends")

if labs_df is not None and not labs_df.empty:
    # Use search terms that match actual MIMIC-IV labels
    key_labs = {
        "WBC": ["wbc", "white blood"],
        "Lactate": ["lactate"],
        "Creatinine": ["creatinine"],
        "Platelets": ["platelet"],
        "Bilirubin": ["bilirubin"],
    }

    lab_cols = st.columns(len(key_labs))
    for i, (lab_name, search_terms) in enumerate(key_labs.items()):
        with lab_cols[i]:
            lab_data = labs_df[labs_df["label"].str.lower().apply(
                lambda x: any(term in str(x).lower() for term in search_terms)
            )]
            if not lab_data.empty and "valuenum" in lab_data.columns:
                time_col = "charttime" if "charttime" in lab_data.columns else lab_data.columns[0]
                valid = lab_data.dropna(subset=["valuenum"])
                if not valid.empty:
                    fig_lab = go.Figure()
                    fig_lab.add_trace(
                        go.Scatter(
                            x=valid[time_col],
                            y=valid["valuenum"],
                            mode="lines+markers",
                            line=dict(width=2),
                            marker=dict(size=6)
                        )
                    )
                    fig_lab.update_layout(
                        title=lab_name,
                        height=250,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=False
                    )
                    st.plotly_chart(fig_lab, use_container_width=True)
                else:
                    st.caption(f"{lab_name}: No data")
            else:
                st.caption(f"{lab_name}: No data")

    # Trend summary
    if trends:
        st.markdown("#### Trend Analysis")
        for trend in trends:
            icon = "🔴" if trend.get("is_concerning") else "🟢"
            direction = trend.get("trend", "stable")
            arrow = "↑" if direction == "rising" else "↓" if direction == "falling" else "→"
            st.markdown(
                f"{icon} **{trend.get('lab_name', 'Unknown')}**: "
                f"{arrow} {direction.capitalize()} — {trend.get('description', '')}"
            )
else:
    st.info("No lab data available.")

st.markdown("---")
st.caption("Data source: MIMIC-IV Clinical Database Demo v2.2 | Scores calculated per Sepsis-3 definitions")
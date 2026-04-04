"""
Page 4: Shift Handoff Report — KILLER WOW FEATURE
AI-generated shift change summary for incoming clinicians.
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.orchestrator import run_shift_handoff

st.set_page_config(page_title="Shift Handoff", page_icon="🔄", layout="wide")

st.markdown(
    '<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;'
    'padding:0.75rem 1rem;font-size:0.85rem;color:#856404;margin-bottom:1rem;">'
    '⚠️ <b>AI-GENERATED HANDOFF AID</b> — Must be verified against the patient chart.</div>',
    unsafe_allow_html=True
)

st.title("🔄 Shift Handoff Report")
st.markdown("One-click AI-generated shift change summary to ensure safe transitions of care.")

if "report_results" not in st.session_state:
    st.warning("Please generate a **Risk Report** first (Page 3), then come here for the handoff.")
    st.stop()

patient_info = st.session_state.get("selected_patient", {})
patient_id = patient_info.get("patient_id", "Unknown")
report = st.session_state["report_results"].get("report", {})

st.markdown(f"**Patient {patient_id}** — "
            f"{patient_info.get('age', '?')}yo {patient_info.get('gender', '?')} — "
            f"{patient_info.get('diagnosis', 'N/A')}")

st.markdown("---")

# Shift selector
col1, col2 = st.columns(2)
with col1:
    current_shift = st.selectbox("Current (Outgoing) Shift", ["Day", "Night"])
with col2:
    incoming_shift = "Night" if current_shift == "Day" else "Day"
    st.markdown(f"**Incoming Shift:** {incoming_shift}")

# Generate Handoff
if st.button("🔄 Generate Shift Handoff Report", type="primary", use_container_width=True):
    with st.spinner("Generating handoff report..."):
        handoff = run_shift_handoff(report, patient_info, current_shift)
    st.session_state["handoff"] = handoff

if "handoff" in st.session_state:
    handoff = st.session_state["handoff"]

    # Header
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;'
        f'padding:1.5rem;border-radius:10px;text-align:center;margin:1rem 0;">'
        f'<h2 style="margin:0;color:white;">SHIFT HANDOFF REPORT</h2>'
        f'<p style="margin:0.5rem 0 0;color:#a0a0a0;">Patient {patient_id} | '
        f'{current_shift} → {incoming_shift} | '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Patient Summary
    st.markdown("### 📝 Patient Summary")
    st.info(handoff.get("patient_summary", "No summary available."))

    # Two-column layout
    col_left, col_right = st.columns(2)

    with col_left:
        # Active Problems
        st.markdown("### 🏥 Active Problems")
        problems = handoff.get("active_problems", [])
        if problems:
            for prob in problems:
                st.markdown(f"- ⚠️ {prob}")
        else:
            st.markdown("- No active problems listed")

        # Recent Changes
        st.markdown("### 📊 Recent Changes (Last 12 Hours)")
        changes = handoff.get("recent_changes", [])
        if changes:
            for change in changes:
                st.markdown(f"- 📌 {change}")
        else:
            st.markdown("- No significant changes")

    with col_right:
        # Critical Alerts
        st.markdown("### 🔴 Critical Alerts")
        alerts = handoff.get("critical_alerts", [])
        if alerts:
            for alert in alerts:
                st.error(f"🚨 {alert}")
        else:
            st.success("No critical alerts")

        # Pending Actions
        st.markdown("### ☑️ Pending Actions")
        actions = handoff.get("pending_actions", [])
        if actions:
            for action in actions:
                st.markdown(f"- ☐ {action}")
        else:
            st.markdown("- No pending actions")

    st.markdown("---")

    # Medications & Risk
    col_m, col_r = st.columns(2)

    with col_m:
        st.markdown("### 💊 Medications Notes")
        st.markdown(handoff.get("current_medications_notes", "No medication notes available."))

    with col_r:
        st.markdown("### 🎯 Risk Assessment")
        risk_text = handoff.get("risk_assessment", "No risk assessment available.")
        st.markdown(risk_text)

    st.markdown("---")

    # SOFA Trend
    st.markdown("### 📈 SOFA Score Trend")
    sofa_trend = handoff.get("sofa_trend", "No SOFA trend data available.")
    st.markdown(f"**{sofa_trend}**")

    st.markdown("---")

    # Safety caveat
    st.markdown(
        '<div style="background:#fff3cd;border:2px solid #ffc107;border-radius:8px;'
        'padding:1rem;margin:1rem 0;">'
        f'<b>⚠️ {handoff.get("safety_caveat", "AI-GENERATED HANDOFF AID — Verify against chart.")}</b>'
        '</div>',
        unsafe_allow_html=True
    )

    # Print-friendly version
    st.markdown("---")
    with st.expander("📄 Print-Friendly Text Version"):
        text_report = f"""
{'='*60}
   SHIFT HANDOFF REPORT — Patient {patient_id}
   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | {current_shift} → {incoming_shift}
{'='*60}

PATIENT SUMMARY:
{handoff.get('patient_summary', 'N/A')}

ACTIVE PROBLEMS:
{chr(10).join('  - ' + p for p in handoff.get('active_problems', ['None']))}

CRITICAL ALERTS:
{chr(10).join('  🔴 ' + a for a in handoff.get('critical_alerts', ['None']))}

RECENT CHANGES (Last 12 Hours):
{chr(10).join('  - ' + c for c in handoff.get('recent_changes', ['None']))}

PENDING ACTIONS:
{chr(10).join('  ☐ ' + a for a in handoff.get('pending_actions', ['None']))}

MEDICATIONS NOTES:
{handoff.get('current_medications_notes', 'N/A')}

RISK ASSESSMENT:
{handoff.get('risk_assessment', 'N/A')}

SOFA TREND: {handoff.get('sofa_trend', 'N/A')}

⚠️ AI-GENERATED HANDOFF AID — Verify against patient chart.
{'='*60}
"""
        st.code(text_report, language="text")

else:
    st.markdown("""
    ### How to Generate a Handoff Report
    1. First, generate a **Risk Report** on Page 3
    2. Select the current shift above
    3. Click **Generate Shift Handoff Report**
    4. Share the report with the incoming team
    """)

st.markdown("---")
st.caption("This feature addresses ICU shift-change risks identified in the HC01 problem statement.")

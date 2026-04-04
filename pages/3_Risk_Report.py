"""
Page 3: Diagnostic Risk Report
Full AI-generated risk assessment with guideline citations.
Structured card layout with PDF download.
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
from io import BytesIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.orchestrator import run_full_pipeline
from backend.data.note_generator import generate_clinical_notes

st.set_page_config(page_title="Risk Report", page_icon="📋", layout="wide")

# Custom CSS for structured report
st.markdown("""
<style>
    .report-card {
        background: white;
        border: 1px solid #e0e7ef;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .report-card h4 {
        margin-top: 0;
        color: #1a1a2e;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .risk-banner-critical {
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white; padding: 1.2rem; border-radius: 12px;
        text-align: center; margin: 0.5rem 0 1rem;
    }
    .risk-banner-high {
        background: linear-gradient(135deg, #fd7e14, #e8690a);
        color: white; padding: 1.2rem; border-radius: 12px;
        text-align: center; margin: 0.5rem 0 1rem;
    }
    .risk-banner-moderate {
        background: linear-gradient(135deg, #ffc107, #e0a800);
        color: #1a1a2e; padding: 1.2rem; border-radius: 12px;
        text-align: center; margin: 0.5rem 0 1rem;
    }
    .risk-banner-low {
        background: linear-gradient(135deg, #28a745, #218838);
        color: white; padding: 1.2rem; border-radius: 12px;
        text-align: center; margin: 0.5rem 0 1rem;
    }
    .score-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .score-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .score-label {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
    }
    .evidence-item {
        background: #f1f5f9;
        border-left: 3px solid #0d6efd;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
    }
    .action-item {
        background: #e8f5e9;
        border-left: 3px solid #28a745;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
    }
    .citation-item {
        background: #e8f4fd;
        border-left: 3px solid #0099cc;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        font-style: italic;
    }
    .outlier-card {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .agent-status-ok {
        background: #e8f5e9;
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        padding: 0.6rem;
        text-align: center;
        font-size: 0.85rem;
    }
    .agent-status-err {
        background: #ffebee;
        border: 1px solid #ffcdd2;
        border-radius: 8px;
        padding: 0.6rem;
        text-align: center;
        font-size: 0.85rem;
    }
    .safety-footer {
        background: #fff8e1;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;'
    'padding:0.6rem 1rem;font-size:0.83rem;color:#856404;margin-bottom:0.8rem;">'
    '⚠️ <b>DECISION-SUPPORT ONLY</b> — Not a clinical diagnosis.</div>',
    unsafe_allow_html=True
)

st.title("📋 Diagnostic Risk Report")

if "full_data" not in st.session_state:
    st.warning("Please select a patient on the **Patient Overview** page first.")
    st.stop()

full_data = st.session_state["full_data"]
patient_info = st.session_state.get("selected_patient", {})
patient_id = patient_info.get("patient_id", "Unknown")

st.markdown(f"**Patient {patient_id}** — "
            f"{patient_info.get('age', '?')}yo {patient_info.get('gender', '?')} — "
            f"{patient_info.get('diagnosis', 'N/A')}")

st.markdown("---")

# Generate Report Button
if st.button("🧠 Generate Diagnostic Risk Report", type="primary", use_container_width=True):
    vitals_df = full_data.get("vitals")
    labs_df = full_data.get("labs")

    with st.spinner("Agent 1: Generating and parsing clinical notes..."):
        notes = generate_clinical_notes(
            patient_id=patient_id,
            vitals_df=vitals_df,
            labs_df=labs_df,
            diagnosis=patient_info.get("diagnosis", "Unknown")
        )

    progress = st.progress(0, text="Starting multi-agent pipeline...")
    progress.progress(15, text="🔍 Agent 1: Parsing clinical notes...")
    progress.progress(35, text="📊 Agent 2: Calculating SOFA/qSOFA & analyzing trends...")
    progress.progress(55, text="📚 Agent 3: Retrieving clinical guidelines...")

    with st.spinner("Running all agents..."):
        results = run_full_pipeline(
            patient_info=patient_info,
            vitals_df=vitals_df,
            labs_df=labs_df,
            notes=notes
        )

    progress.progress(80, text="🧠 Agent 4: Synthesizing final report...")
    progress.progress(100, text="✅ Report complete!")

    st.session_state["report_results"] = results
    st.session_state["notes"] = notes

# Display report if available
if "report_results" in st.session_state:
    results = st.session_state["report_results"]
    report = results.get("report", {})

    risk_level = report.get("risk_level", "UNKNOWN")
    risk_class = risk_level.lower() if risk_level in ["CRITICAL", "HIGH", "MODERATE", "LOW"] else "moderate"
    risk_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}
    icon = risk_icons.get(risk_level, "⚪")

    # === DIAGNOSIS HOLD BANNER (Twist 2) ===
    diagnosis_hold = report.get("diagnosis_hold", False)
    diagnosis_hold_details = report.get("diagnosis_hold_details", [])

    if diagnosis_hold:
        hold_labs = ", ".join(d.get("lab_name", "Unknown") for d in diagnosis_hold_details)
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#d32f2f,#b71c1c);color:white;'
            f'padding:1.2rem 1.5rem;border-radius:12px;margin-bottom:1rem;">'
            f'<h3 style="margin:0 0 0.5rem;color:white;">DIAGNOSIS HOLD — Awaiting Confirmed Redraw</h3>'
            f'<p style="margin:0;font-size:0.95rem;">'
            f'Anomalous lab result(s) detected for <b>{hold_labs}</b> that contradict '
            f'72 hours of consistent prior data. The Chief Synthesis Agent has <b>REFUSED</b> '
            f'to revise the diagnosis until a confirmed redraw is received. '
            f'These values have been <b>excluded</b> from the risk calculation below.</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        # Show individual hold details
        for detail in diagnosis_hold_details:
            st.markdown(
                f'<div style="background:#fff3e0;border-left:4px solid #ff9800;'
                f'padding:0.8rem 1rem;border-radius:0 8px 8px 0;margin-bottom:0.5rem;'
                f'font-size:0.9rem;">'
                f'<b>{detail.get("lab_name", "")}</b>: {detail.get("reason", "")}'
                f'<br><span style="color:#e65100;">Status: {detail.get("status", "AWAITING_REDRAW")}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # === DATA COMPLETENESS WARNING ===
    data_completeness = report.get("data_completeness", {})
    if data_completeness.get("is_incomplete", False):
        missing = data_completeness.get("missing_critical", [])
        pct = data_completeness.get("completeness_pct", 0)
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#e65100,#bf360c);color:white;'
            f'padding:1rem 1.5rem;border-radius:12px;margin-bottom:1rem;">'
            f'<h3 style="margin:0 0 0.4rem;color:white;">⚠️ Incomplete Lab Data — Scores May Underestimate Risk</h3>'
            f'<p style="margin:0;font-size:0.92rem;">'
            f'<b>Missing critical parameters:</b> {", ".join(missing)}<br>'
            f'Data completeness: <b>{pct:.0f}%</b> of key organ-system labs available. '
            f'SOFA/qSOFA scores are unreliable when critical labs are absent. '
            f'The risk level below has been adjusted upward to account for this uncertainty.</p>'
            f'</div>',
            unsafe_allow_html=True
        )

    # === RISK LEVEL BANNER ===
    risk_suffix = ""
    if diagnosis_hold:
        risk_suffix = "  [OUTLIER-ADJUSTED]"
    elif data_completeness.get("is_incomplete", False):
        risk_suffix = "  [DATA-ADJUSTED]"
    st.markdown(
        f'<div class="risk-banner-{risk_class}">'
        f'<h2 style="margin:0;color:{"white" if risk_class != "moderate" else "#1a1a2e"};">'
        f'{icon} Overall Risk Level: {risk_level}{risk_suffix}</h2>'
        f'</div>',
        unsafe_allow_html=True
    )

    # === EXECUTIVE SUMMARY CARD ===
    st.markdown(
        f'<div class="report-card">'
        f'<h4>📝 Executive Summary</h4>'
        f'<p style="font-size:0.95rem;color:#333;">{report.get("summary", "No summary available.")}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    # === CLINICAL SCORES CARD ===
    sofa = report.get("sofa_score", {})
    qsofa = report.get("qsofa_score", {})
    sofa_total = sofa.get("total", 0) if isinstance(sofa, dict) else 0
    qsofa_total = qsofa.get("total", 0) if isinstance(qsofa, dict) else 0

    st.markdown('<div class="report-card"><h4>📊 Clinical Scores</h4>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(
            f'<div class="score-box"><div class="score-value">{sofa_total}/24</div>'
            f'<div class="score-label">SOFA Score</div></div>',
            unsafe_allow_html=True
        )
    with sc2:
        st.markdown(
            f'<div class="score-box"><div class="score-value">{qsofa_total}/3</div>'
            f'<div class="score-label">qSOFA Score</div></div>',
            unsafe_allow_html=True
        )
    with sc3:
        sofa_interp = "Organ Dysfunction" if sofa_total >= 2 else "Normal"
        st.markdown(
            f'<div class="score-box"><div class="score-value" style="font-size:1.2rem;">{sofa_interp}</div>'
            f'<div class="score-label">SOFA Interpretation</div></div>',
            unsafe_allow_html=True
        )
    with sc4:
        st.markdown(
            f'<div class="score-box"><div class="score-value" style="font-size:1.2rem;">'
            f'{datetime.now().strftime("%H:%M")}</div>'
            f'<div class="score-label">Generated At</div></div>',
            unsafe_allow_html=True
        )

    # SOFA components
    if isinstance(sofa, dict):
        components = sofa.get("components", {})
        if components:
            comp_cols = st.columns(6)
            comp_names = ["Respiration", "Coagulation", "Liver", "Cardiovascular", "CNS", "Renal"]
            for i, name in enumerate(comp_names):
                with comp_cols[i]:
                    val = components.get(name.lower(), components.get(name, "?"))
                    st.caption(f"{name}: **{val}/4**")
    st.markdown('</div>', unsafe_allow_html=True)

    # === RISK FLAGS CARD ===
    risk_flags = report.get("risk_flags", [])
    st.markdown('<div class="report-card"><h4>🚩 Risk Flags & Recommendations</h4>', unsafe_allow_html=True)

    if risk_flags:
        for flag in risk_flags:
            if isinstance(flag, dict):
                flag_level = flag.get("risk_level", "MODERATE")
                flag_icon = risk_icons.get(flag_level, "⚪")

                with st.expander(
                    f"{flag_icon} {flag.get('condition', 'Unknown')} — {flag_level} "
                    f"(Confidence: {flag.get('confidence', 0):.0%})",
                    expanded=(flag_level in ["CRITICAL", "HIGH"])
                ):
                    # Evidence
                    evidence = flag.get("evidence", [])
                    if evidence:
                        st.markdown("**Evidence:**")
                        for ev in evidence:
                            st.markdown(f'<div class="evidence-item">📌 {ev}</div>', unsafe_allow_html=True)

                    # Actions
                    actions = flag.get("recommended_actions", [])
                    if actions:
                        st.markdown("**Recommended Actions:**")
                        for action in actions:
                            st.markdown(f'<div class="action-item">✅ {action}</div>', unsafe_allow_html=True)

                    # Citations
                    citations = flag.get("guideline_citations", [])
                    if citations:
                        st.markdown("**Guideline Citations:**")
                        for cite in citations:
                            if isinstance(cite, dict):
                                st.markdown(
                                    f'<div class="citation-item">📚 <b>{cite.get("guideline_name", "")}</b>: '
                                    f'"{cite.get("text", "")}"</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(f'<div class="citation-item">📚 {cite}</div>', unsafe_allow_html=True)

                    # Confidence bar
                    confidence = flag.get("confidence", 0)
                    if confidence and 0 < confidence <= 1:
                        st.progress(confidence, text=f"Confidence: {confidence:.0%}")
            else:
                st.markdown(f"- {flag}")
    else:
        st.success("No significant risk flags detected.")
    st.markdown('</div>', unsafe_allow_html=True)

    # === OUTLIER DETECTION CARD ===
    outliers = report.get("outlier_flags", [])
    st.markdown('<div class="report-card"><h4>⚠️ Outlier Detection (Lab Error Screening)</h4>', unsafe_allow_html=True)
    if outliers:
        for outlier in outliers:
            if isinstance(outlier, dict):
                st.markdown(
                    f'<div class="outlier-card">'
                    f'<b>🔬 {outlier.get("lab_name", "Unknown")}</b> — '
                    f'Value: <b>{outlier.get("flagged_value", "N/A")}</b> | '
                    f'Z-score: <b>{outlier.get("z_score", 0):.1f}</b><br>'
                    f'<span style="font-size:0.85rem;color:#856404;">'
                    f'{outlier.get("recommendation", "Recommend redraw to confirm.")}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.success("No statistical outliers detected in recent lab values.")
    st.markdown('</div>', unsafe_allow_html=True)

    # === DISEASE PROGRESSION CARD ===
    st.markdown('<div class="report-card"><h4>📈 Disease Progression Summary</h4>', unsafe_allow_html=True)
    progression = report.get("disease_progression", "")
    if progression:
        st.markdown(f'<p style="font-size:0.92rem;color:#333;line-height:1.6;">{progression}</p>', unsafe_allow_html=True)
    else:
        agent2 = results.get("agent2", {})
        st.markdown(agent2.get("disease_progression", "No progression data available."))
    st.markdown('</div>', unsafe_allow_html=True)

    # === AGENT STATUS CARD ===
    st.markdown('<div class="report-card"><h4>🤖 Agent Pipeline Status</h4>', unsafe_allow_html=True)
    agent_cols = st.columns(4)
    agent_names = ["Note Parser", "Temporal Mapper", "Guideline RAG", "Chief Synthesis"]
    agent_keys = ["agent1", "agent2", "agent3", "report"]
    agent_icons = ["🔍", "📊", "📚", "🧠"]

    for i, (name, key, ag_icon) in enumerate(zip(agent_names, agent_keys, agent_icons)):
        with agent_cols[i]:
            agent_data = results.get(key, {})
            status = agent_data.get("status", "success") if isinstance(agent_data, dict) else "success"
            if status == "success":
                st.markdown(
                    f'<div class="agent-status-ok">{ag_icon} ✅<br><b>{name}</b></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="agent-status-err">{ag_icon} ❌<br><b>{name}</b></div>',
                    unsafe_allow_html=True
                )
    st.markdown('</div>', unsafe_allow_html=True)

    # === SAFETY FOOTER ===
    st.markdown(
        f'<div class="safety-footer">'
        f'<b>⚠️ IMPORTANT SAFETY NOTICE</b><br>'
        f'{report.get("safety_caveat", "DECISION-SUPPORT ONLY.")}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # === PDF DOWNLOAD ===
    def generate_pdf():
        """Generate a professional PDF report."""
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # Title
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(26, 26, 46)
        pdf.cell(0, 12, "ICU DIAGNOSTIC RISK REPORT", ln=True, align="C")
        pdf.ln(2)

        # Patient info bar
        pdf.set_fill_color(240, 240, 245)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(80, 80, 80)
        info_text = (
            f"Patient: {patient_id}  |  "
            f"{patient_info.get('age', '?')}yo {patient_info.get('gender', '?')}  |  "
            f"Diagnosis: {patient_info.get('diagnosis', 'N/A')}  |  "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        pdf.cell(0, 8, info_text, ln=True, fill=True, align="C")
        pdf.ln(6)

        # Risk Level
        risk_bg = {"CRITICAL": (220, 53, 69), "HIGH": (253, 126, 20),
                   "MODERATE": (255, 193, 7), "LOW": (40, 167, 69)}
        bg = risk_bg.get(risk_level, (108, 117, 125))
        pdf.set_fill_color(*bg)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 12, f"  RISK LEVEL: {risk_level}", ln=True, fill=True)
        pdf.ln(4)

        # Scores
        pdf.set_text_color(26, 26, 46)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"SOFA Score: {sofa_total}/24    |    qSOFA Score: {qsofa_total}/3", ln=True)
        pdf.ln(3)

        # Section helper
        def section_title(title):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(26, 26, 46)
            pdf.cell(0, 8, title, ln=True)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)

        def body_text(text):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(51, 51, 51)
            # Handle encoding issues
            safe_text = text.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, safe_text)

        # Executive Summary
        section_title("EXECUTIVE SUMMARY")
        body_text(report.get("summary", "N/A"))

        # Risk Flags
        section_title("RISK FLAGS")
        for flag in risk_flags:
            if isinstance(flag, dict):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(26, 26, 46)
                condition = flag.get("condition", "Unknown")
                safe_cond = condition.encode('latin-1', 'replace').decode('latin-1')
                pdf.cell(0, 6, f"[{flag.get('risk_level', '')}] {safe_cond} (Confidence: {flag.get('confidence', 0):.0%})", ln=True)

                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(80, 80, 80)
                for ev in flag.get("evidence", []):
                    safe_ev = str(ev).encode('latin-1', 'replace').decode('latin-1')
                    pdf.cell(0, 5, f"    Evidence: {safe_ev}", ln=True)
                for action in flag.get("recommended_actions", []):
                    safe_act = str(action).encode('latin-1', 'replace').decode('latin-1')
                    pdf.cell(0, 5, f"    Action: {safe_act}", ln=True)
                for cite in flag.get("guideline_citations", []):
                    if isinstance(cite, dict):
                        cite_text = f"{cite.get('guideline_name', '')}: {cite.get('text', '')}"
                        safe_cite = cite_text.encode('latin-1', 'replace').decode('latin-1')
                        pdf.cell(0, 5, f"    Citation: {safe_cite[:100]}", ln=True)
                pdf.ln(2)

        # Outlier Flags
        section_title("OUTLIER DETECTION")
        if outliers:
            for o in outliers:
                if isinstance(o, dict):
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.cell(0, 6, f"{o.get('lab_name', '')}: Value={o.get('flagged_value', '')} (Z-score: {o.get('z_score', 0):.1f})", ln=True)
                    pdf.set_font("Helvetica", "", 9)
                    rec = str(o.get("recommendation", "")).encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(0, 5, f"    {rec}")
        else:
            body_text("No statistical outliers detected.")

        # Disease Progression
        section_title("DISEASE PROGRESSION")
        body_text(report.get("disease_progression", "N/A"))

        # Safety Disclaimer
        pdf.ln(6)
        pdf.set_fill_color(255, 248, 225)
        pdf.set_text_color(133, 100, 4)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, "DISCLAIMER: DECISION-SUPPORT ONLY", ln=True, fill=True)
        pdf.set_font("Helvetica", "", 8)
        caveat = report.get("safety_caveat", "").encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 4, caveat, fill=True)

        return bytes(pdf.output())

    pdf_bytes = generate_pdf()

    col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 2])
    with col_dl2:
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"ICU_Risk_Report_Patient_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

else:
    st.markdown("""
    <div class="report-card">
    <h4>How to Generate a Report</h4>
    <ol>
        <li>Select a patient on the <b>Patient Overview</b> page</li>
        <li>Click the <b>Generate Diagnostic Risk Report</b> button above</li>
        <li>The multi-agent pipeline will analyze the patient data</li>
        <li>Results will appear with risk flags, citations, and recommendations</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Powered by: Gemini AI | MIMIC-IV Data | Surviving Sepsis Campaign 2021 Guidelines")

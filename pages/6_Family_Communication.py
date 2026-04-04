"""
Page 6: Family Communication — Compassionate Patient Summary
Jargon-free summary of the patient's last 12 hours for non-medical family members.
Translated into English and a regional language.
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.orchestrator import run_family_communication

st.set_page_config(page_title="Family Communication", page_icon="💬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .family-card {
        background: white;
        border: 1px solid #e0e7ef;
        border-radius: 14px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    }
    .family-card h4 {
        margin-top: 0;
        color: #1a1a2e;
        border-bottom: 2px solid #e8f4fd;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .family-header {
        background: linear-gradient(135deg, #2196F3, #1976D2);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .family-header h2 { margin: 0; color: white; }
    .family-header p { margin: 0.5rem 0 0; color: rgba(255,255,255,0.85); }
    .summary-text {
        font-size: 1.05rem;
        color: #333;
        line-height: 1.8;
        white-space: pre-wrap;
    }
    .lang-tab {
        background: #f8f9fa;
        border: 1px solid #e0e7ef;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 0.5rem;
    }
    .hold-banner {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    .family-safety {
        background: #e3f2fd;
        border: 2px solid #2196F3;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1.5rem;
        font-size: 0.88rem;
        color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="background:#e3f2fd;border:1px solid #2196F3;border-radius:8px;'
    'padding:0.6rem 1rem;font-size:0.83rem;color:#1565C0;margin-bottom:0.8rem;">'
    '💬 <b>FAMILY COMMUNICATION AID</b> — Simplified summary for family members. '
    'Not a clinical document.</div>',
    unsafe_allow_html=True
)

st.title("💬 Family Communication")
st.markdown(
    "A compassionate, jargon-free summary of your loved one's condition "
    "over the last 12 hours — written for family members, not medical professionals."
)

if "report_results" not in st.session_state:
    st.warning("Please generate a **Risk Report** first (Page 3), then come here for the family summary.")
    st.stop()

patient_info = st.session_state.get("selected_patient", {})
patient_id = patient_info.get("patient_id", "Unknown")
report = st.session_state["report_results"].get("report", {})

st.markdown(f"**Patient {patient_id}** — "
            f"{patient_info.get('age', '?')}yo {patient_info.get('gender', '?')} — "
            f"{patient_info.get('diagnosis', 'N/A')}")

st.markdown("---")

# Language selector
LANGUAGES = [
    "English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil",
    "Gujarati", "Kannada", "Malayalam", "Punjabi", "Odia",
    "Urdu", "Spanish", "French", "Arabic", "Mandarin",
]

col_lang, col_space = st.columns([1, 2])
with col_lang:
    regional_language = st.selectbox(
        "Select Regional Language for Translation",
        LANGUAGES,
        index=0,
        help="The summary is always generated in English. If you select a non-English language, "
             "it will also be translated into that language."
    )


# Generate button
if st.button("💬 Generate Family Summary", type="primary", use_container_width=True):
    with st.spinner("Generating compassionate family summary..."):
        family_result = run_family_communication(report, patient_info, regional_language)
    st.session_state["family_communication"] = family_result

# Display if available
if "family_communication" in st.session_state:
    fc = st.session_state["family_communication"]

    # Header
    st.markdown(
        f'<div class="family-header">'
        f'<h2>Family Update — Patient {patient_id}</h2>'
        f'<p>Generated: {fc.get("generated_at", datetime.now()).strftime("%Y-%m-%d %H:%M")} '
        f'| Language{"s: English + " + fc.get("regional_language", regional_language) if fc.get("regional_language", regional_language).lower() != "english" else ": English"}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Diagnosis hold notice (if applicable)
    if fc.get("diagnosis_hold", False):
        hold_details = fc.get("diagnosis_hold_details", [])
        labs_held = ", ".join(d.get("lab_name", "Unknown") for d in hold_details)
        st.markdown(
            f'<div class="hold-banner">'
            f'<b>Note for Family:</b> One or more recent test results '
            f'({labs_held}) looked different from previous results. '
            f'The doctors are re-running these tests to make sure they are accurate. '
            f'No changes have been made to the treatment plan based on these results.'
            f'</div>',
            unsafe_allow_html=True
        )

    # Tabbed view — show translation tab only if a non-English language was selected
    selected_lang = fc.get("regional_language", regional_language)
    is_english_only = selected_lang.lower() == "english"

    if is_english_only:
        st.markdown('<div class="family-card">', unsafe_allow_html=True)
        st.markdown("#### Dear Family,")
        st.markdown(
            f'<div class="summary-text">{fc.get("english_summary", "Summary not available.")}</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        tab_en, tab_regional = st.tabs([
            "English",
            f"{selected_lang}"
        ])

        with tab_en:
            st.markdown('<div class="family-card">', unsafe_allow_html=True)
            st.markdown("#### Dear Family,")
            st.markdown(
                f'<div class="summary-text">{fc.get("english_summary", "Summary not available.")}</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_regional:
            st.markdown('<div class="family-card">', unsafe_allow_html=True)
            st.markdown(f"#### {selected_lang} Translation")
            st.markdown(
                f'<div class="summary-text">{fc.get("regional_summary", "Translation not available.")}</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Risk level context (simplified)
    risk_level = fc.get("risk_level", "UNKNOWN")
    risk_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}.get(risk_level, "⚪")
    risk_simple = {
        "CRITICAL": "Your loved one is in serious condition and is receiving intensive care from the medical team.",
        "HIGH": "The medical team is closely monitoring your loved one as they need extra attention right now.",
        "MODERATE": "Your loved one is stable but the team is keeping a close watch to catch any changes early.",
        "LOW": "Your loved one is doing relatively well. The team continues routine monitoring.",
    }.get(risk_level, "The medical team is monitoring your loved one.")

    st.markdown(
        f'<div class="family-card">'
        f'<h4>{risk_emoji} Current Status (Simplified)</h4>'
        f'<p style="font-size:1rem;color:#333;">{risk_simple}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Safety footer
    st.markdown(
        f'<div class="family-safety">'
        f'<b>Important:</b> {fc.get("safety_caveat", "This is an AI-generated summary for family members.")}'
        f'<br><br>'
        f'If you have questions about your loved one\'s care, please speak directly '
        f'with the attending physician or nursing staff. They can provide the most '
        f'accurate and up-to-date information.'
        f'</div>',
        unsafe_allow_html=True
    )

    # Print-friendly version
    st.markdown("---")
    with st.expander("📄 Print-Friendly Version"):
        regional_block = ""
        if not is_english_only:
            regional_block = f"""
{'='*60}

{fc.get("regional_language", regional_language).upper()} SUMMARY:
{fc.get("regional_summary", "N/A")}
"""

        text_version = f"""
{'='*60}
   FAMILY UPDATE — Patient {patient_id}
   Generated: {fc.get("generated_at", datetime.now()).strftime('%Y-%m-%d %H:%M')}
{'='*60}

ENGLISH SUMMARY:
{fc.get("english_summary", "N/A")}
{regional_block}
{'='*60}

{fc.get("safety_caveat", "")}
{'='*60}
"""
        st.code(text_version, language="text")

else:
    st.markdown("""
    <div class="family-card">
    <h4>How to Generate a Family Summary</h4>
    <ol>
        <li>First, generate a <b>Risk Report</b> on Page 3</li>
        <li>Select the regional language above</li>
        <li>Click <b>Generate Family Summary</b></li>
        <li>Share the summary with the patient's family</li>
    </ol>
    <p style="color:#6c757d;font-size:0.9rem;">
    The summary will cover the last 12 hours of your loved one's care,
    written in simple, compassionate language that anyone can understand.
    </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Family Communication Aid | Powered by Gemini AI | IGNISIA ICU Risk Assistant")

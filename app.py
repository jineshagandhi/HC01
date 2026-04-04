"""
ICU Diagnostic Risk Assistant — Main Streamlit App
HC01: Agentic Diagnostic Risk Assistant for ICU Complication Detection
IGNISIA AI Hackathon 2026
"""
import streamlit as st

st.set_page_config(
    page_title="ICU Diagnostic Risk Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0 !important;
    }
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { display: none; }
    footer { display: none !important; }

    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
        line-height: 1.3;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: #6c757d;
        margin-top: 0;
        margin-bottom: 1.2rem;
        font-weight: 400;
    }
    .agent-card {
        background: white;
        border: 1px solid #e0e7ef;
        border-top: 3px solid #1a1a2e;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        text-align: center;
        min-height: 150px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .agent-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
    .agent-name {
        font-weight: 700;
        font-size: 0.92rem;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .agent-desc {
        font-size: 0.8rem;
        color: #6c757d;
        line-height: 1.4;
    }
    .safety-banner {
        background: #fff8e1;
        border: 1px solid #ffc107;
        border-left: 4px solid #ff9800;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-size: 0.83rem;
        color: #856404;
    }
    .hero-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
    }
    .hero-section h3 { color: white !important; margin-top: 0; font-size: 1.3rem; }
    .stat-box {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.8rem 0.5rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.7rem;
        font-weight: 700;
        color: #4fc3f7;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebarContent"] .stMarkdown p,
    div[data-testid="stSidebarContent"] .stMarkdown li,
    div[data-testid="stSidebarContent"] .stMarkdown h2,
    div[data-testid="stSidebarContent"] .stCaption { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🏥 ICU Risk Assistant")
    st.markdown("**Multi-Agent AI System**")
    st.markdown("---")
    st.markdown("""
    **4 AI Agents:**
    - 🔍 Note Parser
    - 📊 Temporal Lab Mapper
    - 📚 Guideline RAG
    - 🧠 Chief Synthesis

    **Features:**
    - 💬 Family Communication
    - 🛡️ Lab Outlier Diagnosis Hold
    """)
    st.markdown("---")
    st.markdown("""
    **Data Sources:**
    - MIMIC-IV (MIT/Harvard)
    - PhysioNet Sepsis Challenge
    - Surviving Sepsis Campaign 2021
    """)
    st.markdown("---")
    st.markdown(
        '<div class="safety-banner">'
        '⚠️ <b>DECISION-SUPPORT ONLY</b><br>'
        'Not a clinical diagnosis.'
        '</div>',
        unsafe_allow_html=True
    )
    st.caption("IGNISIA AI Hackathon 2026 | HC01")

# ===== MAIN PAGE =====

st.markdown('<p class="main-title">🏥 ICU Diagnostic Risk Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">'
    'Multi-Agent AI System for Early Detection of Sepsis & Organ Failure in ICU Patients'
    '</p>',
    unsafe_allow_html=True
)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h3>Why This Matters</h3>
    <p style="font-size:0.95rem; color:rgba(255,255,255,0.85); margin-bottom:1rem;">
        In ICU environments, critical deterioration patterns like early-onset sepsis are missed
        because no single physician can synthesize all data simultaneously.
        <b style="color:white;">Our 4 AI agents detect these patterns early — before it's too late.</b>
    </p>
    <div style="display:flex; gap:1rem; flex-wrap:wrap;">
        <div class="stat-box" style="flex:1; min-width:100px;">
            <div class="stat-number">4</div>
            <div class="stat-label">AI Agents</div>
        </div>
        <div class="stat-box" style="flex:1; min-width:100px;">
            <div class="stat-number">100+</div>
            <div class="stat-label">ICU Patients</div>
        </div>
        <div class="stat-box" style="flex:1; min-width:100px;">
            <div class="stat-number">40K+</div>
            <div class="stat-label">Sepsis Records</div>
        </div>
        <div class="stat-box" style="flex:1; min-width:100px;">
            <div class="stat-number">5</div>
            <div class="stat-label">Clinical Guidelines</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Agent Cards
col1, col2, col3, col4 = st.columns(4, gap="medium")
with col1:
    st.markdown("""
    <div class="agent-card">
        <div class="agent-icon">🔍</div>
        <div class="agent-name">Agent 1: Note Parser</div>
        <div class="agent-desc">Extracts symptoms, medications & conditions from clinical notes</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="agent-card">
        <div class="agent-icon">📊</div>
        <div class="agent-name">Agent 2: Temporal Mapper</div>
        <div class="agent-desc">Calculates SOFA/qSOFA scores, detects lab trends over time</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="agent-card">
        <div class="agent-icon">📚</div>
        <div class="agent-name">Agent 3: Guideline RAG</div>
        <div class="agent-desc">Retrieves & cites Surviving Sepsis Campaign 2021 guidelines</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="agent-card">
        <div class="agent-icon">🧠</div>
        <div class="agent-name">Agent 4: Chief Synthesis</div>
        <div class="agent-desc">Integrates all agents, detects outliers, generates risk report</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Safety Notice
st.markdown(
    '<div class="safety-banner">'
    '⚠️ <b>SAFETY NOTICE:</b> Clinical decision-support only. '
    'All outputs must be validated by a qualified clinician before any clinical action.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Centered Button
col_left, col_center, col_right = st.columns([2, 1, 2])
with col_center:
    if st.button("🚀 Start Patient Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/1_Patient_Overview.py")

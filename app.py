import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import qrcode
from io import BytesIO
from fpdf import FPDF
import base64
from streamlit.components.v1 import html
import time
import re
from lab_mapper import loaddata, flags
from note_parser import process_notes
from rag_agent import get_guide
from chief_agent import gen_report, family_summary   
st.set_page_config(page_title="ICU Co-Pilot",layout="wide",initial_sidebar_state="collapsed")
st.markdown("""<style>
    .dark-mode {
        background-color: #0a0f1e;
        color: #e0e0e0;
    }
    .dark-mode .stApp {
        background-color: #0a0f1e;
    }
    .vitals-card {
        background: linear-gradient(135deg, #1a1f2e, #0f1420);
        border-radius: 20px;padding: 1rem;text-align: center;box-shadow: 0 8px 16px rgba(0,0,0,0.3);border: 1px solid #2a2f3e;
    }
    .vital-value {
        font-size: 2.5rem;font-weight: bold;font-family: monospace;
    }
    .vital-label {
        font-size: 0.9rem;
        color: #aaa;
    }
    .alert-pulse {
        animation: pulse 1s infinite;
        background-color: #ff4b4b;
        border-radius: 50%;
        width: 12px;height: 12px;
        display: inline-block;margin-left: 8px;
    }
    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 0.7; }
        100% { transform: scale(1.2); opacity: 1; }
    }
    .trust-gauge {
        background: #2e3a4e;
        border-radius: 10px;
        height: 10px;width: 100%;
    }
    .trust-fill {
        background: linear-gradient(90deg, #ff4b4b, #ffa500, #00cc66);
        border-radius: 10px;
        height: 100%;width: 0%;
    }
  .risk-high { background-color: #ff4b4b; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .risk-moderate { background-color: #ffa500; color: black; padding: 4px 12px; border-radius: 20px; }
  .risk-low { background-color: #00cc66; color: black; padding: 4px 12px; border-radius: 20px; }
       .risk-unknown { background-color: #888; color: white; padding: 4px 12px; border-radius: 20px; }
</style>""", unsafe_allow_html=True)
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'injected_error' not in st.session_state:
    st.session_state.injected_error = False
if 'simulate_lactate' not in st.session_state:
    st.session_state.simulate_lactate = None
if 'simulate_wbc' not in st.session_state:
    st.session_state.simulate_wbc = None
if 'custom_note' not in st.session_state:
    st.session_state.custom_note = None
def apply(df, pid):
    if st.session_state.injected_error and pid == st.session_state.get('error_pid', ''):
        dfc = df.copy()
        dfc.iloc[-1, dfc.columns.get_loc('wbc')] = 100.0
        dfc.iloc[-1, dfc.columns.get_loc('notes')] = "LAB ERROR Detected"
        return dfc
    if st.session_state.simulate_lactate is not None and pid==st.session_state.get('sim_pid', ''):
        dfc = df.copy()
        dfc.iloc[-1, dfc.columns.get_loc('lactate')]=st.session_state.simulate_lactate
        return dfc
    if st.session_state.simulate_wbc is not None and pid==st.session_state.get('sim_pid', ''):
        dfc = df.copy()
        dfc.iloc[-1, dfc.columns.get_loc('wbc')]=st.session_state.simulate_wbc
        return dfc
    return df
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    dark_toggle = st.toggle("Dark Mode",value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()
if st.session_state.dark_mode:
    st.markdown('<style>body { background-color: #0a0f1e; color: #e0e0e0; } .stApp { background-color: #0a0f1e; }</style>', unsafe_allow_html=True)
pid = st.selectbox("Select Patient", ["P001","P002","P003","P004","P005"], key="pid")
dfo = loaddata(pid)
df = apply(dfo, pid)
df = flags(df)  
st.markdown("ICU Monitor")
last = df.iloc[-1]
wval = last['wbc']
lval = last['lactate']
cval = last['creatinine']
def color(val, low, high):
    if val > high: return "red"
    if val < low: return "orange"
    return "green"
col_v1, col_v2, col_v3, col_v4 = st.columns(4)
with col_v1:
    st.markdown(f"""<div class='vitals-card'><div class='vital-label'>WBC (x10³/µL)</div><div class='vital-value' style='color:{color(wval, 4, 11)}'>{wval}</div>
    <div class='vital-label'>normal 4-11</div></div>""", unsafe_allow_html=True)
with col_v2:
    st.markdown(f"""<div class='vitals-card'><div class='vital-label'>Lactate (mmol/L)</div><div class='vital-value' style='color:{color(lval, 0.5, 2.0)}'>{lval}</div>
    <div class='vital-label'>normal <2.0</div></div>""", unsafe_allow_html=True)
with col_v3:
    st.markdown(f"""<div class='vitals-card'><div class='vital-label'> Creatinine (mg/dL)</div><div class='vital-value' style='color:{color(cval, 0.6, 1.2)}'>{cval}</div>
    <div class='vital-label'>normal 0.6-1.2</div></div>""", unsafe_allow_html=True)
with col_v4:
    oe = df[['wbc_outlier','lactate_outlier','creatinine_outlier']].any().any()
    risk = "High" if (lval>4.0 or wval>20) and not oe else "Moderate" if (lval>2.0 or wval>12) else "Low"
    if oe:
        risk = "Unknown"
    alert = f"""<div class='vitals-card'><div class='vital-label'>"RISK STATUS</div><div class='vital-value' style='font-size:1.8rem'>{risk}</div>
        {"<div class='alert-pulse'></div>" if risk=="High" else ""}</div>"""
    st.markdown(alert,unsafe_allow_html=True)
def sc(df):
    return process_notes(df)
sym_df = sc(df)
latest = sym_df[sym_df['timestamp']==last['timestamp']]['symptom'].tolist()
query = f"WBC {wval},lactate {lval},symptoms:{', '.join(latest) if latest else 'none'}"
g1 = get_guide(query, top=2)
report = gen_report(pid, df, sym_df, g1)
left_col, right_col = st.columns([2, 1])
with left_col:
    st.subheader("Lab Trends & Outliers")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'],y=df['wbc'],mode='lines+markers',name='WBC',line=dict(color='#00cc96')))
    fig.add_trace(go.Scatter(x=df['timestamp'],y=df['lactate'],mode='lines+markers',name='Lactate',line=dict(color='#ffa500')))
    op = df[df['wbc_outlier'] == True]
    if not op.empty:
        fig.add_trace(go.Scatter(x=op['timestamp'],y=op['wbc'],mode='markers',marker=dict(color='red', size=12, symbol='x'),name='WBC Outlier', hovertemplate='Probable lab error<br>WBC=%{y}<extra></extra>'))
    ol = df[df['lactate_outlier']==True]
    if not ol.empty:
        fig.add_trace(go.Scatter(x=ol['timestamp'],y=ol['lactate'],mode='markers',marker=dict(color='red', size=12, symbol='x'), name='Lactate Outlier'))
    fig.update_layout(height=400, template='plotly_dark' if st.session_state.dark_mode else 'plotly_white', margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Risk Radar")
    wn = min(wval/30,1)
    ln = min(lval/8,1)
    sco = min(len(latest)/5,1)
    gs = g1[0]['relevance'] if g1 else 0
    rf = go.Figure(data=go.Scatterpolar(
        r=[wn,ln,sco,gs],theta=['WBC','Lactate','Symptoms','Guideline'],fill='toself',marker=dict(color='#ff4b4b')))
    rf.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=300, showlegend=False)
    st.plotly_chart(rf, use_container_width=True)
with right_col:
    st.subheader("AI Risk Report")
    st.markdown(f"Trust Score:{report['trust_score']}%")
    st.progress(report['trust_score']/100, text="")
    rc = "risk-low"
    if "High" in report['risk_level']: rc = "risk-high"
    elif "Moderate" in report['risk_level']: rc = "risk-moderate"
    elif "Unknown" in report['risk_level']: rc = "risk-unknown"
    st.markdown(f"<div class='{rc}' style='display:inline-block'>{report['risk_level']}</div>", unsafe_allow_html=True)
    if report['lab_error_refusal']:
        st.warning("Lab error refusal active risk not calculated.")
    st.markdown(f"Recommendation:{report['recommendation']}")
    with st.expander("Causal Explanation"):
        st.write(report['causal_explanation'])
    with st.expander("Second Opinion"):
        st.write(report['second_opinion'])
    with st.expander("Retrieved Guidelines"):
        for g in g1:
            st.markdown(f"**{g['source']}** (relevance {g['relevance']*100}%)")
            st.caption(g['text'])
    if "Conflict" in report['contradiction_warning']:
        st.error(report['contradiction_warning'])
    else:
        st.success("No guideline conflict")
    if report['clinical_pearl']:
        st.toast(f"Clinical Pearl: {report['clinical_pearl']}")
tc,tf = st.tabs(["Clinical Tools & Simulation", "Family Communication"])
with tc:
    st.subheader("Live Simulation & Tools")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        if st.button("Inject Lab Error", help="Adds impossible wbc value"):
            st.session_state.injected_error = True
            st.session_state.error_pid = pid
            st.rerun()
        if st.button("Reset Simulation"):
            st.session_state.injected_error = False
            st.session_state.simulate_lactate = None
            st.session_state.simulate_wbc = None
            st.rerun()
    with colB:
        sim_val = st.slider("Simulate Lactate", 0.5, 15.0, float(lval), 0.5)
        if sim_val != lval:
            st.session_state.simulate_lactate = sim_val
            st.session_state.sim_pid = pid
            st.rerun()
    with colC:
        handoff_text = f"Patient {pid}. Risk level: {report['risk_level']}. Recommendation: {report['recommendation']}. Causal explanation: {report['causal_explanation']}"
        b64 = base64.b64encode(handoff_text.encode()).decode()
        speak_js = f"""
        <audio id="speech" src="data:audio/mp3;base64,{b64}" controls style="display:none;"></audio>
        <script>
        var utterance = new SpeechSynthesisUtterance("{handoff_text.replace('"','\\"')}");
        window.speechSynthesis.speak(utterance);
        </script>
        """
        if st.button(" Speak Handoff"):
            html(speak_js, height=0)
    with colD:
        def s(txt):
            recs = {
                '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'",
                '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u00e9': 'e',
                '\u00e0': 'a', '\u00e8': 'e', '\u00f9': 'u',}
            for u, a in recs.items():
                txt = txt.replace(u, a)
            txt = re.sub(r'[^\x00-\x7F]+', '', txt)
            return txt
        if st.button("Download PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            title = f"ICU Co-Pilot Report - Patient {pid}"
            pdf.cell(200, 10, txt=s(title), ln=1, align='C')
            pdf.ln(10)
            content = f"""
    Risk Level: {report['risk_level']}
    Trust Score: {report['trust_score']}%
    Recommendation: {report['recommendation']}
    Causal Explanation: {report['causal_explanation']}
    Second Opinion: {report['second_opinion']}
    Contradictions: {report['contradiction_warning']}
    Clinical Pearl: {report['clinical_pearl']}
            """
            pdf.multi_cell(0, 10, txt=s(content))
            pdf.output("report.pdf")
            with open("report.pdf", "rb") as f:
                st.download_button("Download PDF", f, file_name=f"patient_{pid}_report.pdf")
    st.subheader("Live Clinician Notes & AI Chat")
    mic_col, chat_col = st.columns(2)
    with mic_col:
        st.markdown("Add a spoken note")
        spoken_note = st.text_input("Or type a new note:",placeholder="e.g., Patient now confused as BP dropping")
        if st.button("Add Note and Re-analyze"):
            if spoken_note:
                new_row = pd.DataFrame([{'patient_id': pid,'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'wbc': last['wbc'],'lactate': last['lactate'],
                    'creatinine': last['creatinine'],'notes': spoken_note}])
                df_extended = pd.concat([df, new_row], ignore_index=True)
                st.session_state.custom_note_df = df_extended
                st.rerun()
    with chat_col:
        st.markdown("Ask about this patient")
        userq = st.chat_input("Type your question")
        if userq:
            from groq import Groq
            import os
            from dotenv import load_dotenv
            load_dotenv()
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            prompt = f"Patient report: {report}\n\nQuestion: {userq}\nAnswer briefly based on the report."
            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",messages=[{"role": "user", "content": prompt}],temperature=0.2,max_tokens=200)
                answer = resp.choices[0].message.content
                st.info(f"AI: {answer}")
            except:
                st.error("API error using fallback: Check lactate and WBC trends.")
    st.markdown("---")
    st.subheader("Share Snapshot")
    snapshot_data = json.dumps({
        "patient": pid,"risk": report['risk_level'],"trust": report['trust_score'],"recommendation": report['recommendation']})
    qr = qrcode.make(snapshot_data)
    buffer = BytesIO()
    qr.save(buffer, format="PNG")
    st.download_button("Generate QR Code", data=buffer.getvalue(), file_name="snapshot.png", mime="image/png")
with tf:
    st.subheader("A Message for the Family")
    with st.spinner(" A summary"):
        en_sum, hi_sum = family_summary(pid, df)
    st.markdown("### In English")
    st.info(en_sum)
    st.markdown("### हिंदी में")
    st.info(hi_sum)
    st.caption("This AI-generated summary helps families understand their loved one's condition. Always discuss with the clinical team for medical decisions.")
hotkey_js = """
<script>
document.addEventListener('keydown', function(e) {
    if (e.key === '1') {const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Inject Lab Error'));
        if (btn) btn.click();
    }
    if (e.key === '2') {const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Speak Handoff'));
        if (btn) btn.click();
    }
    if (e.key === '3') {const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Download PDF'));
        if (btn) btn.click();
    }
});
</script>
"""
html(hotkey_js, height=0)
st.caption("All systems ready ICU CoPilot v1.0AI-powered diagnostic assistant with lab error refusal.")

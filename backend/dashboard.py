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
from chief_agent import gen_report
st.set_page_config(page_title="ICU Co‑Pilot", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    
    .dark-mode {
        background-color: #0a0f1e;
        color: #e0e0e0;
    }
    .dark-mode .stApp {
        background-color: #0a0f1e;
    }/* Vitals card */
    .vitals-card {
        background: linear-gradient(135deg, #1a1f2e, #0f1420);
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        border: 1px solid #2a2f3e;
    }
    .vital-value {
        font-size: 2.5rem;
        font-weight: bold;
        font-family: monospace;
    }
    .vital-label {
        font-size: 0.9rem;
        color: #aaa;
    }
    .alert-pulse {
        animation: pulse 1s infinite;
        background-color: #ff4b4b;
        border-radius: 50%;
        width: 12px;
        height: 12px;
        display: inline-block;
        margin-left: 8px;
    }
    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 0.7; }
        100% { transform: scale(1.2); opacity: 1; }
    }/* Trust gauge */
    .trust-gauge {
        background: #2e3a4e;
        border-radius: 10px;
        height: 10px;
        width: 100%;
    }
    .trust-fill {
        background: linear-gradient(90deg, #ff4b4b, #ffa500, #00cc66);
        border-radius: 10px;
        height: 100%;
        width: 0%;
    }/* Risk badge */
    .risk-high { background-color: #ff4b4b; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .risk-moderate { background-color: #ffa500; color: black; padding: 4px 12px; border-radius: 20px; }
    .risk-low { background-color: #00cc66; color: black; padding: 4px 12px; border-radius: 20px; }
    .risk-unknown { background-color: #888; color: white; padding: 4px 12px; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)
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
        df1copy = df.copy()
        df1copy.iloc[-1, df1copy.columns.get_loc('wbc')] = 100.0
        df1copy.iloc[-1, df1copy.columns.get_loc('notes')] = "LAB ERROR: Wrong"
        return df1copy
    if st.session_state.simulate_lactate is not None and pid == st.session_state.get('sim_pid', ''):
        df1copy = df.copy()
        df1copy.iloc[-1, df1copy.columns.get_loc('lactate')] = st.session_state.simulate_lactate
        return df1copy
    if st.session_state.simulate_wbc is not None and pid == st.session_state.get('sim_pid', ''):
        df1copy = df.copy()
        df1copy.iloc[-1, df1copy.columns.get_loc('wbc')] = st.session_state.simulate_wbc
        return df1copy
    return df
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    dark_toggle = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()
if st.session_state.dark_mode:
    st.markdown('<style>body { background-color: #0a0f1e; color: #e0e0e0; } .stApp { background-color: #0a0f1e; }</style>', unsafe_allow_html=True)
pid = st.selectbox("Select Patient", ["P001","P002","P003","P004","P005"], key="pid")
dforg = loaddata(pid)
df = apply(dforg, pid)
df = flags(df)  
st.markdown("ICU Monitor")
last = df.iloc[-1]
wval = last['wbc']
lval = last['lactate']
cval = last['creatinine']
def color_code(val, low, high):
    if val > high: return "red"
    if val < low: return "orange"
    return "green"
col_v1, col_v2, col_v3, col_v4 = st.columns(4)
with col_v1:
    st.markdown(f"""
    <div class='vitals-card'><div class='vital-label'>WBC (x10³/µL)</div><div class='vital-value' style='color:{color_code(wval, 4, 11)}'>{wval}</div>
    <div class='vital-label'>normal 4-11</div></div>""", unsafe_allow_html=True)
with col_v2:
    st.markdown(f"""<div class='vitals-card'><div class='vital-label'>Lactate (mmol/L)</div><div class='vital-value' style='color:{color_code(lval, 0.5, 2.0)}'>{lval}</div>
                <div class='vital-label'>normal <2.0</div></div>""", unsafe_allow_html=True)
with col_v3:
    st.markdown(f"""<div class='vitals-card'><div class='vital-label'>Creatinine (mg/dL)</div>
    <div class='vital-value' style='color:{color_code(cval, 0.6, 1.2)}'>{cval}</div><div class='vital-label'>normal 0.6-1.2</div>
    </div>""", unsafe_allow_html=True)
with col_v4:
    oe = df[['wbc_outlier','lactate_outlier','creatinine_outlier']].any().any()
    risk_preview = "High" if (lval>4.0 or wval>20) and not oe else "Medium" if (lval>2.0 or wval>12) else "Low"
    if oe:
        risk_preview = "Unknown"
    alert = f"""
    <div class='vitals-card'>
        <div class='vital-label'>RISK STATUS</div>
        <div class='vital-value' style='font-size:1.8rem'>{risk_preview}</div>
        {"<div class='alert-pulse'></div>" if risk_preview=="High" else ""}
    </div>
    """
    st.markdown(alert, unsafe_allow_html=True)
def get(df):
    return process_notes(df)
sym_df = get(df)
lastest = sym_df[sym_df['timestamp'] == last['timestamp']]['symptom'].tolist()
query = f"WBC {wval}, lactate {lval}, symptoms: {', '.join(lastest) if lastest else 'none'}"
g = get_guide(query, top=2)
report = gen_report(pid, df, sym_df, g)
left_col, right_col = st.columns([2, 1])
with left_col:
    st.subheader("Lab Trends & Outliers")
    f = go.Figure()
    f.add_trace(go.Scatter(x=df['timestamp'], y=df['wbc'], mode='lines+markers', name='WBC', line=dict(color='#00cc96')))
    f.add_trace(go.Scatter(x=df['timestamp'], y=df['lactate'], mode='lines+markers', name='Lactate', line=dict(color='#ffa500')))
    opts = df[df['wbc_outlier'] == True]
    if not opts.empty:
        f.add_trace(go.Scatter(x=opts['timestamp'], y=opts['wbc'], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='WBC Outlier', hovertemplate='Probable lab error<br>WBC=%{y}<extra></extra>'))
    ol = df[df['lactate_outlier'] == True]
    if not ol.empty:
        f.add_trace(go.Scatter(x=ol['timestamp'], y=ol['lactate'], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Lactate Outlier'))
    f.update_layout(height=400, template='plotly_dark' if st.session_state.dark_mode else 'plotly_white', margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(f, use_container_width=True)
    st.subheader("Risk Radar")
    wn = min(wval/30, 1)
    lacn = min(lval/8, 1)
    sym_score = min(len(lastest)/5, 1)
    gs= g[0]['relevance'] if g else 0
    rf = go.Figure(data=go.Scatterpolar(r=[wn, lacn, sym_score, gs],theta=['WBC', 'Lactate', 'Symptoms', 'Guideline'],
        fill='toself',
        marker=dict(color='#ff4b4b')))
    rf.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=300, showlegend=False)
    st.plotly_chart(rf, use_container_width=True)
import pandas as pd
import numpy as np
def gen_report(pid, df, sym_df, guides):
    """
    df: patient dataframe with _outlier columns
    sym_df: symptoms dataframe (timestamp, symptom, confidence, is_new)
    guides: list of dicts from rag_agent.get_guide()
    Returns dict with all report fields.
    """
    outlier_cols = [c for c in df.columns if c.endswith('_outlier')]
    has_outlier = df[outlier_cols].any().any()
    last = df.iloc[-1]
    wbc = last['wbc']
    lactate = last['lactate']
    creat = last['creatinine']
    last_ts = last['timestamp']
    latest_syms = sym_df[sym_df['timestamp'] == last_ts]['symptom'].tolist()
    trust = 100
    if has_outlier:
        trust -= 50
    
    if pd.isna(wbc) or pd.isna(lactate):
        trust -= 20
    
    if guides:
        max_rel = max(g['relevance'] for g in guides)
        if max_rel < 0.5:
            trust -= 20
    trust = max(0, min(100, trust))
    if has_outlier:
        risk = "Unknown – lab error suspected"
        refusal = True
        rec = "⚠️ Do not act on outlier values. Request redraw of labs."
    else:
        refusal = False
        
        if lactate > 4.0 and ('fever' in latest_syms or wbc > 15):
            risk = "High"
            rec = "Administer broad-spectrum antibiotics within 1 hour. Repeat lactate in 2h."
        elif lactate > 2.0 or wbc > 12:
            risk = "Moderate"
            rec = "Monitor closely. Consider blood cultures. Repeat labs in 4h."
        else:
            risk = "Low"
            rec = "Routine monitoring. No immediate action needed."
    
    if has_outlier:
        causal = f"Outlier detected in {', '.join([c.replace('_outlier','') for c in outlier_cols if last[c]])}. Risk assessment paused."
    else:
        causal = f"WBC {wbc}, lactate {lactate}. " + ("Fever present. " if 'fever' in latest_syms else "") + f"Guideline suggests {risk.lower()} risk."
    
    if has_outlier:
        
        fake_risk = "High" if (lactate > 2.0 or wbc > 12) else "Moderate"
        second = f"If outlier were real, risk would be {fake_risk}."
    else:
        second = "No outlier detected – second opinion not applicable."
    
    if len(guides) >= 2 and guides[0]['source'] != guides[1]['source']:
        contra = f"Conflict: {guides[0]['source']} vs {guides[1]['source']}. Prefer newer."
    else:
        contra = "No contradiction."
    time_to_act = None
    if risk == "High" and len(df) > 1:
        prev_lact = df.iloc[-2]['lactate']
        if lactate > prev_lact:
            
            time_to_act = max(30, int(360 / (lactate / prev_lact)))  
        else:
            time_to_act = 60
    if time_to_act:
        act_timer = f"Recommended action within {time_to_act} minutes."
    else:
        act_timer = "No urgent timer."
    pearl = guides[0]['text'].split('.')[0] + "." if guides else "No guideline retrieved."
    return {
        "patient_id": pid,
        "risk_level": risk,
        "trust_score": trust,
        "lab_error_refusal": refusal,
        "causal_explanation": causal,
        "recommendation": rec,
        "second_opinion": second,
        "contradiction_warning": contra,
        "clinical_pearl": pearl,
        "time_to_act": act_timer
    }
if __name__ == "__main__":
    from lab_mapper import loaddata, flags
    from note_parser import process_notes
    from rag_agent import get_guide
    pid = "P001"
    df = loaddata(pid)
    df = flags(df)
    sym_df = process_notes(df)
    last = df.iloc[-1]
    sym_list = sym_df[sym_df['timestamp'] == last['timestamp']]['symptom'].tolist()
    query = f"WBC {last['wbc']}, lactate {last['lactate']}, symptoms: {', '.join(sym_list)}"
    guides = get_guide(query, top=2)
    report = gen_report(pid, df, sym_df, guides)
    for k, v in report.items():
        print(f"{k}: {v}")
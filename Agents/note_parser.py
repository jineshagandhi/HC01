import os
import pandas as pd
import json
import time
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
def get_syms(txt):
    prompt = f"""
    From this clinical note, extract symptoms as JSON.
    Output format: {{"symptoms": [{{"name": "symptom", "conf": 0-100}}]}}
    If no symptoms: {{"symptoms": []}}
    Note: {txt}
    """
    try:
        r = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200
        )
        out = r.choices[0].message.content
        if "```json" in out:
            out = out.split("```json")[1].split("```")[0]
        elif "```" in out:
            out = out.split("```")[1].split("```")[0]
        data = json.loads(out)
        return data.get("symptoms", [])
    except Exception as e:
        print(f"API error: {e} - using fallback")
        syms = []
        txt_low = txt.lower()
        kw = {'fever': 90, 'tachycardia': 80, 'hypotension': 85, 'septic shock': 95, 'infection': 70, 'alert': 50}
        for k, conf in kw.items():
            if k in txt_low:
                syms.append({'name': k, 'conf': conf})
        return syms
def process_notes(pat_df):
    rows = []
    seen = set()
    for idx, row in pat_df.iterrows():
        ts = row['timestamp']
        note = row['notes']
        syms = get_syms(note)
        for s in syms:
            name = s['name'].lower()
            conf = s.get('conf', 50)
            is_new = name not in seen
            rows.append({
                'timestamp': ts,
                'symptom': name,
                'confidence': conf,
                'is_new': is_new
            })
            if is_new:
                seen.add(name)
        time.sleep(0.3)
    return pd.DataFrame(rows)
if __name__ == "__main__":
    from lab_mapper import loaddata
    dat = loaddata('P001')
    sym_df = process_notes(dat)
    print(sym_df.head(10))
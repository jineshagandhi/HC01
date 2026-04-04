import pandas as pd
import numpy as np
from datetime import datetime, timedelta
patients = ['P001', 'P002', 'P003', 'P004', 'P005']
start = datetime(2025, 4, 1, 8, 0)
data = []
for p in patients:
    n = np.random.randint(8, 12)
    wbc = np.random.uniform(6, 9)
    lactate = np.random.uniform(0.8, 1.2)
    creat = np.random.uniform(0.7, 0.9)
    error = np.random.randint(1, n - 1)
    for i in range(n):
        t = start + timedelta(hours=4 * i)
        w = wbc + 0.2*i + np.random.normal(0, 0.5)
        l = lactate + 0.02*i + np.random.normal(0, 0.1)
        c = creat + 0.01*i + np.random.normal(0, 0.05)
        # Adding fake lab errors 
        if i == error:
            w = np.random.uniform(45, 55)
            note = "Lab error"
        elif w > 20 and l > 4:
            note = "Septic shock suspected"
        elif w > 15:
            note = "High WBC, monitor"
        elif w > 10:
            note = "Possible infection"
        else:
            note = "Stable"
        data.append({
            'patient_id': p,
            'timestamp': t,
            'wbc': round(w, 1),
            'lactate': round(l, 1),
            'creatinine': round(c, 2),
            'notes': note})
df = pd.DataFrame(data)
df.to_csv("mock.csv")
print("Mock data saved.")
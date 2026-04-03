import pandas as pd
import numpy as np
def loaddata(PID):
    data = pd.read_csv('mock.csv')
    patientdata = data[data['patient_id']==PID].copy()
    patientdata['timestamp'] = pd.to_datetime(patientdata['timestamp'])
    patientdata = patientdata.sort_values('timestamp')
    return patientdata
def detect(values, window=3, threshold=2.0):
    o = [False]*len(values)
    for i in range(window,len(values)):
        r = values[i-window:i]
        mean = np.mean(r)
        std = np.std(r)
        if std>0 and abs(values[i]-mean)>threshold*std:
            o[i]=True
    return o
def flags(data, lab_cols=['wbc','lactate','creatinine']):
    for col in lab_cols:
        data[f'{col}_outlier'] = detect(data[col].values)
    return data
if __name__ == "__main__":
    data = loaddata('P001')
    data = flags(data)
    print(data[['timestamp','wbc','wbc_outlier','lactate','lactate_outlier']])
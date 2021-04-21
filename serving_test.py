import numpy as np
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt
import json
import requests

n_steps = 5
n_features = 1

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

april_pm25 = pd.read_excel('./data/data_past_time_2020_04.xls', sheet_name='Sheet1', header=[0,1])
df_april_pm25 = april_pm25[['날짜', 'PM2.5']]
print(df_april_pm25.isnull().sum())
df_april_pm25 = df_april_pm25.fillna(method='ffill')
april_pm25_seq = array(df_april_pm25["PM2.5"])
X_april_pm25, y_april_pm25 = split_sequence(april_pm25_seq, n_steps)
X_april_pm25 = X_april_pm25.reshape((X_april_pm25.shape[0], X_april_pm25.shape[1], n_features))
print(X_april_pm25)

# data = json.dumps({"signature_name": "serving_default", "instances": X_april_pm25.tolist()})
data = json.dumps({"signature_name": "serving_default", "inputs": {'lstm_input': X_april_pm25.tolist()}})
headers = {"content-type": "application/json"}
json_response = requests.post(f'http://localhost:8501/v1/models/pm25/versions/3:predict',
                              data=data,
                              headers=headers)

print(json.loads(json_response.text).keys())
# print(json.loads(json_response.text)['error'])
predictions = json.loads(json_response.text)['outputs']
# predictions = json.loads(json_response.text)['predictions']
print("Predictions:", predictions)
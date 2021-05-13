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
# version 3 is for custom ConvLSTM (integer output to indicate very bad, bad, good, etc)
json_response = requests.post(f'http://localhost:8501/v1/models/pm25/versions/1:predict',
                              data=data,
                              headers=headers)

print(json.loads(json_response.text).keys())
# print(json.loads(json_response.text)['error'])
predictions = json.loads(json_response.text)['outputs']
# predictions = json.loads(json_response.text)['predictions']
print("Predictions:", predictions)

y_april_forecast = predictions

# date for x axis
mdh = np.arange(0, len(april_pm25_seq) + 1, 100)
dates = april_pm25[['날짜']]
dates = array(dates)
month_day_hour = dates[mdh]

plt.figure(figsize=(20, 10))
plt.plot(y_april_forecast, '#FF4500')
plt.plot(april_pm25_seq, '#4169E1')
plt.legend(['Predicted', 'Actual'], loc='upper right', fontsize=40) # fontsize=40
#path = 'C:/Windows/Fonts/NanumGothicBold.ttf'
#fontprop = fm.FontProperties(fname=path, size=40)

plt.title('April PM2.5 Forecast', fontsize=40) # fontproperties=fontprop, fontsize=40
plt.xticks(mdh, month_day_hour, fontsize=20, rotation=20) # fontsize=30, rotation=45
plt.yticks(fontsize=30) # fontsize=30
plt.ylabel('PM2.5', fontsize=35) # fontsize=35
plt.show()

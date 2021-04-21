import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# univariate convlstm example
from numpy import array
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import ConvLSTM2D


df_01 = pd.read_excel('./data/data_past_time_2020_01.xls', sheet_name='Sheet1', header=[0,1])
df_02 = pd.read_excel('./data/data_past_time_2020_02.xls', sheet_name='Sheet1', header=[0,1])
df_03 = pd.read_excel('./data/data_past_time_2020_03.xls', sheet_name='Sheet1', header=[0,1])
df_01_03 = pd.concat([df_01, df_02, df_03], ignore_index=True)
df_pm25 = df_01_03[['날짜', 'PM2.5']]
df_pm25 = df_pm25.fillna(method='ffill')
pm25_seq = array(df_pm25['PM2.5'])

# split a univariate sequence into samples
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

# choose a number of time steps
n_steps = 6
# split into samples
X, y = split_sequence(pm25_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = n_steps // n_seq
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))

# This might need for error
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
hist = model.fit(X, y, epochs=1, verbose=1)
# demonstrate prediction
# x_input = array([60, 70, 80, 90])
# x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)

plt.plot(hist.history['loss'])
plt.ylim(0.0, 100.0)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['pm25'], loc='upper right')
plt.show()

model.save(f'models/pm25/2')

# Forecast PM2.5
april_pm25 = pd.read_excel('./data/data_past_time_2020_04.xls', sheet_name='Sheet1', header=[0,1])
df_april_pm25 = april_pm25[['날짜', 'PM2.5']]
df_april_pm25 = df_april_pm25.fillna(method='ffill')
april_pm25_seq = array(df_april_pm25["PM2.5"])

# choose a number of time steps
n_steps = 6
# split into samples
X_april_pm25, y_april_pm25 = split_sequence(april_pm25_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = n_steps // n_seq
X_april_pm25 = X_april_pm25.reshape((X_april_pm25.shape[0], n_seq, 1, n_steps, n_features))
y_april_forecast = model.predict(X_april_pm25)

# date for x axis
mdh = np.arange(0, len(april_pm25_seq) + 1, 100)
dates = april_pm25[['날짜']]
dates = array(dates)
month_day_hour = dates[mdh]

plt.figure(figsize=(20, 10))
plt.plot(y_april_forecast, '#FF4500')
plt.plot(april_pm25_seq, '#4169E1')
plt.legend(['Predicted', 'Actual'], loc='upper right', fontsize=40)
#path = 'C:/Windows/Fonts/NanumGothicBold.ttf'
#fontprop = fm.FontProperties(fname=path, size=40)

plt.title('April PM2.5 Forecast', fontsize=40) # fontproperties=fontprop
plt.xticks(mdh, month_day_hour, fontsize=30, rotation=45)
plt.yticks(fontsize=30)
plt.ylabel('PM2.5', fontsize=35)
plt.show()

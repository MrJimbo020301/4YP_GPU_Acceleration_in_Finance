# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import time
import numpy as np
import pandas as pd
import yfinance as yf

from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Dropout

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# Set the style for plots using Seaborn directly
sns.set_style('whitegrid')
sns.set_context('poster')

# Define the start and end dates
start_date = datetime.datetime(2000, 1, 1)
end_date = datetime.datetime.now()

# Download S&P 500 data using yfinance
sp_df = yf.download('^GSPC', start=start_date, end=end_date)
sp_close_series = sp_df['Close']

# Verify the downloaded data
print("First few entries of sp_close_series:")
print(sp_close_series.head())
print("\nData type of sp_close_series:")
print(sp_close_series.dtypes)

# Clean the data: ensure it's numeric and drop NaNs
sp_close_series = pd.to_numeric(sp_close_series, errors='coerce').dropna()

# Verify the cleaning
print("\nAfter cleaning:")
print(sp_close_series.head())
print("\nData type after cleaning:")
print(sp_close_series.dtype)

# Plot the closing prices
plt.figure(figsize=(15, 7))
plt.plot(sp_close_series, color='teal')
plt.title('S&P 500 Closing Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

print("\nData range:")
print(sp_df.index.min(), sp_df.index.max())

WINDOW = 6
PRED_LENGTH = int(WINDOW / 2)

def get_reg_train_test(timeseries, sequence_length=51,
                       train_size=0.9, roll_mean_window=5,
                       normalize=True, scale=False):
    # Ensure the timeseries is numeric
    timeseries = pd.to_numeric(timeseries, errors='coerce').dropna()

    # Smoothen out series
    if roll_mean_window:
        timeseries = timeseries.rolling(roll_mean_window).mean().dropna()
    
    # Create windows
    result = []
    for index in range(len(timeseries) - sequence_length):
        window = timeseries[index: index + sequence_length].values
        result.append(window)
    
    # Normalize data as a variation of 0th index
    if normalize:
        normalised_data = []
        for window in result:
            try:
                normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
                normalised_data.append(normalised_window)
            except ValueError as e:
                print(f"ValueError encountered: {e}")
                continue  # Skip windows with invalid data
        result = normalised_data
    
    # Convert to numpy array
    result = np.array(result)
    
    # Identify train-test splits
    row = round(train_size * result.shape[0])
    
    # Split train and test sets
    train = result[:int(row), :]
    test = result[int(row):, :]
    
    # Scale data in 0-1 range
    scaler = None
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    
    # Split independent and dependent variables  
    x_train = train[:, :-1]
    y_train = train[:, -1]
        
    x_test = test[:, :-1]
    y_test = test[:, -1]
    
    # Reshape for LSTM input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 
    
    return x_train, y_train, x_test, y_test, scaler

# Prepare the training and testing data
x_train, y_train, x_test, y_test, scaler = get_reg_train_test(
    sp_close_series,
    sequence_length=WINDOW + 1,
    roll_mean_window=None,
    normalize=True,
    scale=False
)

print("Data Split Complete")

print("x_train shape={}".format(x_train.shape))
print("y_train shape={}".format(y_train.shape))
print("x_test shape={}".format(x_test.shape))
print("y_test shape={}".format(y_test.shape))



"""
# Continue with model building, training, etc.
# Example:
# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)

# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) if scaler else predictions

# Inverse transform y_test if scaler was used
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)) if scaler else y_test

# Plot the results
plt.figure(figsize=(15, 7))
plt.plot(sp_close_series.index[-len(y_test):], y_test, color='blue', label='Actual S&P 500 Price')
plt.plot(sp_close_series.index[-len(predictions):], predictions, color='red', label='Predicted S&P 500 Price')
plt.title('S&P 500 Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Data Preparation
# Load the dataset
df = pd.read_csv('Gold Price_Year 2018.csv')
df.drop(['Vol.', 'Change %'], axis=1, inplace=True)

# Transforming Data
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

NumCols = df.columns.drop(['Date'])
df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
df[NumCols] = df[NumCols].astype('float64')

# Checking Duplicates and Nulls
df.duplicated().sum()
df.isnull().sum().sum()

# Splitting Data into Training & Test Sets
test_size = df[df.Date.dt.month == 12].shape[0]

# Data Scaling
scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1, 1))

# Restructure Data & Create Sliding Window
window_width = 180

# Train Data Preparation
train_data = df.Price[:-test_size]
train_data_scaled = scaler.transform(train_data.values.reshape(-1, 1))

X_train = []
y_train = []

for i in range(window_width, len(train_data_scaled)):
    X_train.append(train_data_scaled[i - window_width:i, 0])
    y_train.append(train_data_scaled[i, 0])

# Test Data Preparation
test_data = df.Price[-(test_size + window_width):]
test_data_scaled = scaler.transform(test_data.values.reshape(-1, 1))

X_test = []
y_test = []

for i in range(window_width, len(test_data_scaled)):
    X_test.append(test_data_scaled[i - window_width:i, 0])
    y_test.append(test_data_scaled[i, 0])

# Converting Data to NumPy Arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape data for PyTorch LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

print('X_train Shape: ', X_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_test Shape:  ', X_test.shape)
print('y_test Shape:  ', y_test.shape)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

# Use X_test_tensor and y_test_tensor as validation data
X_val_tensor = X_test_tensor
y_val_tensor = y_test_tensor

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        out = out[:, -1, :]  # Take the output from the last time step
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.NAdam(model.parameters())

# Training the Model
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_tensor)
        val_loss = criterion(val_output, y_val_tensor)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

# Model Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

MAPE = mean_absolute_percentage_error(y_test_tensor, y_pred)
Accuracy = 100 - MAPE

print(f'Test MAPE: {MAPE:.2f}%')
print(f'Test Accuracy: {Accuracy:.2f}%')

# Convert tensors to NumPy arrays
y_pred_np = y_pred.detach().cpu().numpy()
y_test_np = y_test_tensor.detach().cpu().numpy()

# Inverse transform y_test and y_pred to their original scale
y_test_true = scaler.inverse_transform(y_test_np)
y_test_pred = scaler.inverse_transform(y_pred_np)

# Inverse transform train_data_scaled to get the original training prices (excluding initial window)
train_data_inv = scaler.inverse_transform(train_data_scaled[window_width:])

# Flatten the arrays for plotting
train_data_inv = train_data_inv.flatten()
y_test_true = y_test_true.flatten()
y_test_pred = y_test_pred.flatten()

# Get the dates corresponding to the training data
train_dates = df['Date'].iloc[window_width:len(train_data_inv) + window_width].reset_index(drop=True)

# Get the dates corresponding to the test data
test_dates = df['Date'].iloc[-(len(y_test_true)):].reset_index(drop=True)

# Verify lengths
print('Length of train_dates:', len(train_dates))
print('Length of train_data_inv:', len(train_data_inv))
print('Length of test_dates:', len(test_dates))
print('Length of y_test_true:', len(y_test_true))
print('Length of y_test_pred:', len(y_test_pred))

# Plotting
plt.figure(figsize=(15, 6), dpi=150)
plt.rcParams['axes.facecolor'] = 'yellow'
plt.rc('axes', edgecolor='white')

# Plot the training data
plt.plot(train_dates, train_data_inv, color='black', lw=2, label='Training Data')

# Plot the actual test data
plt.plot(test_dates, y_test_true, color='blue', lw=2, label='Actual Test Data')

# Plot the predicted test data
plt.plot(test_dates, y_test_pred, color='red', lw=2, label='Predicted Test Data')

# Customize the plot
plt.title('Model Performance on Gold Price Prediction', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(loc='upper left', prop={'size': 15})
plt.grid(color='white')
plt.show()

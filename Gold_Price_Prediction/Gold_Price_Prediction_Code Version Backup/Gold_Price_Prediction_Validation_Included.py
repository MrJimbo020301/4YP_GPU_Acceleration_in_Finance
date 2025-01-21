import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Load and prepare data
df = pd.read_csv('Gold Price_Year 2018.csv')
df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# Transform columns to numerical format
num_cols = df.columns.drop(['Date'])
df[num_cols] = df[num_cols].replace({',': ''}, regex=True)
df[num_cols] = df[num_cols].astype('float64')

# Check for duplicates and nulls
assert df.duplicated().sum() == 0, "Duplicate rows found!"
assert df.isnull().sum().sum() == 0, "Missing values found!"

# Data scaling
scaler = MinMaxScaler()
data = df['Price'].values.reshape(-1, 1)
data_scaled = scaler.fit_transform(data)

# Window width
window_width = 135
# Prepare sequences and corresponding dates
X, y, dates = [], [], []
for i in range(window_width, len(data_scaled)):
    X.append(data_scaled[i - window_width:i, 0])
    y.append(data_scaled[i, 0])
    dates.append(df['Date'].iloc[i])

X = np.array(X)
y = np.array(y)
dates = np.array(dates)

# Ensure that dates and y have the same length
assert len(dates) == len(y), "Mismatch between dates and y lengths"

# Split the data into training, validation, and test sets based on dates
train_mask = dates < pd.to_datetime('2018-10-24')
val_mask = (dates >= pd.to_datetime('2018-10-24')) & (dates < pd.to_datetime('2018-11-28'))
test_mask = dates >= pd.to_datetime('2018-11-28')

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

train_dates = dates[train_mask]
val_dates = dates[val_mask]
test_dates = dates[test_mask]

# Reshape for PyTorch LSTM input
X_train = X_train.reshape(-1, window_width, 1)
X_val = X_val.reshape(-1, window_width, 1)
X_test = X_test.reshape(-1, window_width, 1)
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.02)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.02)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.NAdam(model.parameters())

# Training loop with validation
num_epochs = 160
for epoch in range(num_epochs):
    # Training phase
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Validation phase
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, y_val_tensor)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')

# Evaluate on test data
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

# Calculate MAPE and accuracy
mape = mean_absolute_percentage_error(y_test_tensor.numpy(), y_pred.numpy())
accuracy = 100 - mape * 100
print(f'Test MAPE: {mape * 100:.2f}%')
print(f'Test Accuracy: {accuracy:.2f}%')

# Convert predictions back to original scale
y_train_pred = model(X_train_tensor).detach().numpy()
y_val_pred = model(X_val_tensor).detach().numpy()
y_test_pred = y_pred.detach().numpy()

y_train_inv = scaler.inverse_transform(y_train_tensor.numpy())
y_train_pred_inv = scaler.inverse_transform(y_train_pred)
y_val_inv = scaler.inverse_transform(y_val_tensor.numpy())
y_val_pred_inv = scaler.inverse_transform(y_val_pred)
y_test_inv = scaler.inverse_transform(y_test_tensor.numpy())
y_test_pred_inv = scaler.inverse_transform(y_test_pred)

# Ensure lengths match
assert len(train_dates) == len(y_train_inv.flatten()), "Mismatch in training data lengths"
assert len(val_dates) == len(y_val_inv.flatten()), "Mismatch in validation data lengths"
assert len(test_dates) == len(y_test_inv.flatten()), "Mismatch in test data lengths"

# Combine all dates and corresponding actual and predicted values for smooth plotting
all_dates = np.concatenate((train_dates, val_dates, test_dates))
all_actual = np.concatenate((y_train_inv.flatten(), y_val_inv.flatten(), y_test_inv.flatten()))
all_predicted = np.concatenate((y_train_pred_inv.flatten(), y_val_pred_inv.flatten(), y_test_pred_inv.flatten()))

# Plotting
plt.figure(figsize=(14, 7))

# Actual Price
plt.plot(all_dates, all_actual, color='black', label='Actual Price')

# Training data
plt.plot(train_dates, y_train_pred_inv.flatten(), color='#b8860b', linestyle='--', label='Predicted Training Data')

# Validation data
plt.plot(val_dates, y_val_pred_inv.flatten(), color='purple', linestyle='--', label='Predicted Validation Data')

# Test data
plt.plot(test_dates, y_test_pred_inv.flatten(), color='red', linestyle='--', label='Predicted Test Data')

# Highlight training, validation, and test periods
plt.axvspan(train_dates.min(), train_dates.max(), color='gray', alpha=0.1, label='Training Period')
plt.axvspan(val_dates.min(), val_dates.max(), color='green', alpha=0.1, label='Validation Period')
plt.axvspan(test_dates.min(), test_dates.max(), color='blue', alpha=0.1, label='Test Period')

plt.title('Model Performance on Gold Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
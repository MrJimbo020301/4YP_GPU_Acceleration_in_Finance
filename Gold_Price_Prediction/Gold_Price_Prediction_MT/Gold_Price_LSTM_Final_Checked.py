
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

# Features to include
features = ['Price', 'Open', 'High', 'Low']
data = df[features].values

# Data scaling (scale each feature individually)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Window width
window_width = 30  # Adjusted window width for experimentation

# Prepare sequences and corresponding dates
X, y, dates = [], [], []
for i in range(window_width, len(data_scaled)):
    X.append(data_scaled[i - window_width:i])
    y.append(data_scaled[i, 0])  # Assuming 'Price' is the target
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

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1)
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 32  # Adjusted batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the enhanced LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=len(features),
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.05,
        )
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMModel().to(device)
criterion = nn.HuberLoss()  # Switched to Huber Loss
optimizer = optim.AdamW(model.parameters(), lr=0.0005)  # Adjusted optimizer and learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=175)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=175, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
            
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training loop with early stopping and learning rate scheduler
early_stopping = EarlyStopping(patience=175, min_delta=0)
num_epochs = 800  # Increased number of epochs
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)
    
    # Validation phase
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_output = model(X_batch)
            val_loss = criterion(val_output, y_batch)
            val_losses.append(val_loss.item())
    avg_val_loss = np.mean(val_losses)
    
    # Scheduler step
    scheduler.step(avg_val_loss)
    
    # Check for early stopping
    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        break
    
    # Print losses every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')

# Evaluate on test data
model.eval()
test_losses = []
y_pred_list = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        test_losses.append(loss.item())
        y_pred_list.append(y_pred.cpu().numpy())
avg_test_loss = np.mean(test_losses)
print(f'Test Loss: {avg_test_loss:.6f}')

# Concatenate predictions
y_pred_array = np.concatenate(y_pred_list, axis=0)

# Calculate MAPE and accuracy
mape = mean_absolute_percentage_error(y_test_tensor.numpy(), y_pred_array)
accuracy = 100 - mape * 100
print(f'Test MAPE: {mape * 100:.2f}%')
print(f'Test Accuracy: {accuracy:.2f}%')

# Convert predictions back to original scale
y_train_pred = []
with torch.no_grad():
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        y_train_pred_batch = model(X_batch)
        y_train_pred.append(y_train_pred_batch.cpu().numpy())
y_train_pred = np.concatenate(y_train_pred, axis=0)

y_val_pred = []
with torch.no_grad():
    for X_batch, _ in val_loader:
        X_batch = X_batch.to(device)
        y_val_pred_batch = model(X_batch)
        y_val_pred.append(y_val_pred_batch.cpu().numpy())
y_val_pred = np.concatenate(y_val_pred, axis=0)

y_train_inv = scaler.inverse_transform(np.hstack((y_train_tensor.numpy(), np.zeros((y_train_tensor.shape[0], len(features)-1)))))
y_train_pred_inv = scaler.inverse_transform(np.hstack((y_train_pred, np.zeros((y_train_pred.shape[0], len(features)-1)))))
y_val_inv = scaler.inverse_transform(np.hstack((y_val_tensor.numpy(), np.zeros((y_val_tensor.shape[0], len(features)-1)))))
y_val_pred_inv = scaler.inverse_transform(np.hstack((y_val_pred, np.zeros((y_val_pred.shape[0], len(features)-1)))))
y_test_inv = scaler.inverse_transform(np.hstack((y_test_tensor.numpy(), np.zeros((y_test_tensor.shape[0], len(features)-1)))))
y_test_pred_inv = scaler.inverse_transform(np.hstack((y_pred_array, np.zeros((y_pred_array.shape[0], len(features)-1)))))

# Extract the target variable ('Price') after inverse scaling
y_train_inv = y_train_inv[:, 0]
y_train_pred_inv = y_train_pred_inv[:, 0]
y_val_inv = y_val_inv[:, 0]
y_val_pred_inv = y_val_pred_inv[:, 0]
y_test_inv = y_test_inv[:, 0]
y_test_pred_inv = y_test_pred_inv[:, 0]

# Ensure lengths match
assert len(train_dates) == len(y_train_inv), "Mismatch in training data lengths"
assert len(val_dates) == len(y_val_inv), "Mismatch in validation data lengths"
assert len(test_dates) == len(y_test_inv), "Mismatch in test data lengths"

# Combine all dates and corresponding actual and predicted values for smooth plotting
all_dates = np.concatenate((train_dates, val_dates, test_dates))
all_actual = np.concatenate((y_train_inv, y_val_inv, y_test_inv))
all_predicted = np.concatenate((y_train_pred_inv, y_val_pred_inv, y_test_pred_inv))

# Plotting
plt.figure(figsize=(14, 7))

# Actual Price
plt.plot(all_dates, all_actual, color='black', label='Actual Price')

# Training data
#plt.plot(train_dates, y_train_pred_inv, color='#b8860b', linestyle='--', label='Predicted Training Data')

# Validation data
plt.plot(val_dates, y_val_pred_inv, color='purple', linestyle='--', label='Predicted Validation Data')

# Test data
plt.plot(test_dates, y_test_pred_inv, color='red', linestyle='--', label='Predicted Test Data')

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
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Feature engineering: Moving averages
df['MA_7'] = df['Price'].rolling(window=7).mean()
df['MA_14'] = df['Price'].rolling(window=14).mean()
df['MA_30'] = df['Price'].rolling(window=30).mean()

# Drop rows with NaN values due to moving averages
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Update features to include moving averages
features.extend(['MA_7', 'MA_14', 'MA_30'])

data = df[features].values

# Data scaling (scale each feature individually)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Window width
window_width = 30  # Adjusted window width

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
X_train_tensor = torch.from_numpy(X_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
X_val_tensor = torch.from_numpy(X_val).float().to(device)
y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1).to(device)
X_test_tensor = torch.from_numpy(X_test).float().to(device)
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1).to(device)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 16  # Adjusted batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the bidirectional LSTM model with increased dropout
class BiLSTMModel(nn.Module):
    def __init__(self):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=len(features), hidden_size=64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)  # Increased dropout rate
        self.fc = nn.Linear(64 * 2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = BiLSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
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

# Training loop with early stopping
early_stopping = EarlyStopping(patience=15, min_delta=1e-4)
num_epochs = 300  # Increased number of epochs
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        # Move data to device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
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
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            val_output = model(X_batch)
            val_loss = criterion(val_output, y_batch)
            val_losses.append(val_loss.item())
    avg_val_loss = np.mean(val_losses)
    
    # Learning rate scheduler step
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
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        test_losses.append(loss.item())
        y_pred_list.append(y_pred.cpu().numpy())
avg_test_loss = np.mean(test_losses)
print(f'Test Loss: {avg_test_loss:.6f}')

# Concatenate predictions
y_pred_array = np.concatenate(y_pred_list, axis=0)

# Calculate MAPE and accuracy
mape = mean_absolute_percentage_error(y_test_tensor.cpu().numpy(), y_pred_array)
accuracy = 100 - mape * 100
print(f'Test MAPE: {mape * 100:.2f}%')
print(f'Test Accuracy: {accuracy:.2f}%')

# Convert predictions back to original scale
def inverse_transform(y_scaled):
    # Since multiple features are scaled, we need to create a placeholder
    placeholder = np.zeros((y_scaled.shape[0], len(features)))
    placeholder[:, 0] = y_scaled[:, 0]
    y_inv = scaler.inverse_transform(placeholder)[:, 0]
    return y_inv

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

y_train_inv = inverse_transform(y_train_tensor.cpu().numpy())
y_train_pred_inv = inverse_transform(y_train_pred)
y_val_inv = inverse_transform(y_val_tensor.cpu().numpy())
y_val_pred_inv = inverse_transform(y_val_pred)
y_test_inv = inverse_transform(y_test_tensor.cpu().numpy())
y_test_pred_inv = inverse_transform(y_pred_array)

# Ensure lengths match
assert len(train_dates) == len(y_train_inv), "Mismatch in training data lengths"
assert len(val_dates) == len(y_val_inv), "Mismatch in validation data lengths"
assert len(test_dates) == len(y_test_inv), "Mismatch in test data lengths"

# Combine all dates and corresponding actual and predicted values for smooth plotting
all_dates = np.concatenate((train_dates, val_dates, test_dates))
all_actual = np.concatenate((y_train_inv, y_val_inv, y_test_inv))
all_predicted = np.concatenate((y_train_pred_inv, y_val_pred_inv, y_test_pred_inv))

# Sort the combined arrays by dates
sorted_indices = np.argsort(all_dates)
all_dates = all_dates[sorted_indices]
all_actual = all_actual[sorted_indices]
all_predicted = all_predicted[sorted_indices]

# Plotting
plt.figure(figsize=(14, 7))

# Actual Price
plt.plot(all_dates, all_actual, color='black', label='Actual Price')

# Predicted Price
plt.plot(all_dates, all_predicted, color='blue', linestyle='--', label='Predicted Price')

# Highlight training, validation, and test periods
plt.axvspan(train_dates.min(), train_dates.max(), color='gray', alpha=0.1, label='Training Period')
plt.axvspan(val_dates.min(), val_dates.max(), color='green', alpha=0.1, label='Validation Period')
plt.axvspan(test_dates.min(), test_dates.max(), color='red', alpha=0.1, label='Test Period')

plt.title('Model Performance on Gold Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

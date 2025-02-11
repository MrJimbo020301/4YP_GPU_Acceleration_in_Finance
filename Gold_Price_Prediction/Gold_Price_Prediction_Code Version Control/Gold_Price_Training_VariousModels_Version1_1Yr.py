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
import os
import random
import warnings
import matplotlib.dates as mdates

# --------------------------------------------------------------------------------
# 1. Reproducibility setup
# --------------------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------------------------------------------
# 2. Load and preprocess data
# --------------------------------------------------------------------------------
df = pd.read_csv('Gold Futures Historical Data_1Yr.csv')
df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

num_cols = df.columns.drop(['Date'])
df[num_cols] = df[num_cols].replace({',': ''}, regex=True)
df[num_cols] = df[num_cols].astype('float64')

# (Optional) Quick checks
assert df.duplicated().sum() == 0, "Duplicate rows found!"
assert df.isnull().sum().sum() == 0, "Missing values found!"

# Features you want to use
features = ['Price', 'Open', 'High', 'Low']
data = df[features].values

# Scale features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Window width for sequence
window_width = 30

# --------------------------------------------------------------------------------
# 3. Create sequences and matching labels (y = 'Price')
# --------------------------------------------------------------------------------
X, y, all_dates = [], [], []
for i in range(window_width, len(data_scaled)):
    X.append(data_scaled[i - window_width:i])
    # y is the next day’s Price after the sequence
    y.append(data_scaled[i, 0])  # 0 => 'Price'
    all_dates.append(df['Date'].iloc[i])

X = np.array(X)
y = np.array(y)
all_dates = np.array(all_dates)

assert len(X) == len(y) == len(all_dates), "Mismatch among X, y, and dates lengths!"

# --------------------------------------------------------------------------------
# 4. Split into training and validation, plus future (unlabeled) for predictions
# --------------------------------------------------------------------------------
# Suppose you have data up to 2025-01-31, and you want to use 2024-08-01 onward
# as validation. Then anything after 2025-01-31 might be future data (unlabeled).
# Adjust as needed.

train_cutoff = pd.to_datetime('2024-08-01') 
val_cutoff   = pd.to_datetime('2025-01-31')  # the last date for which we have known labels

# Create masks
train_mask = all_dates < train_cutoff
val_mask   = (all_dates >= train_cutoff) & (all_dates <= val_cutoff)
future_mask = all_dates > val_cutoff  # these are truly "future" with no known label

# Subset
X_train, y_train, train_dates = X[train_mask], y[train_mask], all_dates[train_mask]
X_val,   y_val,   val_dates   = X[val_mask], y[val_mask], all_dates[val_mask]
X_future, future_dates        = X[future_mask], all_dates[future_mask]
# Note: there's no y_future because you do NOT have actual future labels.

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Future (unlabeled) samples: {len(X_future)}")

# --------------------------------------------------------------------------------
# 5. Convert training & validation sets to PyTorch Datasets
# --------------------------------------------------------------------------------
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)

X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

# --------------------------------------------------------------------------------
# 6. Define the model
# --------------------------------------------------------------------------------
class BaseModel(nn.Module):
    def __init__(self, model_type='LSTM'):
        super(BaseModel, self).__init__()
        self.model_type = model_type
        
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=len(features),
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
            )
            self.fc = nn.Linear(128, 1)
        
        elif model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=len(features),
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
            )
            self.fc = nn.Linear(128, 1)
        
        elif model_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=len(features),
                hidden_size=128,
                num_layers=2,
                nonlinearity='relu',
                batch_first=True,
                dropout=0.1,
            )
            self.fc = nn.Linear(128, 1)

        elif model_type == 'CNN':
            self.conv1 = nn.Conv1d(in_channels=len(features), out_channels=64, kernel_size=3)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
            # Output length after two Conv1d layers (kernel_size=3 each)
            conv_out_len = window_width - 2*(3 - 1)
            self.fc = nn.Linear(128 * conv_out_len, 1)
        
        elif model_type == 'EnhancedLSTM':
            self.rnn = nn.LSTM(
                input_size=len(features),
                hidden_size=128,
                num_layers=3,
                batch_first=True,
                dropout=0.2,
            )
            self.dropout = nn.Dropout(0.2)
            self.bn = nn.BatchNorm1d(128)
            self.fc = nn.Linear(128, 1)

        else:
            raise ValueError("model_type not recognized.")
    
    def forward(self, x):
        if self.model_type == 'CNN':
            # x: (batch_size, seq_length, n_features)
            # Need (batch_size, channels, seq_length)
            x = x.permute(0, 2, 1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)  # flatten
            out = self.fc(x)
        
        elif self.model_type == 'EnhancedLSTM':
            out, _ = self.rnn(x)
            out = out[:, -1, :]  # last timestep
            out = self.bn(out)
            out = self.dropout(out)
            out = self.fc(out)
        
        else:
            out, _ = self.rnn(x)
            out = out[:, -1, :]  # last timestep
            out = self.fc(out)
        
        return out

# --------------------------------------------------------------------------------
# 7. Early Stopping
# --------------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > (self.best_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --------------------------------------------------------------------------------
# 8. Utility to invert scaling for 'Price'
# --------------------------------------------------------------------------------
def inverse_transform(y_scaled):
    """
    Return original scale of predicted 'Price' from a 1D array of scaled values.
    """
    y_scaled = np.array(y_scaled).reshape(-1, 1)
    # Zeros for the rest of features
    zeros = np.zeros((y_scaled.shape[0], len(features) - 1))
    combined = np.hstack((y_scaled, zeros))
    y_inv = scaler.inverse_transform(combined)
    return y_inv[:, 0]

# --------------------------------------------------------------------------------
# 9. Training function: we only compute metrics for train & validation
# --------------------------------------------------------------------------------
def train_and_finetune(model_type='LSTM'):
    model = BaseModel(model_type).to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

    early_stopper = EarlyStopping(patience=50, min_delta=0)
    best_val_loss = float('inf')

    train_loss_history = []
    val_loss_history   = []
    
    best_model_path = f'best_model_{model_type}_finetuned.pt'

    # If you want to reload from a pre-trained model, uncomment:
    # if os.path.exists(best_model_path):
    #     model.load_state_dict(torch.load(best_model_path))
    #     print(f"Loaded existing {model_type} model from {best_model_path}")

    # Control how many epochs you want for fine-tuning
    num_epochs = 300
    for epoch in range(num_epochs):
        # 1) Training
        model.train()
        batch_train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        
        avg_train_loss = np.mean(batch_train_losses)
        train_loss_history.append(avg_train_loss)

        # 2) Validation
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_pred = model(Xb)
                val_loss = criterion(val_pred, yb)
                batch_val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(batch_val_losses)
        val_loss_history.append(avg_val_loss)

        # Step scheduler
        scheduler.step(avg_val_loss)

        # Early Stopping
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"[{model_type}] Early Stopping at epoch={epoch+1}")
            break

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

        # Optional: print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"[{model_type}] Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")

    # After training, load best model
    model.load_state_dict(torch.load(best_model_path))
    print(f"Finished fine-tuning {model_type}. Best Val Loss: {best_val_loss:.6f}")

    # Return the model and training history for further usage
    return model, (train_loss_history, val_loss_history)

# --------------------------------------------------------------------------------
# 10. Train different models or just one
# --------------------------------------------------------------------------------
model_variants = ['LSTM', 'GRU', 'RNN', 'CNN', 'EnhancedLSTM']
best_models = {}
histories = {}

for m in model_variants:
    print(f"\n--- Fine-tuning {m} ---")
    best_model, hist = train_and_finetune(m)
    best_models[m] = best_model
    histories[m] = hist

# --------------------------------------------------------------------------------
# 11. Evaluate on validation set (we DO have labels there) 
# --------------------------------------------------------------------------------
for m in model_variants:
    model = best_models[m]
    model.eval()

    # Predict on validation
    val_preds = []
    with torch.no_grad():
        for Xb, _ in val_loader:
            Xb = Xb.to(device)
            preds = model(Xb)
            val_preds.append(preds.cpu().numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    
    # Invert scaling
    val_preds_unscaled = inverse_transform(val_preds)
    y_val_unscaled     = inverse_transform(y_val_tensor.numpy())

    # Compute MAPE or other metrics on validation
    mape_val = mean_absolute_percentage_error(y_val_unscaled, val_preds_unscaled)
    print(f"[{m}] Validation MAPE: {mape_val*100:.2f}%")

# --------------------------------------------------------------------------------
# 12. Generate predictions on *future* unlabeled data
# --------------------------------------------------------------------------------
# If X_future is empty (no future data in your CSV), skip this. 
# If you do have rows that extend beyond your val_cutoff in the CSV, 
# you can attempt to generate predictions as follows:

for m in model_variants:
    model = best_models[m]
    model.eval()

    if len(X_future) == 0:
        print(f"No future data available in {m}. Skipping predictions.")
        continue

    X_future_tensor = torch.from_numpy(X_future).float().to(device)
    with torch.no_grad():
        future_preds = model(X_future_tensor).cpu().numpy()
    future_preds_unscaled = inverse_transform(future_preds)

    print(f"\n--- {m} predictions for future horizon ---")
    for date_point, pred_price in zip(future_dates, future_preds_unscaled):
        print(f"{date_point.strftime('%Y-%m-%d')}: Predicted Price={pred_price:.2f}")

    # You can also store or plot these future predictions as needed.
    # For example, a quick plot:
    plt.figure(figsize=(10,4))
    plt.plot(future_dates, future_preds_unscaled, label=f'{m} Future Predictions', color='orange')
    plt.title(f'{m} - Future Predictions (Unlabeled)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# --------------------------------------------------------------------------------
# 13. Plot training/validation loss curves for each model
# --------------------------------------------------------------------------------
for m in model_variants:
    train_loss_hist, val_loss_hist = histories[m]
    plt.figure(figsize=(8,4))
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(val_loss_hist, label='Val Loss')
    plt.title(f'{m} Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

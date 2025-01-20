##############################################################################
#  1) Imports and Setup
##############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import random
import warnings
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")  # Hide any warnings for cleaner output

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

##############################################################################
#  2) Load and Prepare Data
##############################################################################
df = pd.read_csv('Gold Price_Year 2017-2019.csv')
df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

num_cols = df.columns.drop(['Date'])
df[num_cols] = df[num_cols].replace({',': ''}, regex=True)
df[num_cols] = df[num_cols].astype('float64')

assert df.duplicated().sum() == 0, "Duplicate rows found!"
assert df.isnull().sum().sum() == 0, "Missing values found!"

features = ['Price', 'Open', 'High', 'Low']
data = df[features].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
window_width = 30
X, y, dates = [], [], []
for i in range(window_width, len(data_scaled)):
    X.append(data_scaled[i - window_width : i])
    y.append(data_scaled[i, 0])  # target: Price
    dates.append(df['Date'].iloc[i])

X = np.array(X)
y = np.array(y)
dates = np.array(dates)

# Define date masks
train_mask = (
    (dates >= pd.to_datetime('2017-01-03')) &
    (dates <= pd.to_datetime('2018-12-31'))
)
val_mask = (
    (dates >= pd.to_datetime('2019-01-02')) &
    (dates <= pd.to_datetime('2019-05-31'))
)
test_mask = (
    (dates >= pd.to_datetime('2019-06-03')) &
    (dates <= pd.to_datetime('2019-12-31'))
)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

train_dates = dates[train_mask]
val_dates = dates[val_mask]
test_dates = dates[test_mask]

# Convert to tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1)
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

##############################################################################
#  3) Define Model
##############################################################################
class BaseModel(nn.Module):
    def __init__(self, model_type='LSTM', hidden_size=128, num_layers=2, dropout=0.1, lr=0.0005):
        super(BaseModel, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=len(features),
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.fc = nn.Linear(hidden_size, 1)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=len(features),
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.fc = nn.Linear(hidden_size, 1)
        elif model_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=len(features),
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity='relu',
                dropout=dropout,
            )
            self.fc = nn.Linear(hidden_size, 1)
        elif model_type == 'CNN':
            self.conv1 = nn.Conv1d(
                in_channels=len(features), out_channels=64, kernel_size=3
            )
            self.conv2 = nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3
            )
            conv_output_length = window_width - 2 * (3 - 1)
            self.fc = nn.Linear(128 * conv_output_length, 1)
        elif model_type == 'EnhancedLSTM':
            self.rnn = nn.LSTM(
                input_size=len(features),
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.bn = nn.BatchNorm1d(hidden_size)
            self.dropout_layer = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)
        else:
            raise ValueError("Invalid model_type. Choose from LSTM, GRU, RNN, CNN, EnhancedLSTM.")

    def forward(self, x):
        if self.model_type == 'CNN':
            # (batch_size, seq_len, channels) -> (batch_size, channels, seq_len)
            x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            out = self.fc(x)
        elif self.model_type == 'EnhancedLSTM':
            out, _ = self.rnn(x)
            out = out[:, -1, :]
            out = self.bn(out)
            out = self.dropout_layer(out)
            out = self.fc(out)
        else:
            out, _ = self.rnn(x)
            out = out[:, -1, :]
            out = self.fc(out)
        return out

##############################################################################
#  4) Utility Classes/Functions
##############################################################################
class EarlyStopping:
    def __init__(self, patience=150, min_delta=0):
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

def inverse_transform(y_scaled):
    y_scaled = np.array(y_scaled).reshape(-1, 1)
    zeros = np.zeros((y_scaled.shape[0], len(features) - 1))
    y_combined = np.hstack((y_scaled, zeros))
    y_inv = scaler.inverse_transform(y_combined)
    return y_inv[:, 0]

def evaluate_on_validation(model, val_loader, criterion):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_losses.append(loss.item())
    return np.mean(val_losses)

def train_model(model, train_loader, val_loader, num_epochs=1000, patience=150):
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=model.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience)
    early_stopping = EarlyStopping(patience=patience, min_delta=0)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
        
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)

        # Validation
        avg_val_loss = evaluate_on_validation(model, val_loader, criterion)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
        
        if early_stopping.early_stop:
            break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_loss, train_losses, val_losses

##############################################################################
#  5) Hyper-Parameter Search
##############################################################################
def hyperparameter_search(model_type, train_loader, val_loader):
    hidden_sizes = [64, 128]
    dropouts = [0.1, 0.2]
    lrs = [0.0005, 0.001]
    num_layers_list = [2, 3]

    best_combination = None
    best_val_loss = float('inf')
    best_train_losses = []
    best_val_losses = []

    for hidden_size in hidden_sizes:
        for dropout in dropouts:
            for lr in lrs:
                for num_layers in num_layers_list:
                    model = BaseModel(
                        model_type=model_type,
                        hidden_size=hidden_size,
                        dropout=dropout,
                        lr=lr,
                        num_layers=num_layers
                    ).to(device)

                    print(f"\n[HP-Search] {model_type} | hidden={hidden_size}, drop={dropout}, lr={lr}, layers={num_layers}")
                    val_loss, train_hist, val_hist = train_model(
                        model, train_loader, val_loader, num_epochs=300, patience=50
                    )
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_combination = {
                            'hidden_size': hidden_size,
                            'dropout': dropout,
                            'lr': lr,
                            'num_layers': num_layers,
                        }
                        best_train_losses = train_hist
                        best_val_losses = val_hist

    print(f"\n[HP-Search] Best combination for {model_type}: {best_combination} with val_loss={best_val_loss:.6f}")
    return best_combination, best_train_losses, best_val_losses

##############################################################################
#  6) Load or Train (Skipping Re-Training If File Exists)
##############################################################################
def get_model_outputs(model_type, force_train=False):
    """
    1) Check if 'extended_prediction_best_model_<model_type>.pt' exists.
       - If yes and force_train=False, load the file directly, skip training.
       - Else, do hyper-param search, train, and save the file.
    2) Return dictionary with predictions, best_params, etc.
    """
    extended_pred_path = f'extended_prediction_best_model_{model_type}.pt'
    
    # If the file exists and we do not want to force re-training, just load
    if (not force_train) and os.path.exists(extended_pred_path):
        print(f"\n[INFO] Found existing {extended_pred_path}. Loading without re-training...")
        loaded_data = torch.load(extended_pred_path, map_location=device)
        
        # Extract info
        best_params = loaded_data['best_params']
        y_val_pred_inv = loaded_data['y_val_pred_inv']
        y_test_pred_inv = loaded_data['y_test_pred_inv']
        train_losses_per_epoch = loaded_data['train_losses_per_epoch']
        val_losses_per_epoch = loaded_data['val_losses_per_epoch']
        
        print(f"[INFO] Loaded best hyper-params for {model_type}:", best_params)
        
        # We can also reconstruct the model if needed
        best_model = BaseModel(
            model_type=model_type,
            hidden_size=best_params['hidden_size'],
            dropout=best_params['dropout'],
            lr=best_params['lr'],
            num_layers=best_params['num_layers']
        ).to(device)
        best_model.load_state_dict(loaded_data['state_dict'])
        best_model.eval()

    else:
        # Otherwise, do hyperparam search -> train -> save
        print(f"\n[INFO] No existing file or force_train=True. Searching hyper-params for {model_type}...")
        best_params, _, _ = hyperparameter_search(model_type, train_loader, val_loader)
        
        # Retrain best model
        best_model = BaseModel(
            model_type=model_type,
            hidden_size=best_params['hidden_size'],
            dropout=best_params['dropout'],
            lr=best_params['lr'],
            num_layers=best_params['num_layers']
        ).to(device)

        print(f"\nRetraining {model_type} with best hyperparams: {best_params}")
        _, train_losses_per_epoch, val_losses_per_epoch = train_model(
            best_model, train_loader, val_loader, num_epochs=800, patience=150
        )

        # Evaluate on test
        criterion = nn.HuberLoss()
        best_model.eval()
        test_losses = []
        y_test_pred_list = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = best_model(X_batch)
                loss = criterion(y_pred, y_batch)
                test_losses.append(loss.item())
                y_test_pred_list.append(y_pred.cpu().numpy())

        y_test_pred_array = np.concatenate(y_test_pred_list, axis=0)
        # Validation predictions
        y_val_pred_list = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                y_val_pred_list.append(best_model(X_batch).cpu().numpy())
        y_val_pred = np.concatenate(y_val_pred_list, axis=0)

        y_val_pred_inv = inverse_transform(y_val_pred)
        y_test_pred_inv = inverse_transform(y_test_pred_array)

        # Save extended file
        save_dict = {
            'state_dict': best_model.state_dict(),
            'best_params': best_params,
            'y_val_pred_inv': y_val_pred_inv,
            'y_test_pred_inv': y_test_pred_inv,
            'train_losses_per_epoch': train_losses_per_epoch,
            'val_losses_per_epoch': val_losses_per_epoch,
        }
        torch.save(save_dict, extended_pred_path)
        print(f"[INFO] Saved best {model_type} model + predictions to {extended_pred_path}")

    # Build final output
    results = {
        'model_name': model_type,
        'best_params': best_params,
        'y_val_pred_inv': y_val_pred_inv,
        'y_test_pred_inv': y_test_pred_inv,
        'train_losses_per_epoch': train_losses_per_epoch,
        'val_losses_per_epoch': val_losses_per_epoch,
    }
    return results

##############################################################################
#  7) Main Flow: Get (Load/Train) Each Model's Output
##############################################################################
model_names = ['LSTM', 'GRU', 'RNN', 'CNN', 'EnhancedLSTM']
all_results = {}

for mname in model_names:
    print(f"\n\n>>> Handling {mname} Model <<<")
    # Set force_train=False to skip re-training if file exists
    # (You can set force_train=True if you want to forcibly re-train.)
    results = get_model_outputs(mname, force_train=False)
    all_results[mname] = results

# Inverse transform the actual validation/test data
y_val_inv = inverse_transform(y_val_tensor.numpy())
y_test_inv = inverse_transform(y_test_tensor.numpy())

##############################################################################
#  8) Plot (Zoomed-in) Predictions on Validation + Test
##############################################################################
model_colors = {
    'LSTM': 'red',
    'GRU': 'blue',
    'RNN': 'green',
    'CNN': 'orange',
    'EnhancedLSTM': 'purple',
}

plt.figure(figsize=(14, 7))
zoom_dates = np.concatenate((val_dates, test_dates))
zoom_actual = np.concatenate((y_val_inv, y_test_inv))

# Plot actual
plt.plot(zoom_dates, zoom_actual, color='black', label='Actual Price')

# Plot predictions
for model_name, results in all_results.items():
    y_pred_combined = np.concatenate((results['y_val_pred_inv'], results['y_test_pred_inv']))
    model_dates = np.concatenate((val_dates, test_dates))
    plt.plot(model_dates, y_pred_combined, color=model_colors[model_name], linestyle='--',
             label=f'{model_name} Prediction')

# Highlight validation and test periods
plt.axvspan(val_dates.min(), val_dates.max(), color='green', alpha=0.1, label='Validation Period')
plt.axvspan(test_dates.min(), test_dates.max(), color='yellow', alpha=0.1, label='Test Period')

plt.title('Zoomed-in Model Predictions on Gold Price (Validation + Test)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

##############################################################################
#  9) Polynomial Fitting (Optional)
#     (Remains identical to your existing step-by-step approach)
##############################################################################
# ... The polynomial fitting code from your previous snippet ...
# Simply reuse the same logic for polynomial fits, derivatives, etc.
# You already have "all_results[model_name]['y_val_pred_inv']" / "['y_test_pred_inv']".
#
# e.g.:
#  - Combine validation+test data
#  - Convert dates to numeric
#  - Fit polynomials with even/odd day splits for training vs validation
#  - Plot polynomial fits at degrees that minimize MSE or maximize R²
#  - Plot derivatives at a chosen degree
#
# [Omitted here for brevity, but you can copy-paste your original polynomial
#  fitting code, referencing all_results similarly.]


##############################################################################
#  10) Polynomial Fitting (Step-by-Step) with All Diagrams and Expressions
##############################################################################
# Combine validation + test data
zoom_dates = np.concatenate((val_dates, test_dates))
zoom_actual = np.concatenate((y_val_inv, y_test_inv))

# Convert datetime objects to numeric for polynomial fitting
date_numeric = mdates.date2num(zoom_dates)

# Define a function to format polynomial coefficients for nice math expressions
def format_polynomial(coeffs):
    """
    Formats polynomial coefficients into a readable mathematical expression.
    Highest-degree term first.
    """
    terms = []
    degree = len(coeffs) - 1
    for i, coef in enumerate(coeffs):
        power = degree - i
        if abs(coef) < 1e-12:
            continue  # skip negligible
        sign_str = " + " if coef >= 0 and i != 0 else ""
        sign_str = " - " if coef < 0 and i != 0 else sign_str
        coef_abs = abs(coef)
        if power == 0:
            term = f"{sign_str}{coef_abs:.4f}"
        elif power == 1:
            term = f"{sign_str}{coef_abs:.4f}x"
        else:
            term = f"{sign_str}{coef_abs:.4f}x^{power}"
        terms.append(term)
    return "".join(terms).strip() if terms else "0"

# Split into (training_mask -> even days) and (validation_mask -> odd days)
training_mask = np.array([d.day % 2 == 0 for d in zoom_dates])  # even day
validation_mask = ~training_mask  # odd day

zoom_dates_train = zoom_dates[training_mask]
zoom_actual_train = zoom_actual[training_mask]
date_numeric_train = date_numeric[training_mask]

zoom_dates_val = zoom_dates[validation_mask]
zoom_actual_val = zoom_actual[validation_mask]
date_numeric_val = date_numeric[validation_mask]

model_names_for_poly = ['LSTM', 'GRU', 'RNN', 'CNN', 'EnhancedLSTM']
models_plus_actual = ['Actual'] + model_names_for_poly

# -------------------------------------------------
# 1) Find degrees that minimize MSE and maximize R² on Validation
# -------------------------------------------------
poly_degrees = range(1, 70)
lowest_val_mse = float('inf')
highest_val_r2 = float('-inf')
best_val_mse_info = None
best_val_r2_info = None

for n in poly_degrees:
    # Actual price
    coeff_actual_train = np.polyfit(date_numeric_train, zoom_actual_train, n)
    poly_actual_val = np.polyval(coeff_actual_train, date_numeric_val)
    val_mse_actual = mean_squared_error(zoom_actual_val, poly_actual_val)
    val_r2_actual  = r2_score(zoom_actual_val, poly_actual_val)

    # Check if best for actual
    if val_mse_actual < lowest_val_mse:
        lowest_val_mse = val_mse_actual
        best_val_mse_info = ("Actual", n, val_mse_actual)
    if val_r2_actual > highest_val_r2:
        highest_val_r2 = val_r2_actual
        best_val_r2_info = ("Actual", n, val_r2_actual)

    # Now check for each ML model
    for model_name in model_names_for_poly:
        y_pred_combined = np.concatenate((
            all_results[model_name]['y_val_pred_inv'],
            all_results[model_name]['y_test_pred_inv']
        ))
        y_pred_train = y_pred_combined[training_mask]
        y_pred_val   = y_pred_combined[validation_mask]

        model_coeff_train = np.polyfit(date_numeric_train, y_pred_train, n)
        model_poly_val    = np.polyval(model_coeff_train, date_numeric_val)
        val_mse_model     = mean_squared_error(y_pred_val, model_poly_val)
        val_r2_model      = r2_score(y_pred_val, model_poly_val)

        if val_mse_model < lowest_val_mse:
            lowest_val_mse = val_mse_model
            best_val_mse_info = (model_name, n, val_mse_model)
        if val_r2_model > highest_val_r2:
            highest_val_r2 = val_r2_model
            best_val_r2_info = (model_name, n, val_r2_model)

print("Lowest Validation MSE ->", best_val_mse_info)
print("Highest Validation R²   ->", best_val_r2_info)

# -------------------------------------------------
# 2) Plot the polynomial fits for the identified degrees
# -------------------------------------------------
def plot_poly_fit_for_degree(n):
    # Training Plot
    plt.figure(figsize=(14, 7))
    plt.plot(zoom_dates_train, zoom_actual_train, 'k*', label='Training Data (Actual)')
    # Actual
    coeff_actual_train = np.polyfit(date_numeric_train, zoom_actual_train, n)
    poly_actual_train = np.polyval(coeff_actual_train, date_numeric_train)
    plt.plot(zoom_dates_train, poly_actual_train, color='black', label=f'{n}-deg Fit (Actual)')

    # ML Models
    for model_name in model_names_for_poly:
        y_pred_combined = np.concatenate((
            all_results[model_name]['y_val_pred_inv'],
            all_results[model_name]['y_test_pred_inv']
        ))
        y_pred_train = y_pred_combined[training_mask]
        coeff_pred_train = np.polyfit(date_numeric_train, y_pred_train, n)
        poly_pred_train = np.polyval(coeff_pred_train, date_numeric_train)
        plt.plot(zoom_dates_train, poly_pred_train, color=model_colors[model_name], linestyle='--',
                 label=f'{n}-deg Fit ({model_name} Training)')

    plt.title(f'(Training) Polynomial Fits (Degree = {n})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend(loc='best')

    # Highlight validation and test periods
    plt.axvspan(val_dates.min(), val_dates.max(), color='green', alpha=0.1, label='Validation Period')
    plt.axvspan(test_dates.min(), test_dates.max(), color='yellow', alpha=0.1, label='Test Period')

    plt.gcf().autofmt_xdate()
    plt.show()

    # Validation Plot
    plt.figure(figsize=(14, 7))
    plt.plot(zoom_dates_val, zoom_actual_val, 'k*', label='Validation Data (Actual)')
    poly_actual_val = np.polyval(coeff_actual_train, date_numeric_val)
    plt.plot(zoom_dates_val, poly_actual_val, color='black', label=f'{n}-deg Fit (Actual)')

    for model_name in model_names_for_poly:
        y_pred_combined = np.concatenate((
            all_results[model_name]['y_val_pred_inv'],
            all_results[model_name]['y_test_pred_inv']
        ))
        y_pred_train = y_pred_combined[training_mask]
        coeff_pred_train = np.polyfit(date_numeric_train, y_pred_train, n)
        poly_pred_val = np.polyval(coeff_pred_train, date_numeric_val)
        plt.plot(zoom_dates_val, poly_pred_val, color=model_colors[model_name], linestyle='--',
                 label=f'{n}-deg Fit ({model_name} Validation)')

    plt.title(f'(Validation) Polynomial Fits (Degree = {n})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend(loc='best')

    # Highlight validation and test periods
    plt.axvspan(val_dates.min(), val_dates.max(), color='green', alpha=0.1, label='Validation Period')
    plt.axvspan(test_dates.min(), test_dates.max(), color='yellow', alpha=0.1, label='Test Period')

    plt.gcf().autofmt_xdate()
    plt.show()

# Gather the best degrees for MSE and R²
degrees_to_plot = set()
if best_val_mse_info is not None:
    degrees_to_plot.add(best_val_mse_info[1])
if best_val_r2_info is not None:
    degrees_to_plot.add(best_val_r2_info[1])

for deg in degrees_to_plot:
    plot_poly_fit_for_degree(deg)

# -------------------------------------------------
# 3) Print Polynomial Expressions for the Best Degrees
# -------------------------------------------------
def print_poly_expressions_for_degree(n):
    print(f"\nDegree {n} Polynomial Expressions:\n{'-'*40}")
    # Actual
    coeff_actual_train = np.polyfit(date_numeric_train, zoom_actual_train, n)
    expr_actual = format_polynomial(coeff_actual_train)
    print(f"Actual Price: y = {expr_actual}")
    
    # Models
    for model_name in model_names_for_poly:
        y_pred_combined = np.concatenate((
            all_results[model_name]['y_val_pred_inv'],
            all_results[model_name]['y_test_pred_inv']
        ))
        y_pred_train = y_pred_combined[training_mask]
        coeff_model_train = np.polyfit(date_numeric_train, y_pred_train, n)
        expr_model = format_polynomial(coeff_model_train)
        print(f"{model_name}: y = {expr_model}")

for deg in degrees_to_plot:
    print_poly_expressions_for_degree(deg)

# -------------------------------------------------
# 4) Plot MSE/R² vs Polynomial Degree (focusing up to 20)
# -------------------------------------------------
plot_max_degree = 20
train_mse_dict = {m: np.zeros(plot_max_degree) for m in models_plus_actual}
val_mse_dict   = {m: np.zeros(plot_max_degree) for m in models_plus_actual}
train_r2_dict  = {m: np.zeros(plot_max_degree) for m in models_plus_actual}
val_r2_dict    = {m: np.zeros(plot_max_degree) for m in models_plus_actual}

for n in range(1, plot_max_degree+1):
    idx = n - 1
    # Actual
    coeff_actual = np.polyfit(date_numeric_train, zoom_actual_train, n)
    poly_train_actual = np.polyval(coeff_actual, date_numeric_train)
    poly_val_actual   = np.polyval(coeff_actual, date_numeric_val)

    train_mse_dict['Actual'][idx] = mean_squared_error(zoom_actual_train, poly_train_actual)
    train_r2_dict['Actual'][idx]  = r2_score(zoom_actual_train, poly_train_actual)
    val_mse_dict['Actual'][idx]   = mean_squared_error(zoom_actual_val, poly_val_actual)
    val_r2_dict['Actual'][idx]    = r2_score(zoom_actual_val, poly_val_actual)

    # ML models
    for model_name in model_names_for_poly:
        y_pred_combined = np.concatenate((
            all_results[model_name]['y_val_pred_inv'],
            all_results[model_name]['y_test_pred_inv']
        ))
        y_pred_train = y_pred_combined[training_mask]
        y_pred_val = y_pred_combined[validation_mask]

        coeff_model = np.polyfit(date_numeric_train, y_pred_train, n)
        poly_train_model = np.polyval(coeff_model, date_numeric_train)
        poly_val_model   = np.polyval(coeff_model, date_numeric_val)

        train_mse_dict[model_name][idx] = mean_squared_error(y_pred_train, poly_train_model)
        train_r2_dict[model_name][idx]  = r2_score(y_pred_train, poly_train_model)
        val_mse_dict[model_name][idx]   = mean_squared_error(y_pred_val, poly_val_model)
        val_r2_dict[model_name][idx]    = r2_score(y_pred_val, poly_val_model)

def set_focused_ylim(data_dict, buffer_ratio=0.05):
    all_values = np.concatenate(list(data_dict.values()))
    ymin = np.min(all_values)
    ymax = np.max(all_values)
    buffer = (ymax - ymin) * buffer_ratio
    return ymin - buffer, ymax + buffer

degrees_arr = np.arange(1, plot_max_degree + 1)

# Plot MSE - Training
plt.figure(figsize=(14, 7))
plt.title("Polynomial Degree vs MSE (Training Data)")
for m in models_plus_actual:
    plt.plot(degrees_arr, train_mse_dict[m], label=m)
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.xticks(degrees_arr)
plt.legend(loc='best')
plt.grid(True)
ymin, ymax = set_focused_ylim(train_mse_dict)
plt.ylim(ymin, ymax)
plt.show()

# Plot MSE - Validation
plt.figure(figsize=(14, 7))
plt.title("Polynomial Degree vs MSE (Validation Data)")
for m in models_plus_actual:
    plt.plot(degrees_arr, val_mse_dict[m], label=m)
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.xticks(degrees_arr)
plt.legend(loc='best')
plt.grid(True)
ymin, ymax = set_focused_ylim(val_mse_dict)
plt.ylim(ymin, ymax)
plt.show()

# Plot R² - Training
plt.figure(figsize=(14, 7))
plt.title("Polynomial Degree vs R² (Training Data)")
for m in models_plus_actual:
    plt.plot(degrees_arr, train_r2_dict[m], label=m)
plt.xlabel("Polynomial Degree")
plt.ylabel("R²")
plt.xticks(degrees_arr)
plt.legend(loc='best')
plt.grid(True)
ymin, ymax = set_focused_ylim(train_r2_dict)
plt.ylim(ymin, ymax)
plt.show()

# Plot R² - Validation
plt.figure(figsize=(14, 7))
plt.title("Polynomial Degree vs R² (Validation Data)")
for m in models_plus_actual:
    plt.plot(degrees_arr, val_r2_dict[m], label=m)
plt.xlabel("Polynomial Degree")
plt.ylabel("R²")
plt.xticks(degrees_arr)
plt.legend(loc='best')
plt.grid(True)
ymin, ymax = set_focused_ylim(val_r2_dict)
plt.ylim(ymin, ymax)
plt.show()

# -------------------------------------------------
# 5) Polynomial Gradients (Derivatives) 
#    for Validation at Degree = 29
# -------------------------------------------------
degree_for_derivative = 29
plt.figure(figsize=(14, 7))

# Fit polynomial for Actual Price
coeff_actual_train = np.polyfit(date_numeric_train, zoom_actual_train, degree_for_derivative)
deriv_coeff_actual = np.polyder(coeff_actual_train)
val_deriv_actual = np.polyval(deriv_coeff_actual, date_numeric_val)
plt.plot(zoom_dates_val, val_deriv_actual, 'k-', label='Actual Derivative')

# Fit polynomial for Each Model
for model_name in model_names_for_poly:
    y_pred_combined = np.concatenate((
        all_results[model_name]['y_val_pred_inv'],
        all_results[model_name]['y_test_pred_inv']
    ))
    y_pred_train = y_pred_combined[training_mask]
    coeff_model_train = np.polyfit(date_numeric_train, y_pred_train, degree_for_derivative)
    deriv_coeff_model = np.polyder(coeff_model_train)
    val_deriv_model = np.polyval(deriv_coeff_model, date_numeric_val)
    plt.plot(zoom_dates_val, val_deriv_model, color=model_colors[model_name], label=f'{model_name} Derivative')

plt.title(f'Polynomial Gradients (Derivatives) for Validation at Degree = {degree_for_derivative}')
plt.xlabel('Date')
plt.ylabel('Gradient')
plt.grid(True)
# Highlight validation and test periods
plt.axvspan(val_dates.min(), val_dates.max(), color='green', alpha=0.1, label='Validation Period')
plt.axvspan(test_dates.min(), test_dates.max(), color='yellow', alpha=0.1, label='Test Period')
plt.legend(loc='best')
plt.gcf().autofmt_xdate()
plt.show()

# Print derivative polynomial expressions
def format_polynomial_derivative(coeffs):
    return format_polynomial(coeffs)

print(f"\nDerivative Polynomials (Degree = {degree_for_derivative - 1}) at n = {degree_for_derivative}:\n{'-'*60}")
print(f"Actual Price Derivative: dy/dx = {format_polynomial_derivative(deriv_coeff_actual)}")

for model_name in model_names_for_poly:
    y_pred_combined = np.concatenate((
        all_results[model_name]['y_val_pred_inv'],
        all_results[model_name]['y_test_pred_inv']
    ))
    y_pred_train = y_pred_combined[training_mask]
    coeff_model_train = np.polyfit(date_numeric_train, y_pred_train, degree_for_derivative)
    deriv_coeff_model = np.polyder(coeff_model_train)
    expr_deriv = format_polynomial_derivative(deriv_coeff_model)
    print(f"{model_name} Derivative: dy/dx = {expr_deriv}")

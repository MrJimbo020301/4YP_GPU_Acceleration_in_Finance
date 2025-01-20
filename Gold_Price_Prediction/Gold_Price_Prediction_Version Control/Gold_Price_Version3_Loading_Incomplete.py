import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def polynomial_fitting_for_gold_price(max_degree=16):
    """
    Fits an n-th degree polynomial to the TRAINING portion of the gold price data,
    then predicts/plots the results for both validation and test sets.
    """
    # Convert each dates array (NumPy) into ordinal integers
    x_train = np.array([d.toordinal() for d in train_dates])
    x_val   = np.array([d.toordinal() for d in val_dates])
    x_test  = np.array([d.toordinal() for d in test_dates])
    zoom_dates = np.concatenate((val_dates, test_dates))
    x_val_test = np.array([d.toordinal() for d in zoom_dates])

    # Inverse-transform the scaled y-values to original price scale
    y_train_inv = inverse_transform(y_train_tensor.numpy())
    y_val_inv   = inverse_transform(y_val_tensor.numpy())
    y_test_inv  = inverse_transform(y_test_tensor.numpy())
    zoom_actual = np.concatenate((y_val_inv, y_test_inv))

    plt.figure(figsize=(14, 7))

    # Plot actual data
    plt.plot(val_dates,  y_val_inv,   'k*', label='Validation Actual')
    plt.plot(test_dates, y_test_inv,  'b*', label='Test Actual')

    for n in range(1, max_degree + 1):
        # Fit polynomial on training set only
        coeffs = np.polyfit(x_val_test, zoom_actual, deg=n)

        # Evaluate polynomial on validation and test sets
        y_val_pred  = np.polyval(coeffs, x_val)
        y_test_pred = np.polyval(coeffs, x_test)
        y_val_test_pred = np.polyval(coeffs, x_val_test)
        
        label_str = f'Degree {n}'
        plt.plot(zoom_dates, y_val_test_pred, label=label_str)

    # Highlight the validation & test periods
    plt.axvspan(val_dates.min(), val_dates.max(), color='green', alpha=0.1, label='Validation Period')
    plt.axvspan(test_dates.min(), test_dates.max(), color='yellow', alpha=0.1, label='Test Period')

    plt.title(f'Polynomial Fitting for Gold Price (1 to {max_degree})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

polynomial_fitting_for_gold_price(max_degree=5)
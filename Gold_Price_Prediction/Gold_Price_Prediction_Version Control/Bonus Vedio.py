    
import numpy as np
import matplotlib.pyplot as plt

# Assume x and y are defined arrays representing your data
# For example:
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.1 * np.random.randn(len(x))  # Noisy sine wave

for n in range(1, 17):
    # Fit an nth degree polynomial to the data
    p = np.polyfit(x, y, n)

    # Evaluate the polynomial at the points of our data
    y1 = np.polyval(p, x)

    # Create a subplot
    plt.subplot(3, 1, 1)

    # Plot the noisy data
    plt.plot(x, y, '*', label="Noisy Data")
    
    # Plot the polynomial fit
    plt.plot(x, y1, 'o', label=f"Polynomial Degree {n}")
    plt.ylim([-2, 2])

    # Add title, labels, and text
    plt.title('Training data and model predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    txt = f'Degree of polynomial = {n}'
    plt.text(7.5, 1.5, txt)

    # Show the plot and pause
    plt.legend()
    plt.pause(0.5)
    plt.clf()

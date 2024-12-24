import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the second-order system model with a fixed time offset
def second_order_system(params, t, data):
    A, zeta, wn, phi = params
    wd = wn * np.sqrt(1 - zeta**2)
    fitted_curve = A * np.exp(-zeta * wn * t) * np.cos(wd * t + phi)
    return fitted_curve - data  # Difference between the fitted curve and the data

# Load data from a text file
data = np.loadtxt('data2.txt', delimiter='\t')

# Extract time and position data
time = data[:, 0]
position = data[:, 1]

# Improve initial guesses and parameter bounds
initial_A = (max(position) - min(position)) / 2

# Estimate initial guesses for wn and zeta
peak_index = np.argmax(position)
min_index = np.argmin(position)
initial_wn = 2 * np.pi / (time[peak_index] - time[min_index])
initial_zeta = 0.01  # Modify this as needed
initial_phi = 0  # Initial guess for phase angle

# Define bounds for parameters
bounds = [(0, np.inf), (0, 1), (0, np.inf), (-np.pi, np.pi)]

# Create initial parameter values
initial_params = [initial_A, initial_zeta, initial_wn, initial_phi]

# Define a custom least-squares loss function without clipping
def custom_loss(params):
    fitted = second_order_system(params, time, position)
    return np.sum(fitted**2)

# Optimize the parameters using the custom loss function and bounds
result = minimize(custom_loss, initial_params, bounds=bounds, method='L-BFGS-B')

# Extract the optimized parameters
A_fit, zeta_fit, wn_fit, phi_fit = result.x

# Calculate the damped angular frequency (wd)
wd_fit = wn_fit * np.sqrt(1 - zeta_fit**2)

# Output the results
print(f"Amplitude (A): {A_fit}")
print(f"Damping Ratio (ζ): {zeta_fit}")
print(f"Natural Frequency (wn): {wn_fit}")
print(f"Damped Angular Frequency (wd): {wd_fit}")
print(f"Phase Angle (φ): {phi_fit}")

# Generate the fitted curve
fitted_curve = second_order_system([A_fit, zeta_fit, wn_fit, phi_fit], time, np.zeros_like(time)) + position

# Calculate and plot the squared error for each data point
squared_error = (fitted_curve - position) ** 2

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)  # Create a subplot for data and fitted curve
plt.scatter(time, position, label='Data')
plt.plot(time, fitted_curve, label='Fitted Curve', color='red')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()

plt.subplot(2, 1, 2)  # Create a subplot for the squared error
plt.plot(time, squared_error, label='Squared Error', color='blue')
plt.xlabel('Time')
plt.ylabel('Squared Error')
plt.legend()

plt.tight_layout()
plt.show()

'''
This code works for more accuratelyfor data like this 
Amplitude (A): 0.4364351623448034
Damping Ratio (ζ): 0.40000054973574084
Natural Frequency (wn): 50.00001341404367
Damped Angular Frequency (wd): 45.82575724750211
Phase Angle (φ): -0.41151957296255265


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the second-order system model
def second_order_system(params, t):
    A, zeta, wn, phi = params
    wd = wn * np.sqrt(1 - zeta**2)
    fitted_curve = A * np.exp(-zeta * wn * t) * np.cos(wd * t + phi)
    return fitted_curve

# Load data from a text file
data = np.loadtxt('data2.txt', delimiter='\t')

# Extract time and position data
time = data[:, 0]
position = data[:, 1]

# Initial guesses and parameter bounds
initial_A = (max(position) - min(position)) / 2
initial_zeta = 0.01  # Modify as needed
initial_wn = 2 * np.pi / (time[np.argmax(position)] - time[np.argmin(position)])
initial_phi = 0

# Define bounds for parameters
bounds = [(0, np.inf), (0, 1), (0, np.inf), (-np.pi, np.pi)]

# Create initial parameter values
initial_params = [initial_A, initial_zeta, initial_wn, initial_phi]

# Define a custom least-squares loss function
def custom_loss(params):
    fitted = second_order_system(params, time)
    error = position - fitted
    return np.sum(error**2)

# Optimize the parameters using the 'L-BFGS-B' optimization method with bounds
result = minimize(custom_loss, initial_params, bounds=bounds, method='L-BFGS-B')

# Extract the optimized parameters
A_fit, zeta_fit, wn_fit, phi_fit = result.x

# Calculate the damped angular frequency (wd)
wd_fit = wn_fit * np.sqrt(1 - zeta_fit**2)

# Output the results
print(f"Amplitude (A): {A_fit}")
print(f"Damping Ratio (ζ): {zeta_fit}")
print(f"Natural Frequency (wn): {wn_fit}")
print(f"Damped Angular Frequency (wd): {wd_fit}")
print(f"Phase Angle (φ): {phi_fit}")

# Generate the fitted curve
fitted_curve = second_order_system([A_fit, zeta_fit, wn_fit, phi_fit], time)

# Plot the original data and the fitted curve
plt.scatter(time, position, label='Data')
plt.plot(time, fitted_curve, label='Fitted Curve', color='red')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.show()





'''


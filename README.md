# Second-Order System Fitting

## Overview

This project involves fitting a second-order system model to time-series data using Python. The system model is optimized using the scipy minimize function, and the results are visualized, including the fitted curve and squared error.

## Features

- **Second-Order System Model:** Implements a second-order system model with adjustable parameters.
- **Optimization:** Utilizes scipy's minimize function for optimizing the parameters based on the given data.
- **Visualization:** Generates plots to visualize the original data, the fitted curve, and the squared error.

## Requirements

- Python 3.x
- numpy
- scipy
- matplotlib

## Usage

1. Ensure that you have Python installed.

2. Install the required dependencies:

   ```bash
   pip install numpy scipy matplotlib
   ```

3. Run the script:

   ```bash
   python second_order_system_fitting.py
   ```

4. View the console output for the optimized parameters and visualizations.

## Project Structure

- `second_order_system_fitting.py`: Main script implementing the second-order system fitting.
- `data2.txt`: Input data file containing time and position values.

## Configuration

- Modify the `data2.txt` file to use your own dataset.
- Adjust the initial guesses and parameter bounds as needed in the script.

## Results

The console output will display the optimized parameters for the second-order system model. Plots will be generated to visualize the original data, the fitted curve, and the squared error.

## Acknowledgments

This project is inspired by the need to fit second-order system models to time-series data for various applications.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests.

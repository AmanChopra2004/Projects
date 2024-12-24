# Capital Asset Pricing Model (CAPM) Return Analysis

This GitHub repository consists of a Streamlit web application (`CAPM_return.py`) and a Python module (`CAPM_functions.py`) for conducting return analysis using the Capital Asset Pricing Model (CAPM). The project focuses on providing users with insights into the expected returns of selected stocks based on the CAPM formula.

## Streamlit Web Application (`CAPM_return.py`)

### Overview

The Streamlit web application allows users to input parameters, select stocks, and view calculated beta values and expected returns using the CAPM. The application provides a user-friendly interface for financial analysis.

### Features

1. **User Input:**
   - Users can choose up to four stocks from a predefined list and specify the number of years for historical data retrieval.

2. **Data Retrieval:**
   - Historical stock data for the selected stocks and the S&P 500 index is retrieved using the Yahoo Finance API and the FRED API.

3. **Data Analysis and Visualization:**
   - Daily returns, beta values, and expected returns based on the CAPM are calculated and displayed.
   - Relevant financial data is presented in a DataFrame, and stock prices are visualized using interactive charts.

4. **Page Configuration:**
   - The Streamlit app is configured with a title, icon, and a wide layout for better user experience.

## Python Module (`CAPM_functions.py`)

### Overview

The Python module contains functions required for financial analysis, including interactive plotting using Plotly Express and calculations for normalizing data, daily returns, and beta values.

### Functions

1. `interactive_plot(df)`: Generates an interactive line chart for a DataFrame (`df`), plotting multiple series.

2. `normalize(df_2)`: Normalizes the values of each column in a DataFrame (`df_2`) by dividing each value by the first value in its respective column.

3. `daily_return(df)`: Calculates the daily return for each column in a DataFrame (`df`).

4. `calculate_beta(stocks_daily_return, stock)`: Calculates beta values for a given stock using its daily returns and the daily returns of the S&P 500 index.

## How to Run

1. Install the required Python packages using:

   ```bash
   pip install plotly pandas yfinance pandas-datareader streamlit numpy
   ```

2. Run the Streamlit app:

   ```bash
   streamlit run CAPM_return.py
   ```

3. Explore and analyze expected returns for selected stocks based on the CAPM.

Feel free to customize and extend this project to suit your specific financial analysis needs. If you encounter any issues, please raise issue to let me know.

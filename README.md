# Bike-Sharing Departure Prediction and Feature Analysis

A machine learning project to forecast demand for bike-sharing services in Bergen using historical data and a Random Forest model. This project analyzes the factors influencing station usage to help optimize logistics operations.

---

## Overview

The goal of this project is to build a reliable predictive model to estimate the number of bike departures from a given station. By leveraging historical trip data, weather information, and temporal features, we train a Random Forest model to:
*   **Forecast future demand** to anticipate bike availability needs.
*   **Identify the most influential factors** driving station usage.
*   **Provide clear visualizations** to interpret model performance and usage dynamics.

---

## Project Structure

```
ccu-ml-project/
│
├── main.py                 # Main script for training and evaluation
├── requirements.txt        # Project dependencies
├── .gitignore
├── src/
│   └── datas.py            # Data loading and preprocessing module
└── data/
    └── historical/
        ├── 2024_station_event_1h.parquet  # Hourly aggregated data
        └── 2024_trips.csv                 # Raw trip data
```

---

## Data

The model is powered by two main data sources, enhanced with feature engineering.

*   **Parquet Files**: Hourly aggregated event data, including departures, arrivals, weather information, and calendar details.
*   **CSV File**: Raw trip data containing timestamps and station IDs.

#### **Derived Features**

To improve predictive performance, the following features were created:

*   **Temporal**: `hour`, `day_of_week`, `month`, `week_of_year`.
*   **Cyclical & Event-Based**: `is_weekend`, `is_holiday`, `day_type`.
*   **Historical (Lag & Rolling)**: `prev_departures` (departures from the previous hour), `rolling_mean_departure` (3-hour rolling average).
*   **Statistical**: `station_avg_departure` (average departures for a station at a given hour).
*   **Interactions & Non-linearities**: `temperature_squared`, `hour_x_weekend`.

---

## Model: Random Forest

*   **Algorithm**: A Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`) was chosen for its robustness, ability to handle complex non-linear relationships, and built-in feature importance calculation.
*   **Evaluation Metrics**:
    *   **Mean Squared Error (MSE)**: Measures the average squared difference between the estimated values and the actual value.
    *   **Coefficient of Determination (R² Score)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

---

## Results Analysis and Visualizations

The model's performance analysis reveals accurate predictions and key insights into usage drivers.

#### 1. **Feature Importance**
The chart below shows that recent departure history (`rolling_mean_departure`, `station_avg_departure`) is by far the most powerful predictor. Temporal features like the hour (`hour`) play a secondary but significant role, while weather conditions have a more modest impact.

![A horizontal bar chart showing "Feature Importances". The most important feature is 'rolling_mean_departure' with an importance of over 0.5. This is followed by 'station_avg_departure', and 'prev_departures'. Other features like 'hour', 'week_of_year', and various temperature and date-related features have significantly lower importance.](https://github.com/AstraDev0/Bike-Rental-Departure-Prediction-Random-Forest-Feature-Analysis/raw/main/images/user_image_1.png)

#### 2. **Predictions vs. Actual Values by Hour**
The model successfully captures the daily dynamics, especially the two departure peaks corresponding to morning (around 7 AM) and afternoon (around 3 PM) rush hours. The predictions closely follow the actual trend, validating the model's ability to understand urban mobility patterns.

![A line graph comparing "Average Actual vs Predicted Departures by Hour". The x-axis represents the 'Hour of Day' from 0 to 23, and the y-axis shows 'Average Departures'. The blue line with circles represents 'Actual Departures', and the orange dashed line with 'x' marks represents 'Predicted Departures'. Both lines show two major peaks, one in the morning around 6-7 AM and another in the afternoon around 3 PM, indicating typical commute times. The predicted values closely follow the trend of the actual values.](https://github.com/AstraDev0/Bike-Rental-Departure-Prediction-Random-Forest-Feature-Analysis/raw/main/images/user_image_2.png)

#### 3. **Distribution of Residuals**
The histogram of the residuals (prediction errors) is centered around zero and roughly follows a normal distribution. This indicates that the model has no systematic bias (it does not consistently overestimate or underestimate) and that its errors are evenly distributed.

![A histogram with a density curve showing the "Distribution of Residuals". The x-axis is 'Residual (Actual - Predicted)' and the y-axis is 'Count'. The distribution is roughly bell-shaped and centered around 0, indicating that the model's errors are somewhat normally distributed with a slight tendency to underestimate (a peak just to the left of zero). Most residuals fall between -2 and 2.](https://github.com/AstraDev0/Bike-Rental-Departure-Prediction-Random-Forest-Feature-Analysis/raw/main/images/user_image_3.png)

---

## Usage

1.  **Clone the repository**
    ```bash
    git clone https://github.com/AstraDev0/Bike-Rental-Departure-Prediction-Random-Forest-Feature-Analysis
    cd Bike-Rental-Departure-Prediction-Random-Forest-Feature-Analysis
    ```

2.  **Install dependencies** (using a virtual environment is recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run the main script**
    ```bash
    python main.py
    ```

4.  **Outputs**
    *   The visualization plots will be generated and displayed.
    *   The MSE and R² scores will be printed to the console.

---

## Customization

The project is designed to be modular:
*   **Change Station**: Modify the `station_id` variable in `main.py` to analyze a different station.
*   **Experiment with Features**: Add or remove features in the `src/datas.py` module to test new hypotheses.
*   **Test Other Models**: Replace the `RandomForestRegressor` with another algorithm from Scikit-Learn.

---

## Technical Requirements

*   Python 3.11+
*   pandas
*   scikit-learn
*   matplotlib
*   seaborn
*   pyarrow

---

## Author

**Matthis Brocheton**
Artificial Intelligence & Data Science Student
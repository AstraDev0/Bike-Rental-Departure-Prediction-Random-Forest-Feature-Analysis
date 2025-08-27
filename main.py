import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.datas import Datasets
import pandas as pd

def plot_feature_importances(importances: pd.Series):
    importances.sort_values().plot(kind="barh", figsize=(8, 6), title="Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_departures_over_time(y_test, y_pred, hours, label="Hour of Day"):
    df_plot = pd.DataFrame({
        "hour": hours.values,
        "actual": y_test.values,
        "predicted": y_pred
    })

    hourly_avg = df_plot.groupby("hour").mean()

    plt.figure(figsize=(10, 6))
    plt.plot(hourly_avg.index, hourly_avg["actual"], label="Actual Departures", marker="o")
    plt.plot(hourly_avg.index, hourly_avg["predicted"], label="Predicted Departures", marker="x", linestyle="--")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Departures")
    plt.title("Average Actual vs Predicted Departures by Hour")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Distribution of Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main() -> None:
    df = Datasets().read_parquet('data/historical/2024_station_event_1h.parquet')

    station = df[df["station_id"] == "220"]

    y = station["departure"]
    X = station[[
        "precipitation", "temperature", "is_holiday",
        "hour", "day_of_week", "month", "week_of_year",
        "is_weekend", "prev_departures", "rolling_mean_departure",
        "station_avg_departure", "day_type", "temperature_squared",
        "hour_x_weekend"
    ]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, train_size=0.8
    )

    rf = RandomForestRegressor(n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    plot_feature_importances(importances)
    plot_departures_over_time(y_test, y_pred, X_test['hour'])
    plot_residuals(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"R² Score: {r2:.3f}")

if __name__ == "__main__":
    main()
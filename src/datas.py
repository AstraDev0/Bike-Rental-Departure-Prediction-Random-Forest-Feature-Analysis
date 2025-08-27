import pyarrow.parquet as pq
import pandas as pd

class Datasets():
    def __init__(self):
        pass

    def display_parquet_files(self, filepath: str) -> None:
        table = pq.read_table(filepath)
        print("Schema:")
        print(table.schema)

        print(f"\nShape: ({table.num_rows}, {table.num_columns})")

        print("\nColumn names:")
        print(table.column_names)

        print("\nTable:")
        df = table.to_pandas()
        print(df)

    def display_csv_files(self, filepath: str) -> None:
        df = pd.read_csv(filepath)
        print("DataFrame:")
        print(df)

        print(f"\nShape: {df.shape}")

        print("\nColumn names:")
        print(df.columns.tolist())
    
    def read_parquet(self, filepath: str) -> pd.DataFrame:
        df = pd.read_parquet(filepath)

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.dropna(subset=['timestamp'])

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        df = df.sort_values(by=["station_id", "timestamp"])
        df["prev_departures"] = df.groupby("station_id", observed=False)["departure"].shift(1)

        df["rolling_mean_departure"] = (
            df.groupby("station_id")["departure"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["station_avg_departure"] = df.groupby(["station_id", "hour"])["departure"].transform("mean")

        def classify_day(row):
            if row["is_holiday"]:
                return 0
            elif row["is_weekend"]:
                return 1
            else:
                return 2
        df["day_type"] = df.apply(classify_day, axis=1)

        df["temperature_squared"] = df["temperature"] ** 2

        df["hour_x_weekend"] = df["hour"] * df["is_weekend"]

        return df


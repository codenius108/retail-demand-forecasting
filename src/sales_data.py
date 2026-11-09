"""
Retail Demand Forecasting - Sales Data Preprocessing
Author: Person 1 (Sales data & baseline model)

Rule for assigning cities to stores:
Assign each store a city with:
city = cities[store_id % 5], where cities is the list in data/city_map.csv.
"""

from pathlib import Path
import pandas as pd


def load_and_clean_sales_data():
    # -------------------------------------------------------------
    # Step 1: Check that raw data file exists
    # -------------------------------------------------------------
    raw_path = Path("data/raw/train.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    print("Step 1: Found raw data file at", raw_path)

    # -------------------------------------------------------------
    # Step 2: Load raw sales data
    # -------------------------------------------------------------
    # Use low_memory=False and specify StateHoliday as string to avoid mixed types
    df = pd.read_csv(
        raw_path,
        dtype={"StateHoliday": str},
        low_memory=False
    )
    print(f"Step 2: Loaded raw data with {len(df):,} rows.")

    # -------------------------------------------------------------
    # Step 3: Clean data (rename, parse dates, filter closed, drop duplicates)
    # -------------------------------------------------------------
    # Rename columns to standardized names
    df = df.rename(columns={
        "Store": "store_id",
        "Date": "date",
        "Sales": "sales",
        "Promo": "promo"
    })

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Remove rows where store was closed (Open == 0)
    rows_before_open_filter = len(df)
    df = df[df["Open"] != 0]
    rows_after_open_filter = len(df)
    print(f"Removed {rows_before_open_filter - rows_after_open_filter:,} closed store rows.")

    # Remove duplicates
    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Removed {duplicates_count} duplicate rows.")

    # -------------------------------------------------------------
    # Step 4: Assign cities based on data/city_map.csv
    # -------------------------------------------------------------
    # Read the city list in order from data/city_map.csv
    city_map_path = Path("data/city_map.csv")
    if not city_map_path.exists():
        raise FileNotFoundError(f"City map file not found at: {city_map_path}")

    city_map = pd.read_csv(city_map_path)
    cities = city_map["city"].tolist()

    # Rule: city = cities[store_id % 5]
    city_lookup = {i: cities[i] for i in range(len(cities))}
    df["city"] = (df["store_id"] % len(cities)).map(city_lookup)
    print("Step 4: Assigned cities to stores based on data/city_map.csv.")

    # -------------------------------------------------------------
    # Step 5: Keep 50 stores (10 per city: store_id 1 to 50)
    # -------------------------------------------------------------
    # Store IDs 1 through 50 give exactly 10 stores per city (modulos 0, 1, 2, 3, 4)
    selected_stores = list(range(1, 51))
    df = df[df["store_id"].isin(selected_stores)].copy()

    # Sort values cleanly by store and date
    df = df.sort_values(by=["store_id", "date"]).reset_index(drop=True)
    print(f"Step 5: Filtered to {df['store_id'].nunique()} stores. Total rows: {len(df):,}")

    # -------------------------------------------------------------
    # Step 6: Select required columns and save cleaned CSV
    # -------------------------------------------------------------
    # Keep only the requested columns
    final_columns = [
        "date",
        "store_id",
        "city",
        "sales",
        "promo",
        "StateHoliday",
        "SchoolHoliday"
    ]
    df = df[final_columns].copy()

    # Ensure output directory exists
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sales_clean.csv"

    # Save to CSV
    df.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    print(f"Step 6: Saved cleaned sales data to {output_path}")

    return df


if __name__ == "__main__":
    df_cleaned = load_and_clean_sales_data()

    print("\n" + "=" * 50)
    print("CLEANED SALES DATA SUMMARY (Steps 1 to 6)")
    print("=" * 50)

    print("\n1. Final column list:")
    print(list(df_cleaned.columns))

    print(f"\n2. Total row count: {len(df_cleaned):,}")

    print("\n3. Rows per city:")
    print(df_cleaned["city"].value_counts())

    print("\n4. Missing values per column:")
    print(df_cleaned.isnull().sum())

"""
Retail Demand Forecasting - Data Merging Pipeline
Author: Person 4 (Merging the data)

Combines three clean data sources into a single modeling table:
1. Sales data (data/processed/sales_clean.csv)
2. Weather data (data/processed/weather_clean.csv)
3. Social media trends data (data/processed/trends_clean.csv)
"""

from pathlib import Path
import pandas as pd


def merge_all_data():
    # -------------------------------------------------------------
    # Step 1: Read all cleaned datasets
    # -------------------------------------------------------------
    sales_path = Path("data/processed/sales_clean.csv")
    weather_path = Path("data/processed/weather_clean.csv")
    trends_path = Path("data/processed/trends_clean.csv")

    for path in [sales_path, weather_path, trends_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required cleaned data file not found: {path}")

    sales = pd.read_csv(sales_path, parse_dates=["date"])
    weather = pd.read_csv(weather_path, parse_dates=["date"])
    trends = pd.read_csv(trends_path, parse_dates=["date"])

    print("Step 1: Successfully read all cleaned datasets.")
    print(f"  Sales rows:   {len(sales):,} (dates: {sales['date'].min().date()} to {sales['date'].max().date()})")
    print(f"  Weather rows: {len(weather):,} (dates: {weather['date'].min().date()} to {weather['date'].max().date()})")
    print(f"  Trends rows:  {len(trends):,} (dates: {trends['date'].min().date()} to {trends['date'].max().date()})")

    # -------------------------------------------------------------
    # Step 2: Merge sales and weather on date and city
    # -------------------------------------------------------------
    # Left merge ensures sales is the base table and no store days are lost
    sales_rows_before = len(sales)
    merged = sales.merge(weather, on=["date", "city"], how="left")
    sales_rows_after = len(merged)

    print("\nStep 2: Merged sales and weather on date AND city.")
    print(f"  Row count before: {sales_rows_before:,}")
    print(f"  Row count after:  {sales_rows_after:,}")
    if sales_rows_before != sales_rows_after:
        raise ValueError(f"Row count changed during weather merge: {sales_rows_before} != {sales_rows_after}")

    # -------------------------------------------------------------
    # Step 3: Merge trends without future leakage (merge_asof backward)
    # -------------------------------------------------------------
    # Why we shift trends by 7 days and use merge_asof backward:
    # Google Trends data is reported as weekly aggregates at the end of each week.
    # While a week is underway, store managers do not yet have that week's search total.
    # If we used the current week's trend value, we would leak future data into today's sales.
    # By shifting trends date forward by 7 days, the trend data only becomes available
    # after that week has ended. pd.merge_asof with direction="backward" then matches
    # each day to the most recently completed week's trend.
    trends_shifted = trends.copy()
    trends_shifted["date"] = trends_shifted["date"] + pd.Timedelta(days=7)

    # Sort both datasets by date before merge_asof
    merged_sorted = merged.sort_values("date").reset_index(drop=True)
    trends_shifted_sorted = trends_shifted.sort_values("date").reset_index(drop=True)

    merged = pd.merge_asof(
        merged_sorted,
        trends_shifted_sorted,
        on="date",
        direction="backward"
    )
    print("\nStep 3: Merged weekly trends using 7-day shift and merge_asof(direction='backward').")
    print("  Trends features added:", ["trend_drogerie", "trend_rossmann", "trend_angebote"])

    # -------------------------------------------------------------
    # Step 4: Add sales_lag7 by DATE (not shift(7))
    # -------------------------------------------------------------
    # Because retail stores close on Sundays and public holidays, shift(7)
    # would look back 7 open business days rather than 7 calendar days.
    # Adding 7 days to the date column guarantees we compare against the
    # exact same day of the week from last week.
    df_lag = merged[["store_id", "date", "sales"]].copy()
    df_lag["date"] = df_lag["date"] + pd.Timedelta(days=7)
    df_lag = df_lag.rename(columns={"sales": "sales_lag7"})

    merged = merged.merge(df_lag, on=["store_id", "date"], how="left")
    print("\nStep 4: Added sales_lag7 by exact calendar date matching.")

    # -------------------------------------------------------------
    # Step 5: Add calendar features and drop missing values
    # -------------------------------------------------------------
    merged["dayofweek"] = merged["date"].dt.dayofweek
    merged["month"] = merged["date"].dt.month

    rows_before_drop = len(merged)
    dropped_mask = merged.isnull().any(axis=1)
    dropped_rows = merged[dropped_mask]
    dropped_count = len(dropped_rows)

    dropped_dates = dropped_rows["date"].value_counts().sort_index()
    print("\nStep 5: Added dayofweek and month.")
    print(f"  Rows dropped: {dropped_count:,} out of {rows_before_drop:,}")
    print(f"  Number of unique dates with dropped rows: {len(dropped_dates)}")
    print(f"  Early dropped dates (first week): {list(dropped_dates.head(6).index.strftime('%Y-%m-%d'))}")

    # Drop rows with missing values
    merged = merged.dropna().reset_index(drop=True)
    print(f"  Remaining rows: {len(merged):,}")

    # -------------------------------------------------------------
    # Step 6: Save final merged dataset
    # -------------------------------------------------------------
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "merged.csv"

    # Sort values cleanly by store and date
    merged = merged.sort_values(["store_id", "date"]).reset_index(drop=True)
    merged.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    print(f"\nStep 6: Saved merged dataset to {output_path}")

    return merged, dropped_count, dropped_dates


if __name__ == "__main__":
    df_merged, dropped_count, dropped_dates = merge_all_data()

    print("\n" + "=" * 55)
    print("FINAL MERGED DATA SUMMARY (data/processed/merged.csv)")
    print("=" * 55)

    print("\n1. Final columns:")
    print(list(df_merged.columns))

    print(f"\n2. Total row count: {len(df_merged):,}")

    print("\n3. Missing values per column:")
    print(df_merged.isnull().sum())

    print(f"\n4. Date range: {df_merged['date'].min().strftime('%Y-%m-%d')} to {df_merged['date'].max().strftime('%Y-%m-%d')}")

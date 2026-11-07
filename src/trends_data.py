"""
Retail Demand Forecasting - Social Media Trends Data Collection
Author: Person 3 (Social Media Trends Data)

Why Google Trends is used:
We use Google Trends as a stand-in for social media interest because the
X (Twitter) API requires an expensive paid subscription for historical data.
Google Trends is free, accessible, and provides a good measure of public
interest and brand awareness over time.

Limitations:
1. Weekly Data: For multi-year date ranges (such as 2013-01-01 to 2015-07-31),
   Google Trends natively aggregates search interest into weekly points,
   not daily points.
2. Search vs. Social: Google Trends reflects active search query frequency,
   which is not identical to social media posts, comments, or sentiment.
"""

import os
import time
import pandas as pd
from pytrends.request import TrendReq

# Configuration and parameters
KEYWORDS = ["Drogerie", "Rossmann", "Angebote"]
TIMEFRAME = "2013-01-01 2015-07-31"
GEO = "DE"
TZ = 60  # Germany time offset (UTC+1 in minutes)
DELAY_SECONDS = 10  # Pause between keyword requests to avoid rate limits
RETRY_PAUSE_SECONDS = 60  # Longer pause if rate limited (HTTP 429)
MAX_RETRIES = 3

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "trends_clean.csv")


def get_trend_client() -> TrendReq:
    """Initialize and return the TrendReq client with required language and timezone."""
    return TrendReq(hl="de-DE", tz=TZ, timeout=(10, 25))


def fetch_keyword_data(pytrends: TrendReq, keyword: str) -> pd.DataFrame:
    """
    Fetch weekly Google Trends data for a single keyword.
    Retries with longer delay if a 429 error occurs.
    Saves raw output to data/raw/ to avoid data loss.
    """
    print(f"Fetching Google Trends for keyword: '{keyword}'...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Build payload for one keyword at a time
            pytrends.build_payload([keyword], timeframe=TIMEFRAME, geo=GEO)
            df = pytrends.interest_over_time()

            # If empty dataframe returned, raise error
            if df.empty:
                raise ValueError(f"No data returned for keyword '{keyword}'.")

            # Google Trends returns 'isPartial' column; drop it
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            # Reset index so 'date' becomes a regular column
            df = df.reset_index()

            # Ensure 'date' is a proper datetime object
            df["date"] = pd.to_datetime(df["date"])

            # Make sure raw directory exists
            os.makedirs(RAW_DIR, exist_ok=True)

            # Save raw data for this keyword to data/raw/
            raw_path = os.path.join(RAW_DIR, f"trend_{keyword.lower()}_raw.csv")
            df.to_csv(raw_path, index=False)
            print(f"Saved raw data for '{keyword}' to {raw_path}")

            return df

        except Exception as error:
            error_msg = str(error)
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Error fetching '{keyword}': {error_msg}")

            # Check if error is related to rate limiting (429)
            if "429" in error_msg or "TooManyRequests" in error_msg:
                if attempt < MAX_RETRIES:
                    print(f"Rate limited (HTTP 429). Waiting {RETRY_PAUSE_SECONDS} seconds before retry...")
                    time.sleep(RETRY_PAUSE_SECONDS)
                else:
                    raise RuntimeError(
                        f"Failed to fetch '{keyword}' after {MAX_RETRIES} attempts due to rate limit (429). "
                        "Do not fake numbers."
                    ) from error
            else:
                # For non-429 errors or last attempt, raise the error directly
                if attempt < MAX_RETRIES:
                    print(f"Waiting {DELAY_SECONDS} seconds before retrying...")
                    time.sleep(DELAY_SECONDS)
                else:
                    raise error

    raise RuntimeError(f"Could not retrieve data for '{keyword}'.")


def clean_and_merge_trends(dataframes: dict) -> pd.DataFrame:
    """
    Combine individual keyword DataFrames into one weekly DataFrame.
    Checks for missing weeks and renames columns to the exact required schema:
    date, trend_drogerie, trend_rossmann, trend_angebote
    """
    merged_df = None

    for keyword, df in dataframes.items():
        # Keep only date and the keyword column
        subset = df[["date", keyword]].copy()
        col_name = f"trend_{keyword.lower()}"
        subset = subset.rename(columns={keyword: col_name})

        if merged_df is None:
            merged_df = subset
        else:
            merged_df = pd.merge(merged_df, subset, on="date", how="outer")

    # Sort chronologically
    merged_df = merged_df.sort_values("date").reset_index(drop=True)

    # Check for missing weeks
    merged_df["date"] = pd.to_datetime(merged_df["date"])
    full_range = pd.date_range(
        start=merged_df["date"].min(),
        end=merged_df["date"].max(),
        freq="W-SUN"
    )
    missing_dates = full_range.difference(merged_df["date"])
    if len(missing_dates) > 0:
        print(f"Warning: {len(missing_dates)} missing week(s) detected: {missing_dates}")
    else:
        print("Data integrity check: No missing weeks detected.")

    # Format date as YYYY-MM-DD string
    merged_df["date"] = merged_df["date"].dt.strftime("%Y-%m-%d")

    # Ensure final column ordering matches exact requirements
    target_columns = ["date", "trend_drogerie", "trend_rossmann", "trend_angebote"]
    merged_df = merged_df[target_columns]

    return merged_df


def main():
    """Main workflow to collect, clean, and save Google Trends data."""
    pytrends = get_trend_client()
    keyword_dfs = {}

    for i, keyword in enumerate(KEYWORDS):
        df_k = fetch_keyword_data(pytrends, keyword)
        keyword_dfs[keyword] = df_k

        # Wait 10 seconds between requests (except after the last one)
        if i < len(KEYWORDS) - 1:
            print(f"Waiting {DELAY_SECONDS} seconds before the next request...")
            time.sleep(DELAY_SECONDS)

    # Clean, check, and merge the datasets
    clean_trends_df = clean_and_merge_trends(keyword_dfs)

    # Ensure processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save to data/processed/trends_clean.csv
    clean_trends_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved clean trends data to {OUTPUT_FILE}")
    print(f"Shape: {clean_trends_df.shape}")
    print("First 5 rows:")
    print(clean_trends_df.head())


if __name__ == "__main__":
    main()

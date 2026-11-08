
import requests
import pandas as pd
from pathlib import Path

# Read city information
cities = pd.read_csv("data/city_map.csv")

frames = []

# Fetch weather data for every city
for _, row in cities.iterrows():

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "start_date": "2013-01-01",
        "end_date": "2015-07-31",
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "Europe/Berlin"
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()["daily"]

    df = pd.DataFrame(data)

    # Add city name
    df["city"] = row["city"]

    frames.append(df)

# Combine all cities
weather = pd.concat(frames, ignore_index=True)

# Rename columns
weather = weather.rename(columns={
    "time": "date",
    "temperature_2m_mean": "temperature",
    "precipitation_sum": "rain_mm"
})

# Convert date into proper date format
weather["date"] = pd.to_datetime(weather["date"])

# Sort data city-wise and date-wise
weather = weather.sort_values(["city", "date"])

# Fill missing values using previous day's value
weather[["temperature", "rain_mm"]] = (
    weather.groupby("city")[["temperature", "rain_mm"]]
    .ffill()
)

# Create rainy day feature
weather["is_rainy"] = (weather["rain_mm"] > 1).astype(int)

# Keep exactly the required columns
weather = weather[
    ["date", "city", "temperature", "rain_mm", "is_rainy"]
]

# Create output folder if it doesn't exist
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Save final CSV
weather.to_csv(
    "data/processed/weather_clean.csv",
    index=False,
    date_format="%Y-%m-%d"
)

print("Weather data successfully fetched and cleaned!")
print(weather.head())
print("Total rows:", len(weather))

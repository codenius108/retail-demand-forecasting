"""
Retail Demand Forecasting - Feature Ablation Study
Author: Person 4 (Ablation Test)

Evaluates the relative contribution of different feature groups across
3 consecutive chronological 60-day test windows using Random Forest Regressor:
1. base: sales_lag7, dayofweek, promo
2. base + calendar: + month, SchoolHoliday, one-hot cities
3. base + calendar + weather: + temperature, rain_mm, is_rainy
4. base + calendar + trends: + trend_drogerie, trend_rossmann, trend_angebote
5. base + calendar + weather + trends: all 16 features

Saves metrics to results/ablation_metrics.csv and chart to results/figures/.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Avoid matplotlib cache permissions issue
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "/tmp")


def run_ablation_study():
    # -------------------------------------------------------------
    # Step 1: Load merged dataset
    # -------------------------------------------------------------
    merged_path = Path("data/processed/merged.csv")
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged dataset not found at: {merged_path}")

    # Read data without mixed type warnings
    df = pd.read_csv(
        merged_path,
        parse_dates=["date"],
        dtype={"StateHoliday": str},
        low_memory=False
    )

    # One-hot encode the city column (StateHoliday is strictly excluded)
    df_encoded = pd.get_dummies(df, columns=["city"], dtype=int)
    city_cols = sorted([c for c in df_encoded.columns if c.startswith("city_")])

    # -------------------------------------------------------------
    # Define Feature Groups
    # -------------------------------------------------------------
    base_features = ["sales_lag7", "dayofweek", "promo"]
    calendar_features = ["month", "SchoolHoliday"] + city_cols
    weather_features = ["temperature", "rain_mm", "is_rainy"]
    trends_features = ["trend_drogerie", "trend_rossmann", "trend_angebote"]

    models_config = {
        "base": base_features,
        "base + calendar": base_features + calendar_features,
        "base + calendar + weather": base_features + calendar_features + weather_features,
        "base + calendar + trends": base_features + calendar_features + trends_features,
        "base + calendar + weather + trends": base_features + calendar_features + weather_features + trends_features,
    }

    # -------------------------------------------------------------
    # Define 3 Chronological 60-Day Windows
    # -------------------------------------------------------------
    max_date = df["date"].max()
    w1_start = max_date - pd.Timedelta(days=60)
    w2_start = max_date - pd.Timedelta(days=120)
    w3_start = max_date - pd.Timedelta(days=180)

    # Train strictly on data before each window, test on the window
    window_data = {
        "1": {
            "name": "Window 1 (Last 60 days)",
            "train": df_encoded[df_encoded["date"] < w1_start].copy(),
            "test": df_encoded[df_encoded["date"] >= w1_start].copy()
        },
        "2": {
            "name": "Window 2 (60 days prior)",
            "train": df_encoded[df_encoded["date"] < w2_start].copy(),
            "test": df_encoded[(df_encoded["date"] >= w2_start) & (df_encoded["date"] < w1_start)].copy()
        },
        "3": {
            "name": "Window 3 (120 days prior)",
            "train": df_encoded[df_encoded["date"] < w3_start].copy(),
            "test": df_encoded[(df_encoded["date"] >= w3_start) & (df_encoded["date"] < w2_start)].copy()
        }
    }

    print("==================================================")
    print("CHRONOLOGICAL TEST WINDOWS SUMMARY")
    print("==================================================")
    for w_id, w_info in window_data.items():
        tr = w_info["train"]
        te = w_info["test"]
        print(f"Window {w_id} ({w_info['name']}):")
        print(f"  Train: {len(tr):,} rows | {tr['date'].min().strftime('%Y-%m-%d')} to {tr['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Test:  {len(te):,} rows | {te['date'].min().strftime('%Y-%m-%d')} to {te['date'].max().strftime('%Y-%m-%d')}")

    # -------------------------------------------------------------
    # Train Models & Record MAE
    # -------------------------------------------------------------
    results = []
    avg_maes_dict = {}

    print("\n==================================================")
    print("TRAINING MODELS & EVALUATING MAE")
    print("==================================================")

    for model_name, features in models_config.items():
        print(f"\nModel: {model_name} ({len(features)} features)")
        window_maes = []

        for w_id in ["1", "2", "3"]:
            train_df = window_data[w_id]["train"]
            test_df = window_data[w_id]["test"]

            # Train Random Forest Regressor
            rf = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(train_df[features], train_df["sales"])
            preds = rf.predict(test_df[features])

            mae = mean_absolute_error(test_df["sales"], preds)
            window_maes.append(mae)

            results.append({
                "model": model_name,
                "window": w_id,
                "MAE": round(mae, 2)
            })
            print(f"  Window {w_id} MAE: {mae:,.2f}")

        # Compute average MAE across the 3 windows
        mean_mae = float(np.mean(window_maes))
        avg_maes_dict[model_name] = mean_mae

        results.append({
            "model": model_name,
            "window": "average",
            "MAE": round(mean_mae, 2)
        })
        print(f"  --> Average MAE: {mean_mae:,.2f}")

    # -------------------------------------------------------------
    # Save Metrics to results/ablation_metrics.csv
    # -------------------------------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "ablation_metrics.csv"

    ablation_df = pd.DataFrame(results)
    ablation_df.to_csv(metrics_path, index=False)
    print(f"\nSaved ablation metrics to {metrics_path}")

    # -------------------------------------------------------------
    # Generate & Save Average MAE Bar Chart (No plt.show)
    # -------------------------------------------------------------
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    chart_path = fig_dir / "ablation_model_mae.png"

    plt.figure(figsize=(10, 5.5))
    model_labels = list(avg_maes_dict.keys())
    model_values = list(avg_maes_dict.values())
    colors = ["#7293CB", "#E1974C", "#2CA02C", "#D62728", "#9467BD"]

    bars = plt.bar(model_labels, model_values, color=colors, edgecolor="black", alpha=0.9, width=0.55)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{h:,.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.title("Ablation Study: Average Test MAE Across 3 Windows (Lower is Better)", fontsize=13, pad=12)
    plt.xlabel("Feature Configuration", fontsize=11)
    plt.ylabel("Average MAE (Sales Units)", fontsize=11)
    plt.ylim(0, 1400)
    plt.xticks(rotation=15, ha="right", fontsize=9.5)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved ablation chart to {chart_path}")

    return ablation_df, avg_maes_dict


if __name__ == "__main__":
    run_ablation_study()

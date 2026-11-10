"""
Retail Demand Forecasting - Model Training and Evaluation Pipeline
Author: Person 4 (Modeling)

This script trains and compares 4 models:
1. Linear Regression (Sales only: lag7, dayofweek, promo)
2. Linear Regression (All features: sales + weather + trends + city + calendar)
3. Random Forest (Sales only)
4. Random Forest (All features)

Saves evaluation metrics to results/metrics.csv and charts to results/figures/.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Avoid matplotlib cache directory issues
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "/tmp")


def train_and_evaluate_models():
    # -------------------------------------------------------------
    # Step 1: Read merged data and chronological split
    # -------------------------------------------------------------
    merged_path = Path("data/processed/merged.csv")
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged dataset not found at: {merged_path}")

    # Load cleaned merged dataset
    df = pd.read_csv(
        merged_path,
        parse_dates=["date"],
        dtype={"StateHoliday": str},
        low_memory=False
    )

    # Chronological train / test split (last 60 calendar days for test)
    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(days=60)

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    print("Step 1: Loaded and split data by date.")
    print(f"  Train set: {len(train_df):,} rows ({train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"  Test set:  {len(test_df):,} rows ({test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')})")

    # -------------------------------------------------------------
    # Feature Engineering & One-Hot Encoding
    # -------------------------------------------------------------
    # Model 1 & 3 features: sales-only
    features_sales_only = ["sales_lag7", "dayofweek", "promo"]

    # One-hot encode the 'city' column across the entire dataset
    df_encoded = pd.get_dummies(df, columns=["city"], dtype=int)
    train_encoded = df_encoded[df_encoded["date"] < split_date].copy()
    test_encoded = df_encoded[df_encoded["date"] >= split_date].copy()

    # City one-hot column names
    city_cols = sorted([c for c in df_encoded.columns if c.startswith("city_")])

    # Model 2 & 4 features: sales + weather + trends + calendar + city
    features_all = (
        features_sales_only
        + [
            "temperature",
            "rain_mm",
            "is_rainy",
            "trend_drogerie",
            "trend_rossmann",
            "trend_angebote",
            "month",
            "SchoolHoliday",
        ]
        + city_cols
    )

    y_train = train_df["sales"]
    y_test = test_df["sales"]

    # Dictionary to hold test predictions and metrics
    metrics_list = []
    predictions = {}

    # -------------------------------------------------------------
    # Step 2: Model 1 - Linear Regression (Sales Only)
    # -------------------------------------------------------------
    print("\nStep 2: Training Model 1 - linear_sales_only...")
    m1 = LinearRegression()
    m1.fit(train_df[features_sales_only], y_train)
    p1 = m1.predict(test_df[features_sales_only])
    mae1 = mean_absolute_error(y_test, p1)
    rmse1 = float(np.sqrt(mean_squared_error(y_test, p1)))
    predictions["linear_sales_only"] = p1
    metrics_list.append({
        "model": "linear_sales_only",
        "MAE": round(mae1, 2),
        "RMSE": round(rmse1, 2)
    })
    print(f"  linear_sales_only -> MAE: {mae1:,.2f}, RMSE: {rmse1:,.2f}")

    # -------------------------------------------------------------
    # Step 3: Model 2 - Linear Regression (All Features)
    # -------------------------------------------------------------
    print("\nStep 3: Training Model 2 - linear_all...")
    m2 = LinearRegression()
    m2.fit(train_encoded[features_all], y_train)
    p2 = m2.predict(test_encoded[features_all])
    mae2 = mean_absolute_error(y_test, p2)
    rmse2 = float(np.sqrt(mean_squared_error(y_test, p2)))
    predictions["linear_all"] = p2
    metrics_list.append({
        "model": "linear_all",
        "MAE": round(mae2, 2),
        "RMSE": round(rmse2, 2)
    })
    print(f"  linear_all        -> MAE: {mae2:,.2f}, RMSE: {rmse2:,.2f}")

    # -------------------------------------------------------------
    # Step 4: Model 3 - Random Forest (Sales Only)
    # -------------------------------------------------------------
    print("\nStep 4: Training Model 3 - rf_sales_only...")
    m3 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    m3.fit(train_df[features_sales_only], y_train)
    p3 = m3.predict(test_df[features_sales_only])
    mae3 = mean_absolute_error(y_test, p3)
    rmse3 = float(np.sqrt(mean_squared_error(y_test, p3)))
    predictions["rf_sales_only"] = p3
    metrics_list.append({
        "model": "rf_sales_only",
        "MAE": round(mae3, 2),
        "RMSE": round(rmse3, 2)
    })
    print(f"  rf_sales_only     -> MAE: {mae3:,.2f}, RMSE: {rmse3:,.2f}")

    # -------------------------------------------------------------
    # Step 5: Model 4 - Random Forest (All Features)
    # -------------------------------------------------------------
    print("\nStep 5: Training Model 4 - rf_all...")
    m4 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    m4.fit(train_encoded[features_all], y_train)
    p4 = m4.predict(test_encoded[features_all])
    mae4 = mean_absolute_error(y_test, p4)
    rmse4 = float(np.sqrt(mean_squared_error(y_test, p4)))
    predictions["rf_all"] = p4
    metrics_list.append({
        "model": "rf_all",
        "MAE": round(mae4, 2),
        "RMSE": round(rmse4, 2)
    })
    print(f"  rf_all            -> MAE: {mae4:,.2f}, RMSE: {rmse4:,.2f}")

    # -------------------------------------------------------------
    # Step 6: Save metrics to results/metrics.csv
    # -------------------------------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = results_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nStep 6: Saved metrics to {metrics_path}")

    # -------------------------------------------------------------
    # Step 7: Generate and save 3 figures (never call plt.show)
    # -------------------------------------------------------------
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Chart 7a: MAE of all 4 models as a bar chart
    plt.figure(figsize=(9, 5))
    bar_colors = ["#7293CB", "#2B5C8F", "#E1974C", "#D35E17"]
    bars = plt.bar(metrics_df["model"], metrics_df["MAE"], color=bar_colors, edgecolor="black", alpha=0.9, width=0.55)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{h:,.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.title("Test MAE Across All 4 Models (Lower is Better)", fontsize=14, pad=12)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Mean Absolute Error (Sales Units)", fontsize=12)
    plt.ylim(0, 1300)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_a_path = fig_dir / "model_mae_comparison.png"
    plt.savefig(chart_a_path, dpi=150)
    plt.close()
    print(f"Step 7a: Saved {chart_a_path}")

    # Chart 7b: Predicted vs actual daily total sales over test period for best model (rf_all)
    test_encoded["pred_best"] = p4
    daily_actual = test_encoded.groupby("date")["sales"].sum()
    daily_pred = test_encoded.groupby("date")["pred_best"].sum()

    plt.figure(figsize=(12, 5))
    plt.plot(daily_actual.index, daily_actual.values, label="Actual Daily Sales", color="#1f77b4", linewidth=2)
    plt.plot(daily_pred.index, daily_pred.values, label="Predicted Daily Sales (rf_all)", color="#d62728", linestyle="--", linewidth=2)
    plt.title("Actual vs Predicted Total Daily Sales (rf_all, Test Period)", fontsize=14, pad=12)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Daily Sales Across 50 Stores", fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_b_path = fig_dir / "best_model_pred_vs_actual.png"
    plt.savefig(chart_b_path, dpi=150)
    plt.close()
    print(f"Step 7b: Saved {chart_b_path}")

    # Chart 7c: Feature importance of rf_all model (top 10 features)
    importances = pd.Series(m4.feature_importances_, index=features_all).sort_values(ascending=True)
    top10 = importances.tail(10)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top10.index, top10.values, color="#2ca02c", edgecolor="black", alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w * 100:.1f}%", ha="left", va="center", fontsize=9, fontweight="bold")
    plt.title("Top 10 Feature Importances (Random Forest - rf_all)", fontsize=14, pad=12)
    plt.xlabel("Relative Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.xlim(0, max(top10.values) + 0.08)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_c_path = fig_dir / "rf_all_feature_importance.png"
    plt.savefig(chart_c_path, dpi=150)
    plt.close()
    print(f"Step 7c: Saved {chart_c_path}")

    # -------------------------------------------------------------
    # Step 8: Calculate percent improvements
    # -------------------------------------------------------------
    linear_pct = (mae1 - mae2) / mae1 * 100
    rf_pct = (mae3 - mae4) / mae3 * 100
    overall_pct = (mae1 - mae4) / mae1 * 100

    print("\n" + "=" * 55)
    print("STEP 8 SUMMARY: IMPACT OF WEATHER & TRENDS")
    print("=" * 55)
    print(f"1. Linear Regression MAE improvement:  {linear_pct:.2f}% (from {mae1:.2f} down to {mae2:.2f})")
    print(f"2. Random Forest MAE improvement:      {rf_pct:.2f}% (from {mae3:.2f} down to {mae4:.2f})")
    print(f"3. Overall improvement over baseline:  {overall_pct:.2f}% (from {mae1:.2f} down to {mae4:.2f})")

    return metrics_df, linear_pct, rf_pct, overall_pct


if __name__ == "__main__":
    train_and_evaluate_models()

"""
Retail Demand Forecasting - Inventory Optimization & Safety Stock Decision
Author: Person 4 (Inventory Decision)

Computes required safety stock levels under a 95% cycle service level
for different supply chain replenishment lead times (L = 1, 2, 3, 7 days):
- Comparison Model: base features only (sales_lag7, dayofweek, promo)
- Final Model: base + calendar + weather features (month, SchoolHoliday, one-hot cities, temperature, rain_mm, is_rainy)

Formula:
  Safety Stock = z * sigma * sqrt(L)
  where:
    z = 1.645 (95% cycle service level assumption)
    L = replenishment lead time in days (1, 2, 3, 7 days)
    sigma = standard deviation of forecast errors from the validation period
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Avoid matplotlib cache permissions issue
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "/tmp")

# =============================================================================
# INVENTORY DECISION ASSUMPTIONS:
# 1. Service Level: 95% cycle service level (z = 1.645 for one-tailed normal error).
# 2. Replenishment Lead Times: L = 1 day, 2 days, 3 days, and 7 days.
# 3. Forecast Error: Actual sales minus predicted sales on unseen validation data.
# 4. Standard Deviation (sigma): Captures demand uncertainty during lead time.
# =============================================================================
SERVICE_LEVEL_Z = 1.645
LEAD_TIMES = [1, 2, 3, 7]


def run_inventory_analysis():
    # -------------------------------------------------------------
    # Load Merged Dataset & Setup Features
    # -------------------------------------------------------------
    merged_path = Path("data/processed/merged.csv")
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged dataset not found at: {merged_path}")

    df = pd.read_csv(
        merged_path,
        parse_dates=["date"],
        dtype={"StateHoliday": str},
        low_memory=False
    )

    # One-hot encode cities (StateHoliday strictly omitted as instructed)
    df_encoded = pd.get_dummies(df, columns=["city"], dtype=int)
    city_cols = sorted([c for c in df_encoded.columns if c.startswith("city_")])

    base_features = ["sales_lag7", "dayofweek", "promo"]
    calendar_features = ["month", "SchoolHoliday"] + city_cols
    weather_features = ["temperature", "rain_mm", "is_rainy"]

    final_features = base_features + calendar_features + weather_features

    print("==================================================")
    print("INVENTORY DECISION PARAMETERS & ASSUMPTIONS")
    print("==================================================")
    print(f"Service Level: 95% (z = {SERVICE_LEVEL_Z})")
    print(f"Lead Times evaluated: {LEAD_TIMES} days")
    print(f"Base model features ({len(base_features)}): {base_features}")
    print(f"Final model features ({len(final_features)}): {final_features}")

    # -------------------------------------------------------------
    # Step 1: Validation Period (Error Standard Deviation Sigma)
    # Train strictly before 2015-04-02; Validate on 2015-04-02 to 2015-05-30
    # -------------------------------------------------------------
    train_val = df_encoded[df_encoded["date"] < "2015-04-02"].copy()
    val_df = df_encoded[
        (df_encoded["date"] >= "2015-04-02") & (df_encoded["date"] <= "2015-05-30")
    ].copy()

    print("\nStep 1: Validation Split & Error Metrics")
    print(f"  Train: {len(train_val):,} rows ({train_val['date'].min().strftime('%Y-%m-%d')} to {train_val['date'].max().strftime('%Y-%m-%d')})")
    print(f"  Val:   {len(val_df):,} rows ({val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')})")

    # Train comparison model on validation training data
    rf_base_val = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_base_val.fit(train_val[base_features], train_val["sales"])
    pred_base_val = rf_base_val.predict(val_df[base_features])

    # Train final model on validation training data
    rf_final_val = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_final_val.fit(train_val[final_features], train_val["sales"])
    pred_final_val = rf_final_val.predict(val_df[final_features])

    # Calculate forecast error: actual sales - predicted sales
    error_base_val = val_df["sales"] - pred_base_val
    error_final_val = val_df["sales"] - pred_final_val

    bias_base = float(error_base_val.mean())
    sigma_base = float(error_base_val.std())

    bias_final = float(error_final_val.mean())
    sigma_final = float(error_final_val.std())

    print(f"  Comparison Model (base):")
    print(f"    Average Error (bias):  {bias_base:+.2f} units")
    print(f"    Error Std Dev (sigma): {sigma_base:,.2f} units")
    print(f"  Final Model (base + calendar + weather):")
    print(f"    Average Error (bias):  {bias_final:+.2f} units")
    print(f"    Error Std Dev (sigma): {sigma_final:,.2f} units")

    # -------------------------------------------------------------
    # Step 2: Test Period Training & Predictions
    # Train strictly before 2015-06-02; Test on 2015-06-02 to 2015-07-31
    # -------------------------------------------------------------
    train_test = df_encoded[df_encoded["date"] < "2015-06-02"].copy()
    test_df = df_encoded[
        (df_encoded["date"] >= "2015-06-02") & (df_encoded["date"] <= "2015-07-31")
    ].copy()

    print("\nStep 2: Test Split (Last 60 Days)")
    print(f"  Train: {len(train_test):,} rows ({train_test['date'].min().strftime('%Y-%m-%d')} to {train_test['date'].max().strftime('%Y-%m-%d')})")
    print(f"  Test:  {len(test_df):,} rows ({test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')})")

    rf_base_test = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_base_test.fit(train_test[base_features], train_test["sales"])
    pred_base_test = rf_base_test.predict(test_df[base_features])

    rf_final_test = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_final_test.fit(train_test[final_features], train_test["sales"])
    pred_final_test = rf_final_test.predict(test_df[final_features])

    # -------------------------------------------------------------
    # Step 3: Safety Stock Table per Store per Day
    # SS = z * sigma * sqrt(L)
    # -------------------------------------------------------------
    ss_records = []
    for L in LEAD_TIMES:
        ss_base = SERVICE_LEVEL_Z * sigma_base * np.sqrt(L)
        ss_final = SERVICE_LEVEL_Z * sigma_final * np.sqrt(L)
        reduction_units = ss_base - ss_final
        reduction_pct = (reduction_units / ss_base) * 100

        ss_records.append({
            "lead_time_days": L,
            "safety_stock_base": round(ss_base, 2),
            "safety_stock_final": round(ss_final, 2),
            "reduction_units": round(reduction_units, 2),
            "reduction_pct": round(reduction_pct, 2)
        })

    ss_df = pd.DataFrame(ss_records)
    print("\nStep 3: Safety Stock Comparison Table:")
    print(ss_df.to_string(index=False))

    # -------------------------------------------------------------
    # Step 4: Coverage Check on Test Period (L = 1)
    # Stock target = forecast + z * sigma
    # -------------------------------------------------------------
    target_base = pred_base_test + (SERVICE_LEVEL_Z * sigma_base)
    cov_base = float((test_df["sales"] <= target_base).mean() * 100)

    target_final = pred_final_test + (SERVICE_LEVEL_Z * sigma_final)
    cov_final = float((test_df["sales"] <= target_final).mean() * 100)

    print("\nStep 4: Empirical Coverage Check on Test Set (L = 1)")
    print(f"  Comparison Model Coverage: {cov_base:.2f}% (difference from 95%: {cov_base - 95.0:+.2f}%)")
    print(f"  Final Model Coverage:      {cov_final:.2f}% (difference from 95%: {cov_final - 95.0:+.2f}%)")

    # -------------------------------------------------------------
    # Step 5: Safety Stock Reduction Percentage
    # -------------------------------------------------------------
    pct_reduction = float((sigma_base - sigma_final) / sigma_base * 100)
    print(f"\nStep 5: Safety Stock Reduction: {pct_reduction:.2f}%")

    # -------------------------------------------------------------
    # Step 6: Save CSV & Generate Bar Chart (no plt.show)
    # -------------------------------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "inventory_summary.csv"
    ss_df.to_csv(summary_path, index=False)
    print(f"\nStep 6: Saved inventory summary table to {summary_path}")

    # Generate grouped bar chart
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    chart_path = fig_dir / "inventory_safety_stock.png"

    x = np.arange(len(LEAD_TIMES))
    bar_width = 0.35

    plt.figure(figsize=(9, 5.5))
    bars1 = plt.bar(
        x - bar_width / 2,
        ss_df["safety_stock_base"],
        width=bar_width,
        label="Comparison Model (Sales Only)",
        color="#7293CB",
        edgecolor="black",
        alpha=0.9
    )
    bars2 = plt.bar(
        x + bar_width / 2,
        ss_df["safety_stock_final"],
        width=bar_width,
        label="Final Model (Sales + Calendar + Weather)",
        color="#2CA02C",
        edgecolor="black",
        alpha=0.9
    )

    # Add numeric callouts on bars
    for bar in bars1:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 80, f"{h:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 80, f"{h:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.title("Safety Stock Requirement per Store-Day by Lead Time (95% Service Level)", fontsize=13, pad=12)
    plt.xlabel("Replenishment Lead Time L (Days)", fontsize=11)
    plt.ylabel("Required Safety Stock (Units / Store / Day)", fontsize=11)
    plt.xticks(x, [f"L = {L} day{'s' if L > 1 else ''}" for L in LEAD_TIMES], fontsize=10)
    plt.ylim(0, 9500)
    plt.legend(frameon=True, fontsize=10.5)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved inventory safety stock chart to {chart_path}")

    return ss_df, sigma_base, sigma_final, cov_base, cov_final, pct_reduction


if __name__ == "__main__":
    run_inventory_analysis()

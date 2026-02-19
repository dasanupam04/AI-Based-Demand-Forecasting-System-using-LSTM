import pandas as pd
import numpy as np


# 1️⃣ Category Demand Contribution
def category_demand_share(df):
    """
    Compute percentage contribution of each category to total sales.
    """

    category_sales = df.groupby("Category")["Units Sold"].sum().reset_index()
    total_sales = category_sales["Units Sold"].sum()

    category_sales["Demand Share %"] = (
        category_sales["Units Sold"] / total_sales
    ) * 100

    return category_sales


# 2️⃣ Category Growth Rate (Monthly)
def category_growth_rate(df):
    """
    Compute monthly growth rate per category.
    """

    df["Date"] = pd.to_datetime(df["Date"])
    df["YearMonth"] = df["Date"].dt.to_period("M")

    monthly_sales = df.groupby(["Category", "YearMonth"])["Units Sold"].sum().reset_index()

    monthly_sales["Growth %"] = (
        monthly_sales.groupby("Category")["Units Sold"]
        .pct_change() * 100
    )

    return monthly_sales


# 3️⃣ Category Profitability
def category_profitability(df):
    """
    Estimate profit per category.
    """

    df["Effective Price"] = df["Price"] * (1 - df["Discount"] / 100)
    df["Revenue"] = df["Units Sold"] * df["Effective Price"]
    df["Estimated Cost"] = df["Units Sold"] * (df["Price"] * 0.6)
    df["Estimated Profit"] = df["Revenue"] - df["Estimated Cost"]

    category_profit = df.groupby("Category").agg({
        "Revenue": "sum",
        "Estimated Profit": "sum",
        "Units Sold": "sum"
    }).reset_index()

    category_profit["Profit Margin %"] = (
        category_profit["Estimated Profit"] /
        category_profit["Revenue"] * 100
    )

    return category_profit


# 4️⃣ Category Seasonal Index
def category_seasonal_index(df):
    """
    Compute seasonal index per category (monthly).
    """

    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month

    overall_avg = df.groupby("Category")["Units Sold"].mean().reset_index()
    overall_avg.columns = ["Category", "Overall Avg"]

    monthly_avg = df.groupby(["Category", "Month"])["Units Sold"].mean().reset_index()

    seasonal_index = monthly_avg.merge(overall_avg, on="Category")

    seasonal_index["Seasonal Index"] = (
        seasonal_index["Units Sold"] / seasonal_index["Overall Avg"]
    )

    return seasonal_index

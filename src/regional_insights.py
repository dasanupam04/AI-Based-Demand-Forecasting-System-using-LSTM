import pandas as pd
import numpy as np


def region_store_summary(df):
    """
    Aggregate performance metrics by Region and Store.
    """

    summary = df.groupby(["Region", "Store ID"]).agg({
        "Units Sold": "sum",
        "Inventory Level": "mean",
        "Price": "mean",
        "Discount": "mean"
    }).reset_index()

    return summary


def region_demand_heatmap(df):
    """
    Create pivot table for heatmap visualization.
    """

    pivot = df.pivot_table(
        values="Units Sold",
        index="Region",
        columns="Category",
        aggfunc="sum"
    )

    return pivot


def suggest_inventory_redistribution(df):
    """
    Suggest stock transfer from overstocked stores to understocked stores.
    """

    redistribution_plan = []

    region_groups = df.groupby(["Region", "Store ID"]).agg({
        "Inventory Level": "mean",
        "Units Sold": "mean"
    }).reset_index()

    region_groups["Stock Ratio"] = (
        region_groups["Inventory Level"] /
        (region_groups["Units Sold"] + 1)
    )

    for region in region_groups["Region"].unique():

        region_data = region_groups[region_groups["Region"] == region]

        high_stock = region_data.sort_values("Stock Ratio", ascending=False).iloc[0]
        low_stock = region_data.sort_values("Stock Ratio", ascending=True).iloc[0]

        if high_stock["Stock Ratio"] > 2 and low_stock["Stock Ratio"] < 1:
            redistribution_plan.append({
                "Region": region,
                "From Store": high_stock["Store ID"],
                "To Store": low_stock["Store ID"],
                "Suggested Transfer Units": int(
                    (high_stock["Inventory Level"] - low_stock["Inventory Level"]) * 0.2
                )
            })

    return redistribution_plan



def region_growth_analysis(df):
    """
    Compute month-over-month growth rate for each region.
    """

    df["Date"] = pd.to_datetime(df["Date"])
    df["YearMonth"] = df["Date"].dt.to_period("M")

    monthly_sales = df.groupby(["Region", "YearMonth"])["Units Sold"].sum().reset_index()

    monthly_sales["Growth Rate %"] = (
        monthly_sales.groupby("Region")["Units Sold"]
        .pct_change() * 100
    )

    return monthly_sales



def region_profitability_analysis(df):
    """
    Estimate profitability by region.
    """

    # Effective selling price after discount
    df["Effective Price"] = df["Price"] * (1 - df["Discount"] / 100)

    # Revenue
    df["Revenue"] = df["Units Sold"] * df["Effective Price"]

    # Approx cost assumption (60% of original price)
    df["Estimated Cost"] = df["Units Sold"] * (df["Price"] * 0.6)

    # Profit
    df["Estimated Profit"] = df["Revenue"] - df["Estimated Cost"]

    region_profit = df.groupby("Region").agg({
        "Revenue": "sum",
        "Estimated Profit": "sum",
        "Units Sold": "sum"
    }).reset_index()

    region_profit["Profit Margin %"] = (
        region_profit["Estimated Profit"] /
        region_profit["Revenue"] * 100
    )

    return region_profit





def region_demand_volatility(df):
    """
    Measure demand volatility per region.
    Volatility = Standard deviation of Units Sold.
    """

    volatility = df.groupby("Region")["Units Sold"].std().reset_index()
    volatility.columns = ["Region", "Demand Volatility"]

    return volatility






def region_stock_efficiency(df):
    """
    Measure stock utilization efficiency per region.
    Efficiency = Total Units Sold / Total Inventory Level
    """

    region_eff = df.groupby("Region").agg({
        "Units Sold": "sum",
        "Inventory Level": "sum"
    }).reset_index()

    region_eff["Stock Efficiency"] = (
        region_eff["Units Sold"] /
        region_eff["Inventory Level"]
    )

    return region_eff




def region_category_matrix(df):
    """
    Create Region × Category demand matrix.
    Shows total demand per category in each region.
    """

    matrix = df.pivot_table(
        values="Units Sold",
        index="Region",
        columns="Category",
        aggfunc="sum"
    )

    return matrix



def region_seasonal_index(df):
    """
    Compute monthly seasonal index per region.
    Seasonal Index = Monthly Avg / Overall Avg
    """

    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month

    overall_avg = df.groupby("Region")["Units Sold"].mean().reset_index()
    overall_avg.columns = ["Region", "Overall Avg"]

    monthly_avg = df.groupby(["Region", "Month"])["Units Sold"].mean().reset_index()

    seasonal_index = monthly_avg.merge(overall_avg, on="Region")

    seasonal_index["Seasonal Index"] = (
        seasonal_index["Units Sold"] /
        seasonal_index["Overall Avg"]
    )

    return seasonal_index




def region_growth_momentum(df):
    """
    Compute average monthly growth per region.
    Indicates overall growth direction.
    """

    df["Date"] = pd.to_datetime(df["Date"])
    df["YearMonth"] = df["Date"].dt.to_period("M")

    monthly_sales = df.groupby(["Region", "YearMonth"])["Units Sold"].sum().reset_index()

    monthly_sales["Growth %"] = (
        monthly_sales.groupby("Region")["Units Sold"]
        .pct_change() * 100
    )

    momentum = monthly_sales.groupby("Region")["Growth %"].mean().reset_index()
    momentum.columns = ["Region", "Average Growth %"]

    return momentum




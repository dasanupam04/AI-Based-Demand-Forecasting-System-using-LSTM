import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def time_series_decomposition(df, store_id=None, product_id=None):

    df["Date"] = pd.to_datetime(df["Date"])

    if store_id and product_id:
        df = df[
            (df["Store ID"] == store_id) &
            (df["Product ID"] == product_id)
        ]

    df = df.sort_values("Date")
    ts = df.set_index("Date")["Units Sold"]

    decomposition = seasonal_decompose(
        ts,
        model="additive",
        period=30
    )

    return decomposition


def monthly_seasonal_pattern(df):

    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month

    monthly_pattern = df.groupby("Month")["Units Sold"].mean().reset_index()

    return monthly_pattern


def quarterly_trend_analysis(df):

    df["Date"] = pd.to_datetime(df["Date"])
    df["Quarter"] = df["Date"].dt.to_period("Q")

    quarterly_sales = df.groupby("Quarter")["Units Sold"].sum().reset_index()
    quarterly_sales["Growth %"] = quarterly_sales["Units Sold"].pct_change() * 100

    return quarterly_sales





def seasonality_strength(decomposition):
    """
    Measure strength of seasonality from decomposition object.
    """

    seasonal = decomposition.seasonal
    strength = seasonal.std() / seasonal.mean()

    return strength



def category_decomposition(df, category):
    """
    Perform seasonal decomposition for a specific category.
    """

    df["Date"] = pd.to_datetime(df["Date"])
    cat_df = df[df["Category"] == category].sort_values("Date")

    ts = cat_df.set_index("Date")["Units Sold"]

    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposition = seasonal_decompose(
        ts,
        model="additive",
        period=30
    )

    return decomposition




def region_decomposition(df, region):
    """
    Perform seasonal decomposition for a specific region.
    """

    df["Date"] = pd.to_datetime(df["Date"])
    reg_df = df[df["Region"] == region].sort_values("Date")

    ts = reg_df.set_index("Date")["Units Sold"]

    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposition = seasonal_decompose(
        ts,
        model="additive",
        period=30
    )

    return decomposition





def long_cycle_trend(df, window=90):
    """
    Detect long-term demand cycles using rolling mean.
    """

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    ts = df.set_index("Date")["Units Sold"]

    long_trend = ts.rolling(window=window).mean()

    return long_trend






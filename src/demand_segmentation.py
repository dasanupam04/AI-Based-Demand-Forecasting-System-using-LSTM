import pandas as pd


def product_volatility_classification(df):
    """
    Classify products based on demand volatility.
    """

    volatility = df.groupby("Product ID")["Units Sold"].std().reset_index()
    volatility.columns = ["Product ID", "Demand Std"]

    mean_std = volatility["Demand Std"].mean()

    volatility["Demand Segment"] = pd.cut(
        volatility["Demand Std"],
        bins=[-float("inf"), mean_std*0.75, mean_std*1.25, float("inf")],
        labels=["Stable", "Moderate", "Highly Volatile"]
    )

    return volatility

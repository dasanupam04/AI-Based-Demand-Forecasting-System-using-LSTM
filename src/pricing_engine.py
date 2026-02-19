import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def estimate_price_elasticity(df, store_id=None, product_id=None):
    """
    Estimate price elasticity using log-log regression.
    """

    if store_id and product_id:
        df = df[
            (df["Store ID"] == store_id) &
            (df["Product ID"] == product_id)
        ]

    df = df[df["Price"] > 0]
    df = df[df["Units Sold"] > 0]

    df["Log Price"] = np.log(df["Price"])
    df["Log Demand"] = np.log(df["Units Sold"])

    X = df[["Log Price"]]
    y = df["Log Demand"]

    model = LinearRegression()
    model.fit(X, y)

    elasticity = model.coef_[0]

    return elasticity




def suggest_optimal_price(current_price, elasticity, competitor_price=None):
    """
    Suggest price adjustment based on elasticity.
    """

    recommendation = {}

    if elasticity < -1:
        # Demand is elastic → Lower price slightly
        new_price = current_price * 0.95
        recommendation["Strategy"] = "Lower Price"
    elif elasticity > -1 and elasticity < 0:
        # Inelastic → Can increase price
        new_price = current_price * 1.05
        recommendation["Strategy"] = "Increase Price"
    else:
        new_price = current_price
        recommendation["Strategy"] = "Maintain Price"

    if competitor_price and competitor_price < new_price:
        recommendation["Competitor Alert"] = "Competitor pricing lower than suggested price"

    recommendation["Suggested Price"] = new_price

    return recommendation




def competitor_price_alert(df, threshold_percent=5):
    """
    Alert when competitor price drops significantly.
    """

    df["Price Gap %"] = (
        (df["Competitor Pricing"] - df["Price"]) / df["Price"]
    ) * 100

    alerts = df[df["Price Gap %"] < -threshold_percent]

    return alerts[["Store ID", "Product ID", "Price", "Competitor Pricing", "Price Gap %"]]



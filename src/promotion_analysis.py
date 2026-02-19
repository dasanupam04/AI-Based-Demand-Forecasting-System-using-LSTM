import pandas as pd
import numpy as np


def promotion_uplift_analysis(df):
    """
    Compare average demand during promotion vs non-promotion.
    """

    df["Holiday/Promotion"] = df["Holiday/Promotion"].astype(str)

    promo_data = df[df["Holiday/Promotion"].astype(str).str.lower().isin(["yes", "1", "true"])]
    non_promo_data = df[~df.index.isin(promo_data.index)]


    promo_avg = promo_data["Units Sold"].mean()
    non_promo_avg = non_promo_data["Units Sold"].mean()

    uplift = ((promo_avg - non_promo_avg) / non_promo_avg) * 100

    return {
        "Promotion Avg Sales": promo_avg,
        "Non-Promotion Avg Sales": non_promo_avg,
        "Uplift %": uplift
    }




def promotion_effectiveness_score(df):
    """
    Measure how promotions performed vs forecast.
    """

    promo_df = df[df["Holiday/Promotion"] == "Yes"]

    promo_df["Forecast Error"] = (
        promo_df["Units Sold"] - promo_df["Demand Forecast"]
    )

    avg_error = promo_df["Forecast Error"].mean()

    return {
        "Average Forecast Error During Promotion": avg_error
    }




def holiday_impact_analysis(df):
    """
    Analyze demand during holiday periods.
    """

    # Detect holiday rows dynamically
    holiday_df = df[df["Holiday/Promotion"].astype(str).str.lower().isin(
        ["yes", "holiday", "1", "true"]
    )]

    if holiday_df.empty:
        return {
            "Holiday Avg Sales": 0,
            "Overall Avg Sales": df["Units Sold"].mean(),
            "Holiday Impact %": 0
        }

    holiday_avg = holiday_df["Units Sold"].mean()
    overall_avg = df["Units Sold"].mean()

    impact_percent = ((holiday_avg - overall_avg) / overall_avg) * 100

    return {
        "Holiday Avg Sales": holiday_avg,
        "Overall Avg Sales": overall_avg,
        "Holiday Impact %": impact_percent
    }


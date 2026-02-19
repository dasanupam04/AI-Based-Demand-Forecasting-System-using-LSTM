import numpy as np


def generate_recommendations(
    forecast_total,
    safety_stock,
    current_inventory,
    elasticity,
    competitor_alerts,
    region_growth_data,
    holiday_impact,
    promotion_uplift,
    category_profitability=None,
    region_volatility=None,
    region_efficiency=None,
    seasonality_strength_score=None,
    long_term_trend=None,
    demand_segment=None
):

    """
    Generate AI-driven strategic recommendations using multi-level intelligence.
    """

    recommendations = []

    # ---------------- Inventory Logic ----------------
    reorder_point = forecast_total + safety_stock

    if current_inventory < reorder_point:
        recommendations.append(
            f"Increase inventory by {int(reorder_point - current_inventory)} units to avoid stockout."
        )

    # ---------------- Pricing Logic ----------------
    if elasticity < -1:
        recommendations.append(
            "Demand is highly price sensitive. Consider reducing price slightly to increase volume."
        )
    elif -1 < elasticity < 0:
        recommendations.append(
            "Demand is relatively inelastic. Consider small price increase to improve margins."
        )

    # ---------------- Competitor Monitoring ----------------
    if competitor_alerts is not None and len(competitor_alerts) > 0:
        recommendations.append(
            "Competitor undercut detected. Review pricing strategy immediately."
        )

    # ---------------- Regional Growth ----------------
    if region_growth_data is not None:
        declining_regions = region_growth_data[
            region_growth_data["Growth Rate %"] < 0
        ]

        if len(declining_regions) > 0:
            recommendations.append(
                "Some regions show declining growth. Consider targeted regional campaigns."
            )

    # ---------------- Holiday Impact ----------------
    if holiday_impact and holiday_impact.get("Holiday Impact %", 0) > 15:
        recommendations.append(
            "Holiday demand impact is strong. Increase stock before major holidays."
        )

    # ---------------- Promotion Uplift ----------------
    if promotion_uplift and promotion_uplift.get("Uplift %", 0) > 10:
        recommendations.append(
            "Promotions significantly boost demand. Plan strategic future campaigns."
        )

    # ---------------- Category Profitability ----------------
    if category_profitability is not None:
        low_margin_categories = category_profitability[
            category_profitability["Profit Margin %"] < 10
        ]

        if len(low_margin_categories) > 0:
            recommendations.append(
                "Some categories have low margins. Consider price optimization or cost reduction."
            )

    # ---------------- Region Volatility ----------------
    if region_volatility is not None:
        high_vol_regions = region_volatility[
            region_volatility["Demand Volatility"] >
            region_volatility["Demand Volatility"].mean()
        ]

        if len(high_vol_regions) > 0:
            recommendations.append(
                "High demand volatility detected in some regions. Maintain higher safety stock."
            )

    # ---------------- Stock Efficiency ----------------
    if region_efficiency is not None:
        low_eff_regions = region_efficiency[
            region_efficiency["Stock Efficiency"] <
            region_efficiency["Stock Efficiency"].mean()
        ]

        if len(low_eff_regions) > 0:
            recommendations.append(
                "Low stock efficiency observed. Consider redistributing inventory."
            )

    # ---------------- Seasonality Strength ----------------
    if seasonality_strength_score is not None and seasonality_strength_score > 0.3:
        recommendations.append(
            "Strong seasonality detected. Align inventory and promotions with seasonal peaks."
        )

    # ---------------- Long-Term Trend ----------------
    if long_term_trend is not None:
        if long_term_trend.iloc[-1] > long_term_trend.mean():
            recommendations.append(
                "Long-term demand trend is upward. Consider expansion strategy."
            )
        else:
            recommendations.append(
                "Long-term demand trend is weakening. Monitor demand carefully."
            )

    if len(recommendations) == 0:
        recommendations.append("System stable. No immediate action required.")

    return recommendations



    # ---------------- Demand Segmentation Logic ----------------

    if demand_segment == "Highly Volatile":
        recommendations.append(
            "Product demand is highly volatile. Maintain higher safety stock and avoid aggressive pricing."
        )

    elif demand_segment == "Stable":
        recommendations.append(
            "Product demand is stable. Consider lean inventory strategy and margin optimization."
        )





import numpy as np
def simulate_price_change(current_price, elasticity, change_percent):
    """
    Simulate demand impact if price changes.
    """

    price_factor = 1 + (change_percent / 100)

    demand_change_percent = elasticity * (change_percent)

    return {
        "New Price": current_price * price_factor,
        "Estimated Demand Change %": demand_change_percent
    }


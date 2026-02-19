import numpy as np


def generate_inventory_decision(model,
                                scaled_demand,
                                scaler,
                                residual_std,
                                window_size=30,
                                forecast_days=7,
                                current_inventory=500):

    last_sequence = scaled_demand[-window_size:]
    future_predictions = []

    for _ in range(forecast_days):
        pred = model.predict(last_sequence.reshape(1, window_size, 1))
        future_predictions.append(pred[0][0])
        last_sequence = np.append(last_sequence[1:], pred)

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_demand = scaler.inverse_transform(future_predictions)

    total_forecast = np.sum(future_demand)

    Z = 1.96
    safety_stock = Z * residual_std

    reorder_point = total_forecast + safety_stock

    if current_inventory < reorder_point:
        reorder_quantity = reorder_point - current_inventory
        status = "REORDER REQUIRED"
    else:
        reorder_quantity = 0
        status = "INVENTORY SAFE"

    return {
        "forecast_7_days": float(total_forecast),
        "safety_stock": float(safety_stock),
        "reorder_point": float(reorder_point),
        "reorder_quantity": float(reorder_quantity),
        "status": status
    }

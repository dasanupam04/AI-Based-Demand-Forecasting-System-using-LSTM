import os
import numpy as np
from tensorflow.keras.models import load_model

from src.data_utils import filter_store_product
from src.preprocessing import scale_series, create_sequences, time_series_split
from src.decision_engine import generate_inventory_decision


def run_pipeline_for_product(df, store_id, product_id, window_size=30):

    # Filter data
    ts_df = filter_store_product(df, store_id, product_id)

    if len(ts_df) < window_size + 50:
        print(f"Skipping {product_id} — Not enough data")
        return None

    demand = ts_df['Units Sold'].values.reshape(-1, 1)

    # Preprocess
    scaled_demand, scaler = scale_series(demand)
    X, y = create_sequences(scaled_demand, window_size)
    X_train, X_test, y_train, y_test = time_series_split(X, y)

    # Load pre-trained base model (or retrain if needed)
    model = load_model("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\outputs\\lstm_model.keras")


    # Predict
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inversef_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_test)

    # Compute residual std for uncertainty
    residual_std = np.std(y_actual - y_pred)

    # Generate decision
    decision = generate_inventory_decision(
        model=model,
        scaled_demand=scaled_demand,
        scaler=scaler,
        residual_std=residual_std,
        window_size=window_size
    )

    return {
        "store": store_id,
        "product": product_id,
        "decision": decision
    }

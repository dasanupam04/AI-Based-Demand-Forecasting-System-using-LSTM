import sys
import os
import numpy as np
import matplotlib.pyplot as plt

#Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

#Import modules
from src.data_utils import load_data, filter_store_product
from src.preprocessing import scale_series, create_sequences, time_series_split
from tensorflow.keras.models import load_model

#Load model
model = load_model("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\outputs\\lstm_model.keras")

#Recreate data pipeline
df = load_data("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\data\\raw\\retail_store_inventory.csv")
ts_df = filter_store_product(df, "S001", "P0001")

demand = ts_df['Units Sold'].values.reshape(-1, 1)
scaled_demand, scaler = scale_series(demand)

WINDOW_SIZE = 30
X, y = create_sequences(scaled_demand, WINDOW_SIZE)
X_train, X_test, y_train, y_test = time_series_split(X, y)


#Predict next 7 days
future_steps = 7
last_sequence = scaled_demand[-WINDOW_SIZE:]
future_predictions = []
for _ in range(future_steps):
    pred = model.predict(last_sequence.reshape(1, WINDOW_SIZE, 1))
    future_predictions.append(pred[0][0])
    last_sequence = np.append(last_sequence[1:], pred)

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_demand = scaler.inverse_transform(future_predictions)

#Estimate uncertainty (simple version)
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_test)

residual_std = np.std(y_actual - y_pred)


#Compute safety stock
Z = 1.96
safety_stock = Z * residual_std
print("Recommended Safety Stock:", round(safety_stock, 2))

#Inventory Decision Logic
current_inventory = 500 


#Total expected demand next 7 days
total_forecast = np.sum(future_demand)
print("7-Day Forecasted Demand:", round(total_forecast, 2))


#Decision Rule
reorder_point = total_forecast + safety_stock
if current_inventory < reorder_point:
    print("⚠️ Reorder Required")
    reorder_quantity = reorder_point - current_inventory
    print("Recommended Reorder Quantity:", round(reorder_quantity, 2))
else:
    print("✅ Inventory Level is Safe")



#Risk Classification
if current_inventory < total_forecast:
    print("🚨 High Stockout Risk")
elif current_inventory > total_forecast + 2 * safety_stock:
    print("⚠️ Overstock Risk")
else:
    print("📦 Balanced Inventory Level")






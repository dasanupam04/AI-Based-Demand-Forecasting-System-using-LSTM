import sys
import os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

from src.data_utils import load_data, filter_store_product
from src.preprocessing import scale_series, create_sequences, time_series_split
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error


model = load_model("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\outputs\\lstm_model.keras")


df = load_data("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\data\\raw\\retail_store_inventory.csv")
ts_df = filter_store_product(df, "S001", "P0001")

demand = ts_df['Units Sold'].values.reshape(-1, 1)
scaled_demand, scaler = scale_series(demand)

WINDOW_SIZE = 30
X, y = create_sequences(scaled_demand, WINDOW_SIZE)
X_train, X_test, y_train, y_test = time_series_split(X, y)


y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_test)


historical_mae = mean_absolute_error(y_actual, y_pred)
print("Historical MAE:", historical_mae)
RECENT_WINDOW = 30

recent_actual = y_actual[-RECENT_WINDOW:]
recent_pred = y_pred[-RECENT_WINDOW:]
recent_mae = mean_absolute_error(recent_actual, recent_pred)
print("Recent MAE:", recent_mae)


THRESHOLD = 1.3  # 30% increase
if recent_mae > THRESHOLD * historical_mae:
    print("⚠️ Drift Detected: Model performance degrading.")
else:
    print("✅ No significant drift detected.")


error_series = np.abs(y_actual - y_pred)

plt.figure(figsize=(10,4))
plt.plot(error_series, label="Absolute Error")
plt.axhline(historical_mae, color='r', linestyle='--', label="Historical MAE")
plt.legend()
plt.title("Error Monitoring for Drift Detection")
plt.show()







import sys
import os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

from src.data_utils import load_data, filter_store_product
from src.preprocessing import scale_series, create_sequences, time_series_split
from tensorflow.keras.models import load_model

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


residuals = y_actual - y_pred

residual_std = np.std(residuals)
print("Residual Std Dev:", residual_std)


Z = 1.96

upper_bound = y_pred + Z * residual_std
lower_bound = y_pred - Z * residual_std

plt.figure(figsize=(12, 5))

plt.plot(y_actual, label="Actual")
plt.plot(y_pred, label="Prediction")
plt.fill_between(
    range(len(y_pred)),
    lower_bound.flatten(),
    upper_bound.flatten(),
    alpha=0.3,
    label="95% Confidence Interval"
)

plt.legend()
plt.title("LSTM Forecast with Uncertainty Bands")
plt.show()

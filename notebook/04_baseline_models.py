import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM")))
sys.path.append(PROJECT_ROOT)


from src.data_utils import load_data, filter_store_product
from src.preprocessing import scale_series, create_sequences, time_series_split



df = load_data("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\data\\raw\\retail_store_inve
               ntory.csv")
ts_df = filter_store_product(df, "S001", "P0001")

demand = ts_df['Units Sold'].values.reshape(-1, 1)
scaled_demand, scaler = scale_series(demand)

WINDOW_SIZE = 30
X, y = create_sequences(scaled_demand, WINDOW_SIZE)
X_train, X_test, y_train, y_test = time_series_split(X, y)


#Create naive predictions
y_naive = X_test[:, -1, 0].reshape(-1, 1)

#Define moving average window
MA_WINDOW = 7

#Compute moving average predictions
y_ma = []

for seq in X_test:
    y_ma.append(seq[-MA_WINDOW:, 0].mean())

y_ma = np.array(y_ma).reshape(-1, 1)



#nverse transform 
y_test_actual = scaler.inverse_transform(y_test)
y_naive_actual = scaler.inverse_transform(y_naive)
y_ma_actual = scaler.inverse_transform(y_ma)


from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} → MAE: {mae:.2f}, RMSE: {rmse:.2f}")

evaluate(y_test_actual, y_naive_actual, "Naive Forecast")
evaluate(y_test_actual, y_ma_actual, "Moving Average Forecast")

plt.figure(figsize=(12, 5))
plt.plot(y_test_actual, label="Actual")
plt.plot(y_naive_actual, label="Naive")
plt.plot(y_ma_actual, label="Moving Average")
plt.legend()
plt.title("Baseline Model Comparison")
plt.show()








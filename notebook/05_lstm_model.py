import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

from src.data_utils import load_data, filter_store_product
from src.preprocessing import scale_series, create_sequences, time_series_split


df = load_data("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\data\\raw\\retail_store_inventory.csv")
ts_df = filter_store_product(df, "S001", "P0001")

demand = ts_df['Units Sold'].values.reshape(-1, 1)
scaled_demand, scaler = scale_series(demand)

WINDOW_SIZE = 30
X, y = create_sequences(scaled_demand, WINDOW_SIZE)
X_train, X_test, y_train, y_test = time_series_split(X, y)

print(X_train.shape, y_train.shape)



#Build LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)


model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("LSTM Training Loss")
plt.show()


y_lstm_scaled = model.predict(X_test)



y_lstm = scaler.inverse_transform(y_lstm_scaled)
y_test_actual = scaler.inverse_transform(y_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test_actual, y_lstm)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_lstm))

print(f"LSTM → MAE: {mae:.2f}, RMSE: {rmse:.2f}")


plt.figure(figsize=(12, 5))
plt.plot(y_test_actual, label="Actual")
plt.plot(y_lstm, label="LSTM Prediction")
plt.legend()
plt.title("LSTM vs Actual Demand")
plt.show()


model.save("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\outputs\\lstm_model.keras")








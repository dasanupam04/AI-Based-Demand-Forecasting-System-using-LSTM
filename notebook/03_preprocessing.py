import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

#Import utilities
from src.data_utils import load_data, filter_store_product
df = load_data("X:\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\data\\raw\\retail_store_inventory.csv")
ts_df = filter_store_product(df, "S001", "P0001")

ts_df.head()

#Extract demand values
demand = ts_df['Units Sold'].values.reshape(-1, 1)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
demand_scaled = scaler.fit_transform(demand)
plt.figure(figsize=(10, 4))
plt.plot(demand, label="Original")
plt.plot(demand_scaled, label="Scaled")
plt.legend()
plt.title("Demand: Original vs Scaled")
plt.show()



#Create sequences
X = []
y = []
#Define window size
for i in range(len(demand_scaled) - 30):
    X.append(demand_scaled[i:i + 30])
    y.append(demand_scaled[i + 30])

X = np.array(X)
y = np.array(y)


print("X shape:", X.shape)
print("y shape:", y.shape)



train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


from src.preprocessing import scale_series, create_sequences, time_series_split

demand = ts_df['Units Sold'].values.reshape(-1, 1)

scaled_demand, scaler = scale_series(demand)
X, y = create_sequences(scaled_demand, 30)
X_train, X_test, y_train, y_test = time_series_split(X, y)



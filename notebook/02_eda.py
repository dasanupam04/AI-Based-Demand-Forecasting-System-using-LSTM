import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)


from src.data_utils import load_data, filter_store_product

df = load_data("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\data\\raw\\retail_store_inventory.csv")
ts_df = filter_store_product(df, "S001", "P0001")

print(ts_df.shape)
print(ts_df.head())


plt.figure()
plt.plot(ts_df['Date'], ts_df['Units Sold'])
plt.title("Daily Demand Trend (Store S001, Product P0001)")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.show()


print(ts_df['Units Sold'].describe())


ts_df['Rolling_Mean_30'] = ts_df['Units Sold'].rolling(window=30).mean()

plt.figure(figsize=(12, 5))
plt.plot(ts_df['Date'], ts_df['Units Sold'], alpha=0.5, label='Actual')
plt.plot(ts_df['Date'], ts_df['Rolling_Mean_30'], label='30-Day Rolling Mean')
plt.legend()
plt.title("Demand with Rolling Mean")
plt.show()


ts_df.isna().sum()

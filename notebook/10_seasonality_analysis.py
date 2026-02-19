import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

from src.data_utils import load_data
from src.seasonality_analysis import (
    time_series_decomposition,
    monthly_seasonal_pattern,
    quarterly_trend_analysis
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "retail_store_inventory.csv")
df = load_data("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\data\\raw\\retail_store_inventory.csv")

decomp = time_series_decomposition(df, store_id="S001", product_id="P0001")

decomp.plot()
plt.show()




monthly_pattern = monthly_seasonal_pattern(df)

plt.plot(monthly_pattern["Month"], monthly_pattern["Units Sold"])
plt.title("Monthly Seasonal Demand Pattern")
plt.xlabel("Month")
plt.ylabel("Average Units Sold")
plt.show()



quarterly = quarterly_trend_analysis(df)
quarterly


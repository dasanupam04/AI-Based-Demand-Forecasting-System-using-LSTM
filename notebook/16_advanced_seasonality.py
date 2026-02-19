import sys
import os
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

from src.data_utils import load_data
from src.seasonality_analysis import (
    category_decomposition,
    region_decomposition,
    seasonality_strength,
    long_cycle_trend
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "retail_store_inventory.csv")
df = load_data("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\data\\raw\\retail_store_inventory.csv")




decomp = category_decomposition(df, "Electronics")
decomp.plot()
plt.show()

seasonality_strength(decomp)




reg_decomp = region_decomposition(df, "North")
reg_decomp.plot()
plt.show()

seasonality_strength(reg_decomp)



trend = long_cycle_trend(df)
trend.plot()
plt.title("Long-Term Demand Trend")
plt.show()






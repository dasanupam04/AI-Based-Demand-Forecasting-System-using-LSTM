import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

from src.data_utils import load_data
from src.category_analysis import (
    category_demand_share,
    category_growth_rate,
    category_profitability,
    category_seasonal_index
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "retail_store_inventory.csv")
df = load_data("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\data\\raw\\retail_store_inventory.csv")


category_demand_share(df)


category_growth_rate(df).head()


category_profitability(df)


category_seasonal_index(df).head()

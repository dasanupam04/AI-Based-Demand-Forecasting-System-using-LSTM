import sys
import os
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

from src.data_utils import load_data, filter_store_product

# Load data
df = load_data("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\data\\raw\\retail_store_inventory.csv")

print("Dataset shape:", df.shape)
print("Columns:")
for col in df.columns:
    print(col)

df.head()


# Filter one store-product
ts_df = filter_store_product(df, "S001", "P0001")
ts_df.head()

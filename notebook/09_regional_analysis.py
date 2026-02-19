import sys
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

from
src.data_utils import load_data
from src.regional_insights import (
    region_store_summary,
    region_demand_heatmap,
    suggest_inventory_redistribution
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "retail_store_inventory.csv")
df = load_data("X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM\\data\\raw\\retail_store_inventory.csv")


summary = region_store_summary(df)
summary.head()

heatmap_data = region_demand_heatmap(df)
heatmap_data

redistribution = suggest_inventory_redistribution(df)
redistribution


from src.regional_insights import (
    region_growth_analysis,
    region_profitability_analysis
)


growth_data = region_growth_analysis(df)
growth_data.head()


profit_data = region_profitability_analysis(df)
profit_data


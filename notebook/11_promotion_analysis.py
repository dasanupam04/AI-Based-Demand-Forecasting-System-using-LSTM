import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

from src.data_utils import load_data
from src.promotion_analysis import (
    promotion_uplift_analysis,
    promotion_effectiveness_score,
    holiday_impact_analysis
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "retail_store_inventory.csv")
df = load_data(DATA_PATH)


promotion_uplift_analysis(df)


promotion_effectiveness_score(df)


holiday_impact_analysis(df)



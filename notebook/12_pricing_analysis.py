import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

from src.data_utils import load_data
from src.pricing_engine import (
    estimate_price_elasticity,
    suggest_optimal_price,
    competitor_price_alert
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "retail_store_inventory.csv")
df = load_data(DATA_PATH)


elasticity = estimate_price_elasticity(df, store_id="S001", product_id="P0001")
elasticity


current_price = df["Price"].mean()
competitor_price = df["Competitor Pricing"].mean()

suggest_optimal_price(current_price, elasticity, competitor_price)


competitor_price_alert(df).head()

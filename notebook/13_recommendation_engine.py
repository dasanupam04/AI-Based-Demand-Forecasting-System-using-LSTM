import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "X:\\Data Science Project\\AI-Based-Demand-Forecasting-System-using-LSTM"))
sys.path.append(PROJECT_ROOT)

from src.recommendation_engine import generate_recommendations
from src.data_utils import load_data
from src.pricing_engine import estimate_price_elasticity, competitor_price_alert
from src.promotion_analysis import promotion_uplift_analysis, holiday_impact_analysis
from src.regional_insights import region_growth_analysis

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "retail_store_inventory.csv")
df = load_data(DATA_PATH)


forecast_total = 500
safety_stock = 80
current_inventory = 400

elasticity = estimate_price_elasticity(df, "S001", "P0001")
competitor_alerts = competitor_price_alert(df)

growth_data = region_growth_analysis(df)
holiday_impact = holiday_impact_analysis(df)
promotion_uplift = promotion_uplift_analysis(df)


generate_recommendations(
    forecast_total,
    safety_stock,
    current_inventory,
    elasticity,
    competitor_alerts,
    growth_data,
    holiday_impact,
    promotion_uplift
)



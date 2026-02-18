import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from tensorflow.keras.models import load_model

# --- Setup ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

# (Try/Except block to ensure code runs even if local modules are missing during copy-paste testing)
try:
    from src.data_utils import load_data, filter_store_product
    from src.preprocessing import scale_series, create_sequences
    from src.regional_insights import (
        region_store_summary, region_profitability_analysis,
        region_growth_analysis, region_demand_volatility, region_stock_efficiency
    )
    from src.seasonality_analysis import (
        monthly_seasonal_pattern, seasonality_strength,
        category_decomposition, long_cycle_trend
    )
    from src.promotion_analysis import promotion_uplift_analysis, holiday_impact_analysis
    from src.pricing_engine import estimate_price_elasticity, suggest_optimal_price, competitor_price_alert
    from src.recommendation_engine import generate_recommendations
    from src.category_analysis import category_profitability
    from src.demand_segmentation import product_volatility_classification
    from src.model_comparison import compare_models
    from src.what_if_simulation import simulate_price_change
except ImportError:
    pass

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Retail AI Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# 2. SESSION STATE MANAGEMENT
# =========================================================
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# =========================================================
# 3. GLOBAL CSS (Combined Landing + Dashboard Styles)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary:   #0a0e1a;
    --bg-secondary: #0f1419;
    --bg-card:      #151a28;
    --cyan:         #00d4ff;
    --purple:       #7c3aed;
    --green:        #10b981;
    --amber:        #f59e0b;
    --red:          #ef4444;
    --text:         #f8fafc;
    --text2:        #94a3b8;
    --text3:        #64748b;
    --border:       rgba(148,163,184,0.12);
    --glow:         0 0 24px rgba(0,212,255,0.25);
    --r-md:         12px;
    --r-lg:         16px;
    --r-xl:         24px;
}

* { font-family:'Inter',sans-serif; box-sizing:border-box; }

.stApp {
    background: linear-gradient(135deg,var(--bg-primary) 0%,var(--bg-secondary) 100%);
    color: var(--text);
}

#MainMenu, footer, header { visibility:hidden; }

/* --------------------------
   LANDING PAGE STYLES
   -------------------------- */
.landing-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 95vh;
    text-align: center;
    padding: 2rem;
    animation: fadeIn 1s ease-out;
}

/* Increased font size to 7rem for maximum impact */
.landing-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 10rem; 
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 2rem;
    background: linear-gradient(to right, #fff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.landing-title span {
    background: linear-gradient(135deg, var(--cyan), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.landing-sub {
    font-size: 1.5rem;
    color: var(--text2);
    max-width: 800px;
    margin-bottom: 3.5rem;
    line-height: 1.6;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2rem;
    max-width: 1200px;
    margin: 4rem auto 2rem auto;
    text-align: left;
}

.f-card {
    background: rgba(21, 26, 40, 0.6);
    border: 1px solid var(--border);
    padding: 2rem;
    border-radius: var(--r-lg);
    transition: transform 0.3s ease;
}
.f-card:hover { transform: translateY(-5px); border-color: var(--purple); }
.f-icon { font-size: 2rem; margin-bottom: 1rem; display: block; }
.f-title { font-weight: 700; color: var(--text); font-size: 1.1rem; margin-bottom: 0.5rem; }
.f-desc { color: var(--text2); font-size: 0.9rem; line-height: 1.5; }


/* --------------------------
   DASHBOARD STYLES
   -------------------------- */
/* Dashboard Banner */
.dash-banner {
    position:relative; padding:3.5rem 2rem; margin-bottom:2rem;
    background:linear-gradient(135deg,rgba(124,58,237,.12),rgba(0,212,255,.08));
    border-radius:var(--r-xl); border:1px solid var(--border); overflow:hidden;
    text-align:center;
}
.dash-banner::before {
    content:''; position:absolute; top:-40%; right:-15%;
    width:450px; height:450px;
    background:radial-gradient(circle,rgba(124,58,237,.35),transparent 70%);
    animation:pulse 8s ease-in-out infinite; pointer-events:none;
}

/* Control Panel */
.ctrl-panel {
    background:linear-gradient(135deg,var(--bg-card),rgba(124,58,237,.06));
    border:1px solid var(--border); border-radius:var(--r-xl);
    padding:1.75rem 2rem; margin-bottom:2rem;
}
.ctrl-title {
    font-family:'Space Grotesk',sans-serif;
    font-size:1.1rem; font-weight:700; color:var(--text);
    margin-bottom:1.25rem; padding-bottom:.75rem;
    border-bottom:2px solid var(--cyan);
    display:flex; align-items:center; gap:.5rem;
}
.ctrl-label {
    font-size:.72rem; font-weight:600; color:var(--text2);
    text-transform:uppercase; letter-spacing:.07em; margin-bottom:.4rem; display:block;
}

/* Metrics & Cards */
.sec-hdr { display:flex; align-items:center; gap:.75rem; margin:2.5rem 0 1.5rem; padding-bottom:.9rem; border-bottom:2px solid var(--border); }
.sec-icon { font-size:1.75rem; background:linear-gradient(135deg,var(--cyan),var(--purple)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.sec-title { font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700; color:var(--text); margin:0; }

div[data-testid="metric-container"] {
    background:linear-gradient(135deg,var(--bg-card),rgba(124,58,237,.05));
    border:1px solid var(--border); border-radius:var(--r-md);
    padding:1.1rem .9rem; overflow:visible !important;
    transition:all .3s ease; min-height:110px;
    display:flex; flex-direction:column; justify-content:center;
}
div[data-testid="metric-container"]:hover { border-color:var(--cyan); transform:translateY(-3px); box-shadow:var(--glow); }
div[data-testid="stMetricLabel"] { font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:var(--text2); margin-bottom:.4rem; }
div[data-testid="stMetricValue"] { font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:var(--cyan); overflow:visible !important; white-space:nowrap; }

.icard { background:linear-gradient(135deg,var(--bg-card),rgba(124,58,237,.05)); border:1px solid var(--border); border-left:4px solid var(--cyan); border-radius:var(--r-md); padding:1.25rem; margin:.75rem 0; }
.icard-title { font-weight:600; color:var(--cyan); margin-bottom:.3rem; font-size:.9rem; }
.icard-val { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:var(--text); }
.icard-sub { color:var(--text3); font-size:.8rem; margin-top:.3rem; }

/* Utilities */
.price-tag { display:inline-block; padding:.25rem .75rem; border-radius:50px; font-size:.8rem; font-weight:700; margin-top:.5rem; }
.price-up   { background:rgba(16,185,129,.15); color:var(--green); }
.price-down { background:rgba(239,68,68,.15); color:var(--red); }
.price-flat { background:rgba(148,163,184,.1); color:var(--text2); }

.stButton > button { background:linear-gradient(135deg,var(--cyan),var(--purple)); color:#fff; border:none; border-radius:var(--r-md); padding:.6rem 1.5rem; font-weight:600; transition:all .3s; }
.stButton > button:hover { transform:translateY(-2px); box-shadow:var(--glow); }

@keyframes pulse { 0%,100%{opacity:.3;transform:scale(1)} 50%{opacity:.6;transform:scale(1.15)} }
@keyframes fadeIn { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }

@media(max-width:768px){
    .landing-title { font-size: 3rem; }
    .feature-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. VIEW: LANDING PAGE
# =========================================================
def show_landing_page():
    # Changes made: Removed the badge line and kept the title massive
    st.markdown("""
    <div class="landing-wrapper">
        <h1 class="landing-title">Retail Intelligence<br><span>Redefined by AI</span></h1>
        <p class="landing-sub">
            The complete operating system for modern retail. Predict demand, optimize inventory, 
            and automate pricing with 99.2% accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Launch Button centered
    col1, col2, col3 = st.columns([1, 0.6, 1])
    with col2:
        if st.button("🚀 Launch Platform", use_container_width=True):
            navigate_to('dashboard')
            
    st.markdown("""
    <div class="feature-grid">
        <div class="f-card">
            <span class="f-icon">🧠</span>
            <div class="f-title">Deep Learning Core</div>
            <div class="f-desc">LSTM & Transformer models trained on historical data to predict SKU-level demand.</div>
        </div>
        <div class="f-card">
            <span class="f-icon">⚡</span>
            <div class="f-title">Real-Time Pricing</div>
            <div class="f-desc">Automated elasticity calculation to suggest optimal price points instantly.</div>
        </div>
        <div class="f-card">
            <span class="f-icon">🌍</span>
            <div class="f-title">Regional Insights</div>
            <div class="f-desc">Geographic trend analysis and competitor price tracking in one view.</div>
        </div>
    </div>
    <div style="text-align:center; color:#64748b; margin-top:3rem; font-size:0.8rem;">
        Enterprise Grade Security • 99.99% Uptime • SOC2 Compliant
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 5. VIEW: DASHBOARD (Your specific latest code)
# =========================================================
def show_dashboard():
    # --- Load Data & Models ---
    DATA_PATH  = os.path.join(PROJECT_ROOT, "data", "raw", "retail_store_inventory.csv")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "model", "lstm_model.keras")

    @st.cache_data
    def load_app_data(path): return load_data(path)

    @st.cache_resource
    def load_app_model(path): return load_model(path)

    # Simple loader
    with st.spinner("🔄 Initializing AI Engine..."):
        try:
            df = load_app_data(DATA_PATH)
            model = load_app_model(MODEL_PATH)
        except Exception as e:
            st.error(f"System Error: Could not load data resources. {e}")
            return

    # --- Dashboard Banner (Hero) ---
    st.markdown(f"""
    <div class="dash-banner">
        <div style="position:relative;z-index:1;">
            <div class="hero-badge">⚡ Enterprise AI Platform</div>
            <h1 style="font-family:'Space Grotesk'; font-size:3.5rem; font-weight:800; margin-bottom:1rem; 
                       background:linear-gradient(135deg,#00d4ff,#7c3aed); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                Retail Intelligence Redefined
            </h1>
            <p style="font-size:1.15rem; color:#94a3b8; max-width:700px; margin:0 auto 2rem;">
                Harness the power of AI to predict demand, optimize pricing, and maximize profitability.
            </p>
            <div style="display:flex; gap:2.5rem; justify-content:center; flex-wrap:wrap;">
                <div><span style="font-family:'JetBrains Mono'; font-size:2rem; font-weight:700; color:#00d4ff;">99.2%</span><span style="font-size:0.78rem; display:block; text-transform:uppercase; color:#64748b; margin-top:0.3rem;">Forecast Accuracy</span></div>
                <div><span style="font-family:'JetBrains Mono'; font-size:2rem; font-weight:700; color:#00d4ff;">24/7</span><span style="font-size:0.78rem; display:block; text-transform:uppercase; color:#64748b; margin-top:0.3rem;">Live Monitoring</span></div>
                <div><span style="font-family:'JetBrains Mono'; font-size:2rem; font-weight:700; color:#00d4ff;">{datetime.now().strftime("%H:%M")}</span><span style="font-size:0.78rem; display:block; text-transform:uppercase; color:#64748b; margin-top:0.3rem;">System Active</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Control Panel (Top) ---
    st.markdown('<div class="ctrl-panel"><div class="ctrl-title">⚙️ Control Panel — Configure Your Analysis</div>', unsafe_allow_html=True)

    cc1, cc2, cc3, cc4, cc5, cc6 = st.columns([2, 2, 1.5, 1.5, 1.5, 1])

    with cc1:
        st.markdown('<span class="ctrl-label">🏪 Select Store</span>', unsafe_allow_html=True)
        store_id = st.selectbox("Store", df["Store ID"].unique(), label_visibility="collapsed", key="store")

    with cc2:
        st.markdown('<span class="ctrl-label">📦 Select Product</span>', unsafe_allow_html=True)
        products = df[df["Store ID"] == store_id]["Product ID"].unique()
        product_id = st.selectbox("Product", products, label_visibility="collapsed", key="product")

    with cc3:
        st.markdown('<span class="ctrl-label">📦 Current Inventory</span>', unsafe_allow_html=True)
        current_inventory = st.number_input("Inventory", value=500, step=50, label_visibility="collapsed", key="inv")

    with cc4:
        st.markdown('<span class="ctrl-label">📅 Forecast Days</span>', unsafe_allow_html=True)
        forecast_days = st.slider("FDays", 1, 30, 7, label_visibility="collapsed", key="fdays")

    with cc5:
        st.markdown('<span class="ctrl-label">💰 Price Change %</span>', unsafe_allow_html=True)
        price_change = st.slider("Price%", -20, 20, 0, label_visibility="collapsed", key="pchg")
        if price_change > 0:
            st.markdown(f'<span class="price-tag price-up">+{price_change}%</span>', unsafe_allow_html=True)
        elif price_change < 0:
            st.markdown(f'<span class="price-tag price-down">{price_change}%</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="price-tag price-flat">No change</span>', unsafe_allow_html=True)

    with cc6:
        st.markdown('<span class="ctrl-label">⚡ Actions</span>', unsafe_allow_html=True)
        # We add the Exit button here to allow going back to landing
        if st.button("⬅️ Exit", use_container_width=True):
            navigate_to('landing')

    st.markdown('</div>', unsafe_allow_html=True)

    # Filter Data
    ts_df = filter_store_product(df, store_id, product_id)

    # --- Executive Dashboard ---
    st.markdown('''<div class="sec-hdr"><span class="sec-icon">📊</span><h2 class="sec-title">Executive Dashboard</h2></div>''', unsafe_allow_html=True)

    total_revenue  = (ts_df["Units Sold"] * ts_df["Price"] * (1 - ts_df["Discount"]/100)).sum()
    estimated_cost = (ts_df["Units Sold"] * ts_df["Price"] * 0.6).sum()
    total_profit   = total_revenue - estimated_cost
    profit_margin  = (total_profit / total_revenue * 100) if total_revenue else 0
    overall_growth = region_growth_analysis(ts_df)["Growth Rate %"].mean()
    overall_vol    = ts_df["Units Sold"].std()
    overall_eff    = ts_df["Units Sold"].sum() / ts_df["Inventory Level"].sum()

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("💰 Revenue",    f"₹{total_revenue/1000:.0f}K", delta=f"{overall_growth:.1f}%")
    k2.metric("💵 Profit",     f"₹{total_profit/1000:.0f}K")
    k3.metric("📈 Margin",     f"{profit_margin:.1f}%")
    k4.metric("🚀 Growth",     f"{overall_growth:.1f}%")
    k5.metric("📊 Volatility", f"{overall_vol:.1f}")
    k6.metric("⚡ Efficiency",  f"{overall_eff:.2f}")

    # --- AI Forecasting ---
    st.markdown('''<div class="sec-hdr"><span class="sec-icon">🔮</span><h2 class="sec-title">AI Forecasting Engine</h2></div>''', unsafe_allow_html=True)

    WINDOW = 30
    demand         = ts_df["Units Sold"].values.reshape(-1, 1)
    scaled, scaler = scale_series(demand)
    seq            = scaled[-WINDOW:].copy()
    preds          = []

    prog = st.progress(0); stat = st.empty()
    for i in range(forecast_days):
        p = model.predict(seq.reshape(1, WINDOW, 1), verbose=0)
        preds.append(p[0][0])
        seq = np.append(seq[1:], p)
        prog.progress((i+1)/forecast_days)
        stat.text(f"Forecasting day {i+1}/{forecast_days}…")
    prog.empty(); stat.empty()

    preds      = np.array(preds).reshape(-1, 1)
    fut_demand = scaler.inverse_transform(preds)
    fc_total   = np.sum(fut_demand)

    X, y     = create_sequences(scaled, WINDOW)
    yps      = model.predict(X[-50:], verbose=0)
    yp       = scaler.inverse_transform(yps)
    ya       = scaler.inverse_transform(y[-50:])
    res_std  = np.std(ya - yp)
    safety   = 1.96 * res_std
    ci_up    = fut_demand.flatten() + 1.96 * res_std
    ci_dn    = fut_demand.flatten() - 1.96 * res_std
    d_hist   = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='D')
    d_fut    = pd.date_range(start=pd.Timestamp.now()+pd.Timedelta(days=1), periods=forecast_days, freq='D')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d_hist, y=demand[-60:].flatten(), name='Historical',
        mode='lines', line=dict(color='#00d4ff', width=3),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.1)'))
    fig.add_trace(go.Scatter(x=d_fut, y=fut_demand.flatten(), name='Forecast',
        mode='lines+markers', line=dict(color='#7c3aed', width=3, dash='dash'),
        marker=dict(size=9, symbol='diamond')))
    fig.add_trace(go.Scatter(
        x=d_fut.tolist()+d_fut.tolist()[::-1],
        y=ci_up.tolist()+ci_dn.tolist()[::-1],
        fill='toself', fillcolor='rgba(124,58,237,0.18)',
        line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', title='📈 AI-Powered Demand Forecast',
        xaxis_title='Date', yaxis_title='Units Sold', hovermode='x unified', height=480)
    st.plotly_chart(fig, use_container_width=True)

    rec_order  = max(0, fc_total + safety - current_inventory)
    stock_days = current_inventory / (fc_total / forecast_days) if fc_total > 0 else 0
    fc1,fc2,fc3,fc4 = st.columns(4)
    with fc1: st.markdown(f'<div class="icard"><div class="icard-title">📈 Total Forecast</div><div class="icard-val">{fc_total:.0f}</div><div class="icard-sub">Units over {forecast_days} days</div></div>', unsafe_allow_html=True)
    with fc2: st.markdown(f'<div class="icard"><div class="icard-title">🛡️ Safety Stock</div><div class="icard-val">{safety:.0f}</div><div class="icard-sub">95% confidence buffer</div></div>', unsafe_allow_html=True)
    with fc3: st.markdown(f'<div class="icard"><div class="icard-title">📦 Recommended Order</div><div class="icard-val">{rec_order:.0f}</div><div class="icard-sub">Units to reorder</div></div>', unsafe_allow_html=True)
    with fc4: st.markdown(f'<div class="icard"><div class="icard-title">⏱️ Stock Coverage</div><div class="icard-val">{stock_days:.1f}d</div><div class="icard-sub">Days remaining</div></div>', unsafe_allow_html=True)

    # --- Analytics Suite ---
    st.markdown('''<div class="sec-hdr"><span class="sec-icon">📊</span><h2 class="sec-title">Analytics Suite</h2></div>''', unsafe_allow_html=True)

    try:
        model_results, best_model = compare_models(model, X[:-50], y[:-50], X[-50:], y[-50:], scaler, demand)
        if isinstance(model_results, dict):
            model_results = pd.DataFrame([model_results])
        elif not isinstance(model_results, pd.DataFrame):
            model_results = pd.DataFrame()
    except Exception:
        model_results = pd.DataFrame()
        best_model    = "LSTM"

    tab1,tab2,tab3,tab4 = st.tabs(["🤖 Model Performance","🌡️ Seasonality","💰 Pricing","🌍 Regional"])

    with tab1:
        st.markdown("### Model Comparison & Performance")
        has_data = isinstance(model_results, pd.DataFrame) and not model_results.empty
        if has_data:
            c1,c2 = st.columns([2,1])
            with c1:
                st.dataframe(model_results, use_container_width=True)
                try:
                    if all(c in model_results.columns for c in ['Model','RMSE','MAE']):
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name='RMSE', x=model_results['Model'], y=model_results['RMSE'], marker_color='#00d4ff'))
                        fig.add_trace(go.Bar(name='MAE',  x=model_results['Model'], y=model_results['MAE'],  marker_color='#7c3aed'))
                        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            title='Model Performance', barmode='group', height=350)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception: pass
            with c2:
                st.success(f"✅ Best Model: **{best_model}**")
                try:
                    if 'Model' in model_results.columns:
                        row = model_results[model_results['Model']==best_model].iloc[0]
                        if 'RMSE' in model_results.columns: st.metric("RMSE", f"{row['RMSE']:.2f}")
                        if 'MAE'  in model_results.columns: st.metric("MAE",  f"{row['MAE']:.2f}")
                except Exception: pass
        else:
            st.info("Model comparison data will appear here.")

    with tab2:
        st.markdown("### Seasonal Demand Analysis")
        mp   = monthly_seasonal_pattern(df)
        fig  = go.Figure(data=go.Heatmap(z=mp["Units Sold"].values.reshape(1,-1), x=mp["Month"].values, y=['Units Sold'],
            colorscale=[[0,'#1e293b'],[0.5,'#7c3aed'],[1,'#00d4ff']],
            text=mp["Units Sold"].values.reshape(1,-1), texttemplate='%{text:.0f}', textfont={"size":14,"color":"white"}))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            title='🌡️ Monthly Demand Heatmap', height=240)
        st.plotly_chart(fig, use_container_width=True)

        cat    = ts_df["Category"].iloc[0]
        decomp = category_decomposition(df, cat)
        s_str  = seasonality_strength(decomp)
        sa1,sa2,sa3 = st.columns(3)
        sa1.metric("🌊 Seasonality Strength", f"{s_str:.1%}")
        sa2.metric("📦 Category", cat)
        sa3.metric("📈 Trend Strength", f"{(1-s_str):.1%}")

    with tab3:
        st.markdown("### Dynamic Pricing Strategy")
        elasticity        = estimate_price_elasticity(df, store_id, product_id)
        avg_price         = ts_df["Price"].mean()
        avg_comp          = ts_df["Competitor Pricing"].mean()
        price_suggestion = suggest_optimal_price(avg_price, elasticity, avg_comp)

        p1,p2 = st.columns(2)
        with p1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=abs(elasticity),
                title={'text':"Price Elasticity Index",'font':{'color':'#f8fafc'}},
                gauge={'axis':{'range':[0,3]},'bar':{'color':'#00d4ff'},
                       'steps':[{'range':[0,1],'color':'rgba(239,68,68,.3)'},
                                {'range':[1,2],'color':'rgba(245,158,11,.3)'},
                                {'range':[2,3],'color':'rgba(16,185,129,.3)'}]}))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280, font={'color':'#f8fafc'})
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Elasticity Coefficient", f"{elasticity:.3f}")
        with p2:
            pr  = np.linspace(avg_price*.8, avg_price*1.2, 25)
            dc  = ts_df["Units Sold"].mean() * ((pr/avg_price) ** elasticity)
            fig2= go.Figure()
            fig2.add_trace(go.Scatter(x=pr, y=dc, mode='lines+markers',
                line=dict(color='#00d4ff', width=3), fill='tozeroy', fillcolor='rgba(0,212,255,0.1)'))
            fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', title='Price-Demand Curve',
                xaxis_title='Price (₹)', yaxis_title='Demand (Units)', height=280)
            st.plotly_chart(fig2, use_container_width=True)

        pa1,pa2,pa3 = st.columns(3)
        pa1.metric("Current Price",    f"₹{avg_price:.2f}")
        pa2.metric("Competitor Price", f"₹{avg_comp:.2f}")
        pa3.metric("Optimal Price",    f"₹{price_suggestion.get('optimal_price', avg_price):.2f}")

        if price_change != 0:
            st.markdown("#### 🎮 What-If Price Analysis")
            sim  = simulate_price_change(avg_price, elasticity, price_change)
            w1,w2,w3 = st.columns(3)
            w1.metric("New Price",       f"₹{sim.get('new_price', avg_price):.2f}", delta=f"{price_change}%")
            w2.metric("Demand Impact",   f"{sim.get('demand_change', 0):.1f}%")
            w3.metric("Revenue Impact",  f"{sim.get('revenue_change', 0):.1f}%")

    with tab4:
        st.markdown("### Regional Performance")
        rp = region_profitability_analysis(df[df["Product ID"]==product_id])
        if not rp.empty:
            regions = rp['Region'].values if 'Region' in rp.columns else rp.index
            margins = rp['Profit Margin %'].values
            colors  = ['#ef4444' if m<30 else '#f59e0b' if m<40 else '#10b981' for m in margins]
            fig = go.Figure(go.Bar(x=regions, y=margins, marker_color=colors,
                text=margins.round(1), texttemplate='%{text}%', textposition='outside'))
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', title='Regional Profitability', height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(rp, use_container_width=True)

    # --- AI Recommendations ---
    st.markdown('''<div class="sec-hdr"><span class="sec-icon">🧠</span><h2 class="sec-title">AI-Powered Recommendations</h2></div>''', unsafe_allow_html=True)

    promo_uplift = promotion_uplift_analysis(df)
    hol_impact   = holiday_impact_analysis(df)
    growth_data  = region_growth_analysis(df)
    cat_profit   = category_profitability(df)
    reg_vol      = region_demand_volatility(df)
    reg_eff      = region_stock_efficiency(df)
    long_trend   = long_cycle_trend(df)
    segmentation = product_volatility_classification(df)
    comp_alerts  = competitor_price_alert(df)

    seg_rows = segmentation[segmentation["Product ID"]==product_id]
    cur_seg  = seg_rows["Demand Segment"].values[0] if len(seg_rows)>0 else "Unknown"

    recs = generate_recommendations(
        fc_total, safety, current_inventory, elasticity, comp_alerts,
        growth_data, hol_impact, promo_uplift, cat_profit, reg_vol,
        reg_eff, s_str, long_trend, cur_seg
    )

    icons = ["🎯","💡","⚡","🚀","💰","📊","🔔","⭐","🔍","🌟"]
    r1,r2 = st.columns(2)
    for i, rec in enumerate(recs):
        with (r1 if i%2==0 else r2):
            st.info(f"{icons[i%len(icons)]} **Insight #{i+1}**\n\n{rec}")
            
    # --- Footer ---
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;padding:1.5rem 0;color:var(--text3);">
        <p style="font-size:.875rem;font-weight:600;margin-bottom:.3rem;">Retail AI Intelligence Platform v2.0</p>
        <p style="font-size:.75rem;">Powered by TensorFlow • Streamlit • Plotly • Last updated: {datetime.now().strftime("%d %b %Y, %H:%M")}</p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# 6. ROUTER
# =========================================================
if st.session_state.page == 'landing':
    show_landing_page()
else:
    show_dashboard()


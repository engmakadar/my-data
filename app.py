import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── Page Configuration ───────────────────────────────────
st.set_page_config(
    page_title = "Walmart Sales Intelligence",
    page_icon  = "🛒",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─── Load Custom CSS ──────────────────────────────────────
def load_css():
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>',
                    unsafe_allow_html=True)

load_css()

# ─── Load Data ────────────────────────────────────────────
@st.cache_data
def load_app_data():
    from utils.data_loader import (
        load_data, load_models,
        load_results, load_shap_values,
        get_feature_cols
    )
    df       = load_data()
    models   = load_models()
    results  = load_results()
    shap_vals= load_shap_values()
    features = get_feature_cols()
    return df, models, results, shap_vals, features

df, models, results, shap_values, feature_cols = load_app_data()

# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/"
             "commons/c/ca/Walmart_logo.svg", width=160)
    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    api_key = st.text_input(
        "🔑 Claude API Key",
        type     = "password",
        help     = "Get from console.anthropic.com",
        value    = st.session_state.get('api_key', '')
    )
    if api_key:
        st.session_state['api_key'] = api_key
        st.success("✅ API Key saved!")

    st.markdown("---")
    selected_store = st.selectbox(
        "🏪 Select Store",
        options = sorted(df['Store'].unique()),
        index   = 0
    )

    date_range = st.date_input(
        "📅 Date Range",
        value = [df['Date'].min(), df['Date'].max()],
        min_value = df['Date'].min().date(),
        max_value = df['Date'].max().date()
    )

    st.markdown("---")
    st.markdown("### 📊 Navigation")
    st.markdown("""
    - 🏠 **Home** — Overview
    - 📊 **EDA** — Data Analysis  
    - 🤖 **Models** — Forecasting
    - 🔍 **SHAP** — Explainability
    - 💡 **AI Advisor** — Recommendations
    """)

# ─── HOMEPAGE ─────────────────────────────────────────────
st.title("🛒 Walmart AI-Powered Sales Intelligence Platform")
st.markdown("*Forecasting · Explainability · AI Recommendations*")
st.markdown("---")

# ─── KPI Metrics Row ──────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_sales  = df['Weekly_Sales'].sum()
avg_sales    = df['Weekly_Sales'].mean()
total_stores = df['Store'].nunique()
holiday_lift = (
    df[df['Holiday_Flag']==1]['Weekly_Sales'].mean() /
    df[df['Holiday_Flag']==0]['Weekly_Sales'].mean() - 1
) * 100
best_store   = df.groupby('Store')['Weekly_Sales'].mean().idxmax()

with col1:
    st.metric("💰 Total Sales",
              f"${total_sales/1e9:.2f}B",
              delta="All Time")
with col2:
    st.metric("📊 Avg Weekly Sales",
              f"${avg_sales:,.0f}",
              delta="Per Store")
with col3:
    st.metric("🏪 Total Stores",
              f"{total_stores}",
              delta="Nationwide")
with col4:
    st.metric("🎉 Holiday Lift",
              f"+{holiday_lift:.1f}%",
              delta="vs Non-Holiday")
with col5:
    st.metric("🏆 Best Store",
              f"Store #{best_store}",
              delta="Highest Avg Sales")

st.markdown("---")

# ─── Sales Overview Chart ─────────────────────────────────
st.subheader("📈 Total Weekly Sales Trend")

weekly_total = df.groupby('Date')['Weekly_Sales'].sum().reset_index()

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x    = weekly_total['Date'],
    y    = weekly_total['Weekly_Sales'],
    mode = 'lines',
    name = 'Weekly Sales',
    line = dict(color='#3b82f6', width=2),
    fill = 'tozeroy',
    fillcolor = 'rgba(59, 130, 246, 0.1)'
))

# Add rolling average
rolling_avg = weekly_total['Weekly_Sales'].rolling(8).mean()
fig_trend.add_trace(go.Scatter(
    x    = weekly_total['Date'],
    y    = rolling_avg,
    mode = 'lines',
    name = '8-Week Rolling Avg',
    line = dict(color='#f59e0b', width=2, dash='dash')
))

fig_trend.update_layout(
    template    = 'plotly_dark',
    height      = 350,
    showlegend  = True,
    xaxis_title = 'Date',
    yaxis_title = 'Total Weekly Sales ($)',
    hovermode   = 'x unified',
    margin      = dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig_trend, use_container_width=True)

# ─── Model Performance Summary ────────────────────────────
st.markdown("---")
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("🤖 Model Performance Leaderboard")
    results_display = results.copy()
    results_display['RMSE']  = results_display['RMSE'].apply(
        lambda x: f"${x:,.2f}"
    )
    results_display['MAE']   = results_display['MAE'].apply(
        lambda x: f"${x:,.2f}"
    )
    results_display['R2']    = results_display['R2'].apply(
        lambda x: f"{x*100:.2f}%"
    )
    results_display['MAPE']  = results_display['MAPE'].apply(
        lambda x: f"{x:.2f}%"
    )
    st.dataframe(
        results_display.set_index('Model'),
        use_container_width = True,
        height              = 220
    )

with col_r:
    st.subheader("🏪 Store Performance Ranking")
    store_rank = df.groupby('Store')['Weekly_Sales'].agg(
        ['mean','std','sum']
    ).round(2).sort_values('mean', ascending=False).head(10)
    store_rank.columns = ['Avg Weekly ($)', 'Std Dev ($)', 'Total ($)']
    store_rank['Avg Weekly ($)'] = store_rank['Avg Weekly ($)'].apply(
        lambda x: f"${x:,.0f}"
    )
    st.dataframe(store_rank, use_container_width=True, height=220)

# ─── Footer ───────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#718096; font-size:0.85rem'>
    🛒 Walmart AI Sales Intelligence Platform &nbsp;|&nbsp;
    Built with Streamlit + Claude AI &nbsp;|&nbsp;
    Models: XGBoost · Random Forest · LSTM · ARIMA · SARIMA
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_data

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("📊 Exploratory Data Analysis")
st.markdown("Deep dive into Walmart sales data patterns and behavior")
st.markdown("---")

df = load_data()

# ─── Filters ──────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    stores = st.multiselect(
        "🏪 Filter Stores",
        options = sorted(df['Store'].unique()),
        default = sorted(df['Store'].unique())[:10]
    )
with col2:
    years = st.multiselect(
        "📅 Filter Years",
        options = sorted(df['Year'].unique()),
        default = sorted(df['Year'].unique())
    )
with col3:
    holiday_filter = st.radio(
        "🎉 Holiday Filter",
        options = ["All", "Holiday Only", "Non-Holiday Only"],
        horizontal = True
    )

# ─── Apply Filters ────────────────────────────────────────
df_f = df[df['Store'].isin(stores) & df['Year'].isin(years)]
if holiday_filter == "Holiday Only":
    df_f = df_f[df_f['Holiday_Flag'] == 1]
elif holiday_filter == "Non-Holiday Only":
    df_f = df_f[df_f['Holiday_Flag'] == 0]

# ─── KPIs ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records",    f"{len(df_f):,}")
k2.metric("Avg Sales",  f"${df_f['Weekly_Sales'].mean():,.0f}")
k3.metric("Max Sales",  f"${df_f['Weekly_Sales'].max():,.0f}")
k4.metric("Min Sales",  f"${df_f['Weekly_Sales'].min():,.0f}")
st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trends", "🏪 Stores",
    "🎉 Holidays", "🌡️ External Factors", "🔥 Correlation"
])

with tab1:
    # Monthly Seasonality
    monthly = df_f.groupby('Month')['Weekly_Sales'].mean().reset_index()
    months  = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
    monthly['Month_Name'] = monthly['Month'].apply(lambda x: months[x-1])

    fig_month = px.bar(
        monthly, x='Month_Name', y='Weekly_Sales',
        title    = 'Average Weekly Sales by Month (Seasonality)',
        color    = 'Weekly_Sales',
        color_continuous_scale = 'Blues',
        template = 'plotly_dark'
    )
    st.plotly_chart(fig_month, use_container_width=True)

    # Year-over-Year
    yearly = df_f.groupby(['Year','Month'])['Weekly_Sales'].mean().reset_index()
    fig_yoy = px.line(
        yearly, x='Month', y='Weekly_Sales', color='Year',
        title    = 'Year-over-Year Sales Comparison',
        template = 'plotly_dark',
        markers  = True
    )
    fig_yoy.update_xaxes(
        tickvals = list(range(1,13)),
        ticktext = months
    )
    st.plotly_chart(fig_yoy, use_container_width=True)

with tab2:
    col_l, col_r = st.columns(2)
    with col_l:
        store_avg = df_f.groupby('Store')['Weekly_Sales'].mean().sort_values(
            ascending=False
        ).reset_index()
        fig_store = px.bar(
            store_avg, x='Store', y='Weekly_Sales',
            title    = 'Average Sales by Store',
            color    = 'Weekly_Sales',
            color_continuous_scale = 'Viridis',
            template = 'plotly_dark'
        )
        st.plotly_chart(fig_store, use_container_width=True)

    with col_r:
        store_box = px.box(
            df_f, x='Store', y='Weekly_Sales',
            title    = 'Sales Distribution by Store',
            template = 'plotly_dark',
            color_discrete_sequence = ['#3b82f6']
        )
        st.plotly_chart(store_box, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        holiday_avg = df_f.groupby('Holiday_Flag')['Weekly_Sales'].mean()
        fig_h = px.bar(
            x     = ['Non-Holiday', 'Holiday'],
            y     = holiday_avg.values,
            title = 'Avg Sales: Holiday vs Non-Holiday',
            color = ['Non-Holiday', 'Holiday'],
            color_discrete_map = {
                'Non-Holiday': '#3b82f6',
                'Holiday'    : '#f59e0b'
            },
            template = 'plotly_dark'
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with c2:
        holiday_month = df_f[df_f['Holiday_Flag']==1].groupby(
            'Month'
        )['Weekly_Sales'].mean().reset_index()
        holiday_month['Month_Name'] = holiday_month['Month'].apply(
            lambda x: months[x-1]
        )
        fig_hm = px.bar(
            holiday_month,
            x        = 'Month_Name',
            y        = 'Weekly_Sales',
            title    = 'Holiday Sales by Month',
            color    = 'Weekly_Sales',
            color_continuous_scale = 'Oranges',
            template = 'plotly_dark'
        )
        st.plotly_chart(fig_hm, use_container_width=True)

with tab4:
    ext_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for i in range(0, len(ext_features), 2):
        c1, c2 = st.columns(2)
        for col, feat in zip([c1, c2], ext_features[i:i+2]):
            with col:
                corr = df_f[feat].corr(df_f['Weekly_Sales'])
                fig_scatter = px.scatter(
                    df_f.sample(min(2000, len(df_f))),
                    x        = feat,
                    y        = 'Weekly_Sales',
                    trendline= 'ols',
                    title    = f'{feat} vs Sales (r={corr:.3f})',
                    opacity  = 0.4,
                    template = 'plotly_dark',
                    color_discrete_sequence = ['#3b82f6']
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

with tab5:
    import plotly.figure_factory as ff
    num_cols = ['Weekly_Sales','Temperature','Fuel_Price',
                'CPI','Unemployment','Holiday_Flag',
                'Month','Week','Year']
    corr_matrix = df_f[num_cols].corr().round(3)
    fig_heatmap = px.imshow(
        corr_matrix,
        title        = 'Feature Correlation Heatmap',
        color_continuous_scale = 'RdBu',
        aspect       = 'auto',
        template     = 'plotly_dark',
        text_auto    = True
    )
    fig_heatmap.update_layout(height=550)
    st.plotly_chart(fig_heatmap, use_container_width=True)
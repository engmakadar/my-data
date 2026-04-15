import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.data_loader import (
    load_data, load_models,
    load_results, get_feature_cols
)

st.set_page_config(
    page_title="Models", page_icon="🤖", layout="wide"
)
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',
                unsafe_allow_html=True)

st.title("🤖 Forecasting Models")
st.markdown("Compare all 5 models and generate store-level forecasts")
st.markdown("---")

df      = load_data()
models  = load_models()
results = load_results()
feature_cols = get_feature_cols()

# ─── Model Comparison ─────────────────────────────────────
st.subheader("📊 Model Performance Comparison")

tab1, tab2, tab3 = st.tabs([
    "📋 Leaderboard", "📊 Visual Comparison", "🔮 Store Forecast"
])

with tab1:
    st.dataframe(
        results.set_index('Model').style.highlight_min(
            subset=['RMSE','MAE','MAPE'], color='#1a472a'
        ).highlight_max(
            subset=['R2'], color='#1a472a'
        ).format({
            'RMSE' : '${:,.2f}',
            'MAE'  : '${:,.2f}',
            'MAPE' : '{:.2f}%',
            'R2'   : '{:.4f}'
        }),
        use_container_width=True,
        height=250
    )

    best_model = results.loc[results['RMSE'].idxmin(), 'Model']
    best_r2    = results.loc[results['RMSE'].idxmin(), 'R2']
    best_rmse  = results.loc[results['RMSE'].idxmin(), 'RMSE']
    best_mape  = results.loc[results['RMSE'].idxmin(), 'MAPE']

    st.markdown(f"""
    <div class='success-box'>
        🏆 <b>Best Model: {best_model}</b><br>
        R² Score: {best_r2*100:.2f}% accuracy &nbsp;|&nbsp;
        RMSE: ${best_rmse:,.2f} &nbsp;|&nbsp;
        MAPE: {best_mape:.2f}%
    </div>
    """, unsafe_allow_html=True)

with tab2:
    metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
    fig = go.Figure()
    colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6']

    for i, row in results.iterrows():
        fig.add_trace(go.Bar(
            name = row['Model'],
            x    = metrics,
            y    = [row['RMSE'], row['MAE'],
                    row['R2']*100, row['MAPE']],
            marker_color = colors[i % len(colors)]
        ))

    fig.update_layout(
        barmode   = 'group',
        template  = 'plotly_dark',
        height    = 400,
        title     = 'All Models — Metric Comparison',
        yaxis_title = 'Score',
        legend    = dict(orientation='h', y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🔮 Generate Store Forecast")
    c1, c2, c3 = st.columns(3)
    with c1:
        store_sel  = st.selectbox(
            "Select Store",
            options = sorted(df['Store'].unique())
        )
    with c2:
        weeks_ahead = st.slider(
            "Weeks to Forecast", 1, 12, 4
        )
    with c3:
        model_sel  = st.selectbox(
            "Select Model",
            options = ['XGBoost', 'Random Forest']
        )

    if st.button("🚀 Generate Forecast"):
        with st.spinner("Generating forecast..."):

            store_data = df[df['Store']==store_sel].copy()
            model = models['xgboost'] if model_sel=='XGBoost' \
                    else models['random_forest']

            last_row    = store_data[feature_cols].iloc[-1:]
            last_actual = store_data['Weekly_Sales'].iloc[-1]

            preds = []
            current = last_row.copy()

            for w in range(weeks_ahead):
                current['Sales_Lag_1'] = last_actual if w==0 \
                                         else preds[-1]
                current['Week'] = (
                    current['Week'].values[0] + w
                ) % 52 + 1
                p = model.predict(current)[0]
                preds.append(p)

            weeks_labels = [f"Week {i+1}" for i in range(weeks_ahead)]
            colors_bar   = [
                '#10b981' if p > last_actual else '#ef4444'
                for p in preds
            ]

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Bar(
                x              = weeks_labels,
                y              = preds,
                marker_color   = colors_bar,
                name           = 'Forecast',
                text           = [f'${p:,.0f}' for p in preds],
                textposition   = 'outside'
            ))
            fig_fc.add_hline(
                y         = last_actual,
                line_dash = 'dash',
                line_color= '#f59e0b',
                annotation_text = f"Current: ${last_actual:,.0f}"
            )
            fig_fc.update_layout(
                title    = f'Store {store_sel} — {weeks_ahead}-Week Forecast ({model_sel})',
                template = 'plotly_dark',
                height   = 400,
                yaxis_title = 'Predicted Sales ($)'
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Summary table
            fc_df = pd.DataFrame({
                'Week'             : weeks_labels,
                'Predicted Sales'  : [f"${p:,.2f}" for p in preds],
                'vs Current'       : [
                    f"{((p-last_actual)/last_actual*100):+.2f}%"
                    for p in preds
                ],
                'Direction'        : [
                    "📈 UP" if p > last_actual else "📉 DOWN"
                    for p in preds
                ]
            })
            st.dataframe(fc_df, use_container_width=True)
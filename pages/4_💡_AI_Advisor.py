import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import (
    load_data, load_models,
    load_shap_values, get_feature_cols
)
from utils.ai_utils import (
    get_ai_recommendation,
    get_store_executive_summary
)
import shap
import joblib

st.set_page_config(
    page_title="AI Advisor", page_icon="💡", layout="wide"
)
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',
                unsafe_allow_html=True)

st.title("💡 AI Sales Advisor")
st.markdown(
    "Powered by Claude AI — Get actionable recommendations "
    "based on your forecast data"
)
st.markdown("---")

# ─── API Key Check ────────────────────────────────────────
api_key = st.session_state.get('api_key', '')
if not api_key:
    st.warning("""
    ⚠️ Please enter your Claude API key in the sidebar 
    (Settings → Claude API Key) to use the AI Advisor.
    
    Get your key at: https://console.anthropic.com
    """)
    st.stop()

df           = load_data()
models       = load_models()
shap_values  = load_shap_values()
feature_cols = get_feature_cols()

# ─── Rebuild explainer ────────────────────────────────────
@st.cache_resource
def get_explainer():
    xgb = load_models()['xgboost']
    return shap.TreeExplainer(xgb)

explainer = get_explainer()

# ─── Tabs ─────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🎯 Single Prediction Advisor",
    "🏪 Store Executive Report"
])

with tab1:
    st.subheader("🎯 Get AI Recommendation for Any Prediction")

    df_model = df.dropna().reset_index(drop=True)
    split    = int(len(df_model) * 0.80)
    X_test   = df_model[feature_cols].iloc[split:].reset_index(drop=True)
    y_test   = df_model['Weekly_Sales'].iloc[split:].reset_index(drop=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        pred_idx = st.number_input(
            "Prediction Index",
            min_value = 0,
            max_value = min(len(X_test)-1, len(shap_values)-1),
            value     = 0
        )
        pred_val = models['xgboost'].predict(
            X_test.iloc[[pred_idx]]
        )[0]
        act_val  = y_test.iloc[pred_idx]
        baseline = float(explainer.expected_value)
        pct_chg  = ((pred_val - baseline) / baseline) * 100

        st.metric("Predicted", f"${pred_val:,.2f}")
        st.metric("Actual",    f"${act_val:,.2f}")
        st.metric("Change",    f"{pct_chg:+.2f}%",
                  delta="vs baseline")

        generate_btn = st.button(
            "🤖 Generate AI Recommendation",
            use_container_width=True
        )

    with c2:
        if generate_btn:
            with st.spinner("🤖 Claude is analyzing your forecast..."):

                shap_single  = shap_values[pred_idx]
                contrib_df   = pd.DataFrame({
                    'Feature' : feature_cols,
                    'SHAP'    : shap_single,
                    'Value'   : X_test.iloc[pred_idx].values
                }).sort_values('SHAP', key=abs, ascending=False)

                insights = {
                    'predicted_sales'  : float(pred_val),
                    'actual_sales'     : float(act_val),
                    'baseline_sales'   : baseline,
                    'pct_change'       : pct_chg,
                    'direction'        : "INCREASE" if pred_val > baseline
                                         else "DECREASE",
                    'top_positive_drivers': [
                        {
                            'feature': r['Feature'],
                            'value'  : float(r['Value']),
                            'impact' : float(r['SHAP'])
                        }
                        for _, r in contrib_df[
                            contrib_df['SHAP']>0
                        ].head(5).iterrows()
                    ],
                    'top_negative_drivers': [
                        {
                            'feature': r['Feature'],
                            'value'  : float(r['Value']),
                            'impact' : float(r['SHAP'])
                        }
                        for _, r in contrib_df[
                            contrib_df['SHAP']<0
                        ].head(5).iterrows()
                    ]
                }

                recommendation = get_ai_recommendation(
                    insights, api_key,
                    store_number = int(X_test.iloc[pred_idx]['Store'])
                )

                st.markdown(f"""
                <div class='ai-recommendation'>
                {recommendation.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.subheader("🏪 Store Executive Report")

    c1, c2, c3 = st.columns(3)
    with c1:
        exec_store = st.selectbox(
            "Select Store",
            options = sorted(df['Store'].unique()),
            key     = "exec_store"
        )
    with c2:
        exec_weeks = st.slider(
            "Forecast Weeks", 1, 8, 4,
            key = "exec_weeks"
        )
    with c3:
        st.write("")
        st.write("")
        run_exec = st.button(
            "📋 Generate Executive Report",
            use_container_width=True
        )

    if run_exec:
        with st.spinner("Generating executive report..."):

            store_data  = df[df['Store']==exec_store].copy()
            last_row    = store_data[feature_cols].iloc[-1:]
            last_actual = store_data['Weekly_Sales'].iloc[-1]

            preds   = []
            current = last_row.copy()
            for w in range(exec_weeks):
                current['Sales_Lag_1'] = last_actual if w==0 \
                                         else preds[-1]
                current['Week'] = (
                    current['Week'].values[0] + w
                ) % 52 + 1
                p = models['xgboost'].predict(current)[0]
                preds.append(p)

            # ── Forecast Chart ────────────────────────
            week_labels = [f"Week {i+1}" for i in range(exec_weeks)]
            fig_exec = go.Figure()
            fig_exec.add_trace(go.Scatter(
                x    = week_labels,
                y    = preds,
                mode = 'lines+markers',
                name = 'Forecast',
                line = dict(color='#3b82f6', width=3),
                marker = dict(size=10)
            ))
            fig_exec.add_hline(
                y         = last_actual,
                line_dash = 'dash',
                line_color= '#f59e0b',
                annotation_text = f"Current: ${last_actual:,.0f}"
            )
            fig_exec.update_layout(
                title    = f'Store {exec_store} — Executive Forecast',
                template = 'plotly_dark',
                height   = 350,
                yaxis_title = 'Predicted Sales ($)'
            )
            st.plotly_chart(fig_exec, use_container_width=True)

            # ── AI Executive Summary ───────────────────
            st.subheader("🤖 AI Executive Summary")
            with st.spinner("Claude is writing your report..."):
                exec_summary = get_store_executive_summary(
                    store_number = exec_store,
                    forecasts    = preds,
                    last_actual  = last_actual,
                    api_key      = api_key
                )
                st.markdown(f"""
                <div class='ai-recommendation'>
                {exec_summary.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

            # ── Forecast Table ────────────────────────
            fc_df = pd.DataFrame({
                'Week'           : week_labels,
                'Predicted ($)'  : [f"${p:,.2f}" for p in preds],
                'Change'         : [
                    f"{((p-last_actual)/last_actual*100):+.2f}%"
                    for p in preds
                ],
                'Signal'         : [
                    "📈 BUY MORE STOCK" if p > last_actual * 1.05
                    else "📉 REDUCE STOCK" if p < last_actual * 0.95
                    else "⚖️ MAINTAIN"
                    for p in preds
                ]
            })
            st.dataframe(fc_df, use_container_width=True)
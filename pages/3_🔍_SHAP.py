import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from utils.data_loader import (
    load_data, load_models,
    load_shap_values, get_feature_cols
)

st.set_page_config(
    page_title="SHAP", page_icon="🔍", layout="wide"
)
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',
                unsafe_allow_html=True)

st.title("🔍 SHAP Explainability")
st.markdown("Understand *why* the model makes each prediction")
st.markdown("---")

df           = load_data()
models       = load_models()
shap_values  = load_shap_values()
feature_cols = get_feature_cols()

tab1, tab2, tab3 = st.tabs([
    "🌍 Global Importance",
    "🎯 Single Prediction",
    "📊 Feature Interactions"
])

with tab1:
    st.subheader("🌍 Global Feature Importance (SHAP)")
    st.markdown("Which features matter most across ALL predictions?")

    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df   = pd.DataFrame({
        'Feature'   : feature_cols,
        'Mean_SHAP' : mean_shap
    }).sort_values('Mean_SHAP', ascending=False)

    fig_bar = px.bar(
        shap_df.head(20),
        x        = 'Mean_SHAP',
        y        = 'Feature',
        orientation = 'h',
        title    = 'Top 20 Features — Mean SHAP Value',
        color    = 'Mean_SHAP',
        color_continuous_scale = 'Blues',
        template = 'plotly_dark'
    )
    fig_bar.update_layout(
        height   = 550,
        yaxis    = dict(autorange='reversed')
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.info("""
    💡 **How to read this chart:**
    Higher SHAP value = feature has MORE impact on predictions.
    These are the features your model relies on most to forecast sales.
    """)

with tab2:
    st.subheader("🎯 Explain a Single Prediction")

    sample_idx = st.slider(
        "Select prediction to explain",
        0, min(len(shap_values)-1, 200), 0
    )

    df_model = df.dropna().reset_index(drop=True)
    split    = int(len(df_model) * 0.80)
    X_test   = df_model[feature_cols].iloc[split:].reset_index(drop=True)
    y_test   = df_model['Weekly_Sales'].iloc[split:].reset_index(drop=True)

    if sample_idx < len(X_test) and sample_idx < len(shap_values):
        sample_shap   = shap_values[sample_idx]
        actual_val    = y_test.iloc[sample_idx]
        predicted_val = models['xgboost'].predict(
            X_test.iloc[[sample_idx]]
        )[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Actual Sales",    f"${actual_val:,.2f}")
        c2.metric("Predicted Sales", f"${predicted_val:,.2f}")
        c3.metric("Error",
                  f"${abs(actual_val-predicted_val):,.2f}",
                  delta=f"{((predicted_val-actual_val)/actual_val*100):+.2f}%")

        contrib_df = pd.DataFrame({
            'Feature' : feature_cols,
            'SHAP'    : sample_shap,
            'Value'   : X_test.iloc[sample_idx].values
        }).sort_values('SHAP', key=abs, ascending=False).head(15)

        fig_wf = px.bar(
            contrib_df,
            x        = 'SHAP',
            y        = 'Feature',
            orientation = 'h',
            color    = 'SHAP',
            color_continuous_scale = 'RdBu',
            color_continuous_midpoint = 0,
            title    = f'SHAP Waterfall — Prediction #{sample_idx}',
            template = 'plotly_dark',
            text     = contrib_df['Value'].round(2).astype(str)
        )
        fig_wf.update_layout(
            height = 500,
            yaxis  = dict(autorange='reversed')
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        pos_feat = contrib_df[contrib_df['SHAP']>0]['Feature'].tolist()[:3]
        neg_feat = contrib_df[contrib_df['SHAP']<0]['Feature'].tolist()[:3]

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown(f"""
            <div class='success-box'>
            ✅ <b>Top factors INCREASING this prediction:</b><br>
            {'<br>'.join([f'• {f}' for f in pos_feat])}
            </div>""", unsafe_allow_html=True)
        with col_neg:
            st.markdown(f"""
            <div class='warning-box'>
            ⚠️ <b>Top factors DECREASING this prediction:</b><br>
            {'<br>'.join([f'• {f}' for f in neg_feat])}
            </div>""", unsafe_allow_html=True)

with tab3:
    st.subheader("📊 Feature Interactions")

    feat_select = st.selectbox(
        "Select feature to analyze",
        options = feature_cols
    )
    feat_idx = feature_cols.index(feat_select)

    df_model   = df.dropna().reset_index(drop=True)
    split      = int(len(df_model) * 0.80)
    X_test     = df_model[feature_cols].iloc[split:].reset_index(drop=True)

    scatter_df = pd.DataFrame({
        'Feature_Value' : X_test[feat_select].values,
        'SHAP_Value'    : shap_values[:len(X_test), feat_idx],
        'Sales'         : df_model['Weekly_Sales'].iloc[split:].values
    })

    fig_dep = px.scatter(
        scatter_df,
        x       = 'Feature_Value',
        y       = 'SHAP_Value',
        color   = 'Sales',
        title   = f'SHAP Dependence: {feat_select}',
        template= 'plotly_dark',
        color_continuous_scale = 'RdBu',
        opacity = 0.6,
        trendline = 'lowess'
    )
    fig_dep.add_hline(
        y=0, line_dash='dash',
        line_color='white', opacity=0.5
    )
    fig_dep.update_layout(
        height      = 450,
        xaxis_title = feat_select,
        yaxis_title = 'SHAP Value (Impact on Prediction)'
    )
    st.plotly_chart(fig_dep, use_container_width=True)

    st.info(f"""
    💡 **Reading this chart for {feat_select}:**
    Points above zero = this feature VALUE is pushing sales prediction UP.
    Points below zero = this feature VALUE is pushing sales prediction DOWN.
    """)

import pandas as pd
import numpy as np
import joblib
import os

def load_data():
    """Load and return cleaned dataset"""
    df = pd.read_csv('saved_outputs/cleaned_data.csv',
                     parse_dates=['Date'])
    return df

def load_models():
    """Load all saved models"""
    models = {}
    try:
        models['xgboost'] = joblib.load(
            'saved_models/xgboost_model.pkl'
        )
        models['random_forest'] = joblib.load(
            'saved_models/random_forest_model.pkl'
        )
        models['scaler'] = joblib.load(
            'saved_models/lstm_scaler.pkl'
        )
        from tensorflow.keras.models import load_model
        models['lstm'] = load_model(
            'saved_models/lstm_model.keras'
        )
    except Exception as e:
        print(f"Warning loading models: {e}")
    return models

def load_results():
    """Load model comparison results"""
    return pd.read_csv('saved_outputs/model_comparison.csv')

def load_shap_values():
    """Load saved SHAP values"""
    return np.load(
        'saved_outputs/shap_values.npy',
        allow_pickle=True
    )

def get_feature_cols():
    """Return feature column names"""
    return [
        'Year', 'Month', 'Week', 'Quarter', 'DayOfYear',
        'Is_Month_Start', 'Is_Month_End',
        'Season_Encoded', 'Is_Holiday_Season',
        'Store', 'Holiday_Flag',
        'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_4',
        'Sales_Lag_8', 'Sales_Lag_52',
        'Sales_RollMean_4', 'Sales_RollMean_8',
        'Sales_RollMean_12', 'Sales_RollStd_4',
        'Sales_Momentum',
        'Store_Mean_Sales', 'Store_Std_Sales',
        'Store_Rank', 'Sales_vs_Store_Avg'
    ]
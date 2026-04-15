import numpy as np
import pandas as pd
import shap
import joblib


def get_xgb_prediction(model, feature_data: pd.DataFrame) -> float:
    """
    Make a single prediction using XGBoost model.

    HOW IT WORKS:
    - Takes a single row of feature data (one week's worth of data)
    - Passes it through the trained XGBoost model
    - Returns a single dollar value prediction

    Args:
        model        : Trained XGBoost model (loaded from .pkl)
        feature_data : One row DataFrame with all feature columns

    Returns:
        float: Predicted weekly sales in dollars
    """
    prediction = model.predict(feature_data)
    return float(prediction[0])


def get_rf_prediction(model, feature_data: pd.DataFrame) -> float:
    """
    Make a single prediction using Random Forest model.

    HOW IT WORKS:
    - Random Forest averages predictions from 300 decision trees
    - Each tree was trained on a random subset of data
    - The average of all trees is returned as final prediction

    Args:
        model        : Trained Random Forest model
        feature_data : One row DataFrame with all feature columns

    Returns:
        float: Predicted weekly sales in dollars
    """
    prediction = model.predict(feature_data)
    return float(prediction[0])


def get_lstm_prediction(
        model,
        scaler,
        last_sequence: np.ndarray) -> float:
    """
    Make a single prediction using LSTM model.

    HOW IT WORKS:
    - LSTM expects a sequence of past N weeks (lookback window)
    - The sequence must be SCALED (0-1 range) before input
    - After prediction, we inverse-transform back to dollar value
    - Shape required: (1, LOOKBACK, 1)

    Args:
        model         : Trained LSTM Keras model
        scaler        : MinMaxScaler fitted on training data
        last_sequence : Array of last N scaled sales values

    Returns:
        float: Predicted weekly sales in dollars
    """
    # Reshape to (1 sample, N timesteps, 1 feature)
    seq   = last_sequence.reshape(1, len(last_sequence), 1)
    pred  = model.predict(seq, verbose=0)
    value = scaler.inverse_transform(pred)[0, 0]
    return float(value)


def recursive_forecast(
        model,
        last_row      : pd.DataFrame,
        feature_cols  : list,
        last_actual   : float,
        weeks         : int = 4) -> list:
    """
    Generate multi-week forecast by feeding each prediction
    back as input for the next week (recursive/rolling forecast).

    HOW IT WORKS:
    Week 1: Use real last week's sales as Sales_Lag_1 → predict Week 1
    Week 2: Use Week 1 prediction as Sales_Lag_1  → predict Week 2
    Week 3: Use Week 2 prediction as Sales_Lag_1  → predict Week 3
    ...and so on.

    This is called RECURSIVE FORECASTING — each prediction
    becomes the input for the next one.

    Args:
        model        : Trained ML model (XGBoost or RF)
        last_row     : Last known feature row from dataset
        feature_cols : List of feature column names
        last_actual  : Last known actual sales value
        weeks        : Number of weeks to forecast ahead

    Returns:
        list: Predicted sales values for each week
    """
    predictions = []
    current_row = last_row.copy()

    for week_num in range(weeks):
        # For week 1, use real last actual sale
        # For subsequent weeks, use the previous prediction
        if week_num == 0:
            current_row['Sales_Lag_1'] = last_actual
        else:
            current_row['Sales_Lag_1'] = predictions[-1]

        # Roll the week number forward (wrap at 52)
        current_row['Week'] = (
            current_row['Week'].values[0] + week_num
        ) % 52 + 1

        # Update month based on new week
        current_row['Month'] = (
            (int(current_row['Week'].values[0]) - 1) // 4
        ) + 1

        # Make prediction for this week
        pred = model.predict(current_row[feature_cols])[0]
        predictions.append(float(pred))

    return predictions


def compute_shap_for_sample(
        explainer,
        sample_row   : pd.DataFrame,
        feature_cols : list) -> dict:
    """
    Compute SHAP values for a single prediction and return
    structured positive/negative driver breakdown.

    HOW SHAP WORKS:
    SHAP (SHapley Additive exPlanations) answers the question:
    "How much did each feature contribute to THIS specific prediction?"

    It uses game theory — each feature is like a player in a game,
    and SHAP calculates each player's fair contribution to the outcome.

    Positive SHAP = feature pushed prediction HIGHER than baseline
    Negative SHAP = feature pushed prediction LOWER  than baseline

    Args:
        explainer    : SHAP TreeExplainer fitted on model
        sample_row   : Single row of feature data to explain
        feature_cols : List of feature names

    Returns:
        dict: {
            'positive_drivers': top features pushing prediction up,
            'negative_drivers': top features pushing prediction down,
            'baseline'        : model's average prediction,
            'shap_values'     : raw array of SHAP values
        }
    """
    shap_vals = explainer.shap_values(sample_row)

    contrib_df = pd.DataFrame({
        'Feature'    : feature_cols,
        'SHAP_Value' : shap_vals[0],
        'Abs_SHAP'   : np.abs(shap_vals[0]),
        'Feature_Val': sample_row.values[0]
    }).sort_values('Abs_SHAP', ascending=False)

    positive = contrib_df[contrib_df['SHAP_Value'] > 0].head(5)
    negative = contrib_df[contrib_df['SHAP_Value'] < 0].head(5)

    return {
        'positive_drivers': positive.to_dict('records'),
        'negative_drivers': negative.to_dict('records'),
        'baseline'        : float(explainer.expected_value),
        'shap_values'     : shap_vals[0],
        'contrib_df'      : contrib_df
    }


def evaluate_predictions(
        y_true : np.ndarray,
        y_pred : np.ndarray,
        model_name: str = "Model") -> dict:
    """
    Compute all evaluation metrics for a model.

    METRICS EXPLAINED:
    ─────────────────────────────────────────────────────────
    MAE  (Mean Absolute Error):
         Average dollar amount the model is wrong by.
         Easy to understand — if MAE=$500, model is off by
         $500 on average per week.

    RMSE (Root Mean Squared Error):
         Similar to MAE but PENALIZES large errors more.
         If the model occasionally makes huge mistakes,
         RMSE will be much larger than MAE.
         Lower is better.

    MAPE (Mean Absolute Percentage Error):
         Error as a percentage — model-size independent.
         MAPE=5% means predictions are 5% off on average.
         Useful for comparing across different scales.

    R²   (R-Squared / Coefficient of Determination):
         How much variance the model explains.
         R²=0.95 means model explains 95% of sales variation.
         Closer to 1.0 = better.
         Negative R² = model is WORSE than just predicting mean!
    ─────────────────────────────────────────────────────────

    Args:
        y_true     : Array of actual values
        y_pred     : Array of predicted values
        model_name : Name label for display

    Returns:
        dict: All metrics
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"📊 {model_name} — Results")
    print(f"{'='*50}")
    print(f"  MAE  : ${mae:,.2f}   ← avg $ error per week")
    print(f"  RMSE : ${rmse:,.2f}   ← penalizes big errors")
    print(f"  MAPE : {mape:.2f}%    ← % error on average")
    print(f"  R²   : {r2:.4f}      ← 1.0 = perfect")

    return {
        'Model': model_name,
        'MAE'  : mae,
        'RMSE' : rmse,
        'MAPE' : mape,
        'R2'   : r2
    }
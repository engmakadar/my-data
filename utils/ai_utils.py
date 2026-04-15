import anthropic
import json

def get_ai_recommendation(
        insights     : dict,
        api_key      : str,
        store_number : int = None) -> str:
    """
    Send SHAP insights to Claude API
    and return business recommendation
    """
    client = anthropic.Anthropic(api_key=api_key)

    store_info = f"Store #{store_number}" if store_number else "All Stores"

    prompt = f"""
You are an expert retail sales analyst for Walmart ({store_info}).
Analyze this AI forecast data and provide actionable business recommendations.

FORECAST DATA:
- Predicted Sales    : ${insights['predicted_sales']:,.2f}
- Baseline Average   : ${insights['baseline_sales']:,.2f}
- Change from Base   : {insights['pct_change']:+.2f}%
- Direction          : {insights['direction']}

FACTORS DRIVING SALES UP:
{json.dumps(insights['top_positive_drivers'], indent=2)}

FACTORS PULLING SALES DOWN:
{json.dumps(insights['top_negative_drivers'], indent=2)}

Respond in this exact format:

## 📈 Forecast Summary
[2 sentences — what is predicted and by how much]

## 🔍 Key Business Drivers
[Bullet points — explain each driver in plain business language]

## ⚠️ Risk Factors
[Bullet points — what is pulling sales down and why it matters]

## 💡 Top 3 Recommendations
[Numbered list — specific, practical actions for the sales manager]

## 📅 Watch List
[2 key metrics to monitor in the coming weeks]

Keep it concise, professional, and actionable.
"""
    message = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 1000,
        messages   = [{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def get_store_executive_summary(
        store_number : int,
        forecasts    : list,
        last_actual  : float,
        api_key      : str) -> str:
    """Generate executive summary for store forecast"""

    client  = anthropic.Anthropic(api_key=api_key)
    weeks   = len(forecasts)
    trend   = ((forecasts[-1] - last_actual) / last_actual) * 100

    prompt = f"""
You are a senior retail analyst. Provide an executive summary for 
Walmart Store #{store_number} based on the following {weeks}-week forecast:

Current Sales : ${last_actual:,.2f}
Forecasts     : {[f'Week {i+1}: ${v:,.0f}' for i, v in enumerate(forecasts)]}
Overall Trend : {trend:+.2f}%

Write a concise 3-paragraph executive summary:
Paragraph 1: Overall performance outlook
Paragraph 2: Key trends and patterns observed  
Paragraph 3: Three strategic recommendations

Keep it under 200 words. Professional tone.
"""
    message = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 500,
        messages   = [{"role": "user", "content": prompt}]
    )
    return message.content[0].text
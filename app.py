import json
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline

API_KEY = "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"  # Groq API Key

def truncate_text(text, max_length=512):
    return text[:max_length]

def clean_price_column(df):
    """Clean and convert price column to numeric."""
    if 'Price' in df.columns:
        # Convert to string, remove any non-numeric characters except decimal
        df['Price'] = df['Price'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    return df

def clean_discount_column(df):
    """Clean and convert discount column to numeric."""
    if 'Discount' in df.columns:
        df['Discount'] = df['Discount'].astype(str).str.replace('%', '').str.strip()
        df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce')
    return df

def load_competitor_data():
    """Load competitor data from a CSV file."""
    data = pd.read_csv("competitor_price_data.csv")
    # Clean the data
    data = clean_price_column(data)
    data = clean_discount_column(data)
    print(data.head())
    return data

def load_reviews_data():
    """Load reviews data from a CSV file."""
    reviews = pd.read_csv("reviews_data.csv")
    return reviews

def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline(reviews)

def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    data["Discount"] = pd.to_numeric(data["Discount"], errors='coerce')
    data["Price"] = pd.to_numeric(data["Price"], errors='coerce')
    data["Predicted_Discount"] = data["Discount"] + (data["Price"] * 0.05).round(2)

    X = data[["Price", "Discount"]]
    y = data["Predicted_Discount"]
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, train_size=0.8
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """

    if data.empty:
        st.warning("No valid discount data available for forecasting.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Ensure Discount column is numeric
    if 'Discount' in data.columns:
        data["Discount"] = pd.to_numeric(data["Discount"], errors='coerce')
        data = data.dropna(subset=["Discount"])
    
    discount_series = data["Discount"]

    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError("Index must be datetime or convertible to datetime.") from e

    # Check if discount_series is empty after processing
    if discount_series.empty:
        st.warning("No valid historical discount data for ARIMA model.")
        return pd.DataFrame()

    model = ARIMA(discount_series, order=(0, 1, 2))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=future_days)
    future_dates = pd.date_range(
        start=pd.Timestamp.today().normalize(),  # Start from today
        periods=future_days
    )

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast.round(2)})
    forecast_df.set_index("Date", inplace=True)

    return forecast_df

def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""
    date = datetime.now()
    prompt = f"""
    You are a highly skilled business strategist specializing in e-commerce. Based on the following details, suggest actionable strategies to optimize pricing, promotions, and customer satisfaction for the selected product:

1. **Product Name**: {product_name}

2. **Competitor Data** (including current prices, discounts, and predicted discounts):
{competitor_data}

3. **Sentiment Analysis**:
{sentiment}

5. **Today's Date**: {str(date)}

Provide your recommendations in a structured format:
1. **💰Pricing Strategy**
2. **🎯Promotional Campaign Ideas**
3. **⭐Customer Satisfaction Recommendations**
    """

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama-3.1-8b-instant",  # Updated model
        "temperature": 0,
    }

    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=data,
            headers=headers,
            timeout=30,
        )
        
        response_data = response.json()
        
        if response.status_code != 200:
            error_msg = response_data.get('error', {}).get('message', 'Unknown API error')
            return f"API Error {response.status_code}: {error_msg}"
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        else:
            return "No response generated. API returned unexpected format."
            
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def generate_price_recommendation(selected_product, product_data_with_predictions, sentiments):
    """Generate an optimal selling price recommendation using LLM with reduced input size."""
    
    # Reduce competitor data to the last 5 rows
    competitor_summary = product_data_with_predictions.tail(5).to_string()

    # Summarize discount trends
    if not product_data_with_predictions.empty and "Predicted_Discount" in product_data_with_predictions.columns:
        avg_predicted_discount = product_data_with_predictions["Predicted_Discount"].mean()
        discount_summary = f"Average Predicted Discount: {avg_predicted_discount:.2f}% over the next week."
    else:
        discount_summary = "No discount prediction data available."

    # Summarize sentiment analysis
    sentiment_summary = sentiments if sentiments else "No sentiment data available."

    prompt = f"""
    You are a pricing strategist for an e-commerce platform. Your goal is to determine the best selling price for **{selected_product}**.

    **Key Insights:**
    1. **Competitor Pricing & Discounts (Last 5 Entries):**  
    {competitor_summary}

    2. **Discount Trend Summary:**  
    {discount_summary}

    3. **Customer Sentiment Analysis:**  
    {sentiment_summary}

    **Task:**
    - Identify market pricing trends.
    - Analyze how predicted discounts impact sales.
    - Adjust pricing based on customer sentiment.
    - Consider profit from sellers point of view
    - Recommend an **optimal selling price in Indian Rupees (₹)** that balances **profitability,sales and customer satisfaction**.
    
    ### Output Format:
    📌 **Optimal Selling Price**: ₹ <value>  
    💡 **Reasoning**:  
    - **Competitor Pricing**: ₹X with avg discount of Y%.  
    - **Discount Trend**: Expected Z% drop, so adjusting accordingly.  
    - **Customer Sentiment**: Users prefer products priced around ₹<range>.  
    - **Final Suggestion**: ₹<value> ensures profitability while increasing sales volume.  
    """

    chat_data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama-3.1-8b-instant",
        "temperature": 0.5, 
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(chat_data),
            headers=headers,
            timeout=10,
        ).json()

        return res.get("choices", [{}])[0].get("message", {}).get("content", "No price recommendation available.")
    
    except Exception as e:
        return f"Error in price recommendation: {e}"

def chatbot_response(user_query, selected_product, product_data_with_predictions, sentiments):
    """Generate chatbot response for user queries."""
    chat_prompt = f"""
You are an expert e-commerce analyst. Based on the following product data(having indian rupees as currency), answer the user's query accordingly. 
    1. **Product Name**: {selected_product}
    2. **Competitor Data**: 
    {product_data_with_predictions.to_string() if not product_data_with_predictions.empty else "No data available"}
    3. **Discount Predictions**: 
    {product_data_with_predictions['Predicted_Discount'].to_list() if not product_data_with_predictions.empty and 'Predicted_Discount' in product_data_with_predictions.columns else "No predictions available"}
    4. **Sentiment Analysis**: 
    {sentiments if sentiments else "No sentiment data available."}
    
    ### User Question:
    {user_query}
    """
    
    chat_data = {
        "messages": [{"role": "user", "content": chat_prompt}],
        "model": "llama-3.1-8b-instant",  # Updated model
        "temperature": 0.5,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    
    try:
        chat_res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(chat_data),
            headers=headers,
            timeout=10,
        ).json()
        
        return chat_res.get("choices", [{}])[0].get("message", {}).get("content", "No response available.")
    
    except Exception as e:
        return f"Error in chatbot: {e}"


####-----------------------Frontend---------------------------##########

st.set_page_config(
    page_title="AI-Powered E-Commerce Intelligence Tool", 
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# 🎨 Modern UI Styling
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background with Light Blue */
    .stApp {
        background: #E3F2FD;
    }
    
    /* Main Container Background */
    .main .block-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #1E3A8A;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Sidebar Select Box Override */
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: black !important;
    }
    
    /* Headers - Consistent Blue Color */
    h1, h2, h3, h4 {
        color: #1E3A8A !important;
        font-weight: 800 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h2 {
        font-size: 1.8rem !important;
        margin-top: 1.5rem !important;
        border-bottom: 3px solid #1E3A8A;
        padding-bottom: 0.5rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.3rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #1E3A8A;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.12);
    }
    
    /* Buttons */
    .stButton > button {
        background: #1E3A8A;
        color: white;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: #2563EB;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(30, 58, 138, 0.3);
    }
    
    /* Tables - Modern Styling */
    .stTable {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stTable table {
        border-collapse: collapse;
        width: 100%;
    }
    
    .stTable th {
        background: #1E3A8A;
        color: white !important;
        padding: 12px !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .stTable td {
        padding: 10px !important;
        border-bottom: 1px solid #E3F2FD;
        background-color: white;
        color: #1F2937;
    }
    
    .stTable tr:hover td {
        background-color: #F0F9FF;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #F0F9FF;
        border-radius: 10px;
        font-weight: 600;
        color: #1E3A8A;
        border: 1px solid #1E3A8A;
    }
    
    /* Chat Box */
    textarea {
        border-radius: 12px !important;
        border: 2px solid #1E3A8A !important;
        transition: all 0.3s ease;
    }
    
    textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.1) !important;
    }
    
    /* Success/Info/Warning Messages */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #1E3A8A;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        color: #6b7280;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1E3A8A;
        border-bottom-color: #1E3A8A;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: #1E3A8A;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #E3F2FD;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1E3A8A;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2563EB;
    }
    
    /* Remove extra spacing from containers */
    .stMarkdown {
        margin-bottom: 0;
    }
    
    /* Ensure no empty boxes */
    .element-container {
        margin-bottom: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header with Animation
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">
            🚀 AI-Powered E-Commerce Intelligence Tool
        </h1>
    </div>
""", unsafe_allow_html=True)

# Load Data
competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

# Sidebar with Modern Design
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">🎯</div>
            <h3 style="margin-top: 0.5rem;">Dashboard Controls</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Product Selection with Icon
    st.markdown("### 📦 Product Selection")
    products = competitor_data['Product name'].unique()
    selected_product = st.selectbox("Choose a product to analyze:", products)
    
    st.markdown("---")
    
    # Quick Stats Card
    st.markdown("### 📊 Quick Stats")
    product_data_temp = competitor_data[competitor_data["Product name"] == selected_product].copy()
    
    # Ensure Price column is numeric for calculations
    product_data_temp["Price"] = pd.to_numeric(product_data_temp["Price"], errors='coerce')
    product_data_temp["Discount"] = pd.to_numeric(product_data_temp["Discount"], errors='coerce')
    
    avg_price = product_data_temp["Price"].mean() if not product_data_temp["Price"].isna().all() else 0
    avg_discount = product_data_temp["Discount"].mean() if not product_data_temp["Discount"].isna().all() else 0
    
    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 12px; margin: 0.5rem 0;">
            <div style="font-size: 0.9rem; opacity: 0.8;">Average Price</div>
            <div style="font-size: 1.5rem; font-weight: bold;">₹{avg_price:.2f}</div>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 12px; margin: 0.5rem 0;">
            <div style="font-size: 0.9rem; opacity: 0.8;">Average Discount</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{avg_discount:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chatbot Toggle with Styling
    st.markdown("### 💬 AI Assistant")
    enable_chatbot = st.checkbox("Enable Chatbot Assistant", value=False, key="enable_chatbot")

# Filter Data for Selected Product
product_data = competitor_data[competitor_data["Product name"] == selected_product].copy()
product_reviews = reviews_data[reviews_data["Product name"] == selected_product]

# Convert numeric columns
product_data["Price"] = pd.to_numeric(product_data["Price"], errors='coerce')
product_data["Discount"] = pd.to_numeric(product_data["Discount"], errors='coerce')

# Main Content Area with Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Market Analysis", "💡 Insights", "🎯 Strategy"])

with tab1:
    # Overview Section
    st.markdown(f"<h2 style='margin-top: 0; margin-bottom: 1rem;'>📊 {selected_product} Overview</h2>", unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = product_data["Price"].iloc[-1] if not product_data.empty and not pd.isna(product_data["Price"].iloc[-1]) else 0
    current_discount = product_data["Discount"].iloc[-1] if not product_data.empty and not pd.isna(product_data["Discount"].iloc[-1]) else 0
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">💰</div>
                <div style="font-size: 0.9rem; color: #6b7280;">Current Avg Price</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #1E3A8A;">₹{current_price:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">🏷️</div>
                <div style="font-size: 0.9rem; color: #6b7280;">Current Discount</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #1E3A8A;">{current_discount:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">🏪</div>
                <div style="font-size: 0.9rem; color: #6b7280;">Active Competitors</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #1E3A8A;">{product_data["Source"].nunique()}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">⭐</div>
                <div style="font-size: 0.9rem; color: #6b7280;">Reviews Analyzed</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #1E3A8A;">{len(product_reviews)}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent Data Table
    st.markdown("<h3>📋 Recent Competitor Data</h3>", unsafe_allow_html=True)
    
    # Create a display table with formatted values
    display_data = product_data.tail(5).copy()
    display_data["Price"] = display_data["Price"].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "N/A")
    display_data["Discount"] = display_data["Discount"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    
    st.table(display_data.set_index(product_data.columns[0]))

with tab2:
    # Market Analysis Section
    st.markdown("<h2 style='margin-bottom: 1rem;'>📈 Market Analysis Dashboard</h2>", unsafe_allow_html=True)
    
    product_data["Date"] = pd.to_datetime(product_data["Date"], errors='coerce')
    product_data = product_data.dropna(subset=["Date"])
    latest_date = product_data["Date"].max() if not product_data.empty else pd.Timestamp.now()
    two_months_ago = latest_date - pd.DateOffset(months=2)
    filtered_product_data = product_data[product_data["Date"] >= two_months_ago]
    
    # Price and Discount Trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>💲 Price Trends</h3>", unsafe_allow_html=True)
        fig_price = px.line(
            filtered_product_data, 
            x="Date",
            y="Price", 
            color="Source",  
            title="Price Comparison: Flipkart vs Amazon",
            markers=True, 
            line_shape="linear",
            color_discrete_map={"Flipkart": "#1E3A8A", "Amazon": "#2563EB"},
            template="plotly_white"
        )
        fig_price.update_layout(
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=16,
            title_font_family="Inter"
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        st.markdown("<h3>🏷️ Discount Trends</h3>", unsafe_allow_html=True)
        fig_discount = px.line(
            filtered_product_data, 
            x="Date",
            y="Discount", 
            color="Source",  
            title="Discount Comparison: Flipkart vs Amazon",
            markers=True, 
            line_shape="linear",
            color_discrete_map={"Flipkart": "#1E3A8A", "Amazon": "#2563EB"},
            template="plotly_white"
        )
        fig_discount.update_layout(
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=16,
            title_font_family="Inter"
        )
        st.plotly_chart(fig_discount, use_container_width=True)
    
    # Sentiment Analysis
    st.markdown("<h3>😊 Customer Sentiment Analysis</h3>", unsafe_allow_html=True)
    
    if not product_reviews.empty:
        product_reviews["reviews"] = product_reviews["reviews"].apply(
            lambda x: truncate_text(str(x), 512)
        )
        reviews = product_reviews["reviews"].tolist()
        sentiments = analyze_sentiment(reviews)
        
        sentiment_df = pd.DataFrame(sentiments)
        color_map = {"POSITIVE": "#10b981", "NEGATIVE": "#ef4444"}
        
        fig = px.bar(
            sentiment_df, 
            x="label", 
            y="score",
            title="Sentiment Distribution",
            color="label",
            color_discrete_map=color_map,
            template="plotly_white",
            text="score"
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=16,
            title_font_family="Inter",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No reviews available for this product.")
        sentiments = None
    
    # Forecasting Section
    st.markdown("<h3>🔮 Discount Forecast (Next 5 Days)</h3>", unsafe_allow_html=True)
    
    product_data_forecast = product_data.copy()
    product_data_forecast["Date"] = pd.to_datetime(product_data_forecast["Date"], errors="coerce")
    product_data_forecast = product_data_forecast.dropna(subset=["Date"])
    product_data_forecast.set_index("Date", inplace=True)
    product_data_forecast = product_data_forecast.sort_index()
    
    forecast_df = forecast_discounts_arima(product_data_forecast)
    
    if not forecast_df.empty:
        st.dataframe(
            forecast_df.style.format({"Predicted_Discount": "{:.2f}"}),
            use_container_width=True
        )
        
        # Forecast Visualization
        fig_forecast = px.line(
            forecast_df.reset_index(),
            x="Date",
            y="Predicted_Discount",
            title="Predicted Discount Trend",
            markers=True,
            template="plotly_white"
        )
        fig_forecast.update_traces(line_color="#1E3A8A", marker_color="#2563EB")
        fig_forecast.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=16
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("Insufficient data for discount forecasting.")

with tab3:
    # Insights Section
    st.markdown("<h2 style='margin-bottom: 1rem;'>💡 AI-Powered Insights</h2>", unsafe_allow_html=True)
    
    # Prepare Data for Recommendations
    product_data_forecast_tab3 = product_data.copy()
    product_data_forecast_tab3["Date"] = pd.to_datetime(product_data_forecast_tab3["Date"], errors="coerce")
    product_data_forecast_tab3 = product_data_forecast_tab3.dropna(subset=["Date"])
    product_data_forecast_tab3.set_index("Date", inplace=True)
    product_data_forecast_tab3 = product_data_forecast_tab3.sort_index()
    
    forecast_df_tab3 = forecast_discounts_arima(product_data_forecast_tab3)
    
    # Strategic Recommendations
    st.markdown("<h3>📢 Strategic Recommendations</h3>", unsafe_allow_html=True)
    
    with st.spinner("Generating strategic insights..."):
        recommendations = generate_strategy_recommendation(
            selected_product,
            forecast_df_tab3 if not forecast_df_tab3.empty else pd.DataFrame(),
            sentiments if not product_reviews.empty else "No reviews available",
        )
    
    # Display recommendations without the box wrapper
    st.write(recommendations)
    
    # Price Recommendation
    st.markdown("<h3 style='margin-top: 1.5rem;'>💰 Optimal Price Recommendation</h3>", unsafe_allow_html=True)
    
    if st.button("🚀 Generate Optimal Price", key="price_btn"):
        with st.spinner("Calculating optimal price..."):
            price_recommendation = generate_price_recommendation(
                selected_product, 
                forecast_df_tab3 if not forecast_df_tab3.empty else pd.DataFrame(), 
                sentiments if not product_reviews.empty else "No reviews available"
            )
        
        st.markdown("### 🎯 Recommended Selling Price")
        st.write(price_recommendation)
        st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    # Strategy Section
    st.markdown("<h2 style='margin-bottom: 1rem;'>🎯 Actionable Strategy Guide</h2>", unsafe_allow_html=True)
    
    # Create a summary dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #1E3A8A;">🎯 Pricing Strategy</h4>
                <p style="color: #6b7280;">Based on market analysis and competitor trends</p>
                <ul style="color: #6b7280;">
                    <li>Dynamic pricing based on competitor movements</li>
                    <li>Seasonal discount optimization</li>
                    <li>Price elasticity consideration</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #1E3A8A;">📢 Promotional Campaigns</h4>
                <p style="color: #6b7280;">Data-driven campaign recommendations</p>
                <ul style="color: #6b7280;">
                    <li>Festive season bundle offers</li>
                    <li>Limited-time flash sales</li>
                    <li>Loyalty program integration</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #1E3A8A;">⭐ Customer Satisfaction</h4>
                <p style="color: #6b7280;">Enhance customer experience</p>
                <ul style="color: #6b7280;">
                    <li>Personalized recommendations</li>
                    <li>Post-purchase follow-ups</li>
                    <li>Quick resolution channels</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #1E3A8A;">📊 Performance Metrics</h4>
                <p style="color: #6b7280;">Track and optimize KPIs</p>
                <ul style="color: #6b7280;">
                    <li>Conversion rate optimization</li>
                    <li>Customer lifetime value</li>
                    <li>Market share growth</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# Chatbot Section (if enabled)
if enable_chatbot:
    st.markdown("---")
    st.markdown("<h2 style='margin-bottom: 1rem;'>💬 AI Assistant</h2>", unsafe_allow_html=True)
    
    # Prepare data for chatbot
    product_data_forecast_chat = product_data.copy()
    product_data_forecast_chat["Date"] = pd.to_datetime(product_data_forecast_chat["Date"], errors="coerce")
    product_data_forecast_chat = product_data_forecast_chat.dropna(subset=["Date"])
    product_data_forecast_chat.set_index("Date", inplace=True)
    product_data_forecast_chat = product_data_forecast_chat.sort_index()
    
    forecast_df_chat = forecast_discounts_arima(product_data_forecast_chat)
    
    # Chat interface
    user_query = st.text_area(
        "Ask me anything about pricing, discounts, or market strategy:",
        placeholder="Example: What's the best price for this product?",
        key="chat_input",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        submit_query = st.button("💬 Send Message", key="submit_chat", use_container_width=True)
    
    if submit_query and user_query:
        with st.spinner("Thinking..."):
            response = chatbot_response(
                user_query, 
                selected_product, 
                forecast_df_chat if not forecast_df_chat.empty else pd.DataFrame(), 
                sentiments if not product_reviews.empty else "No reviews available"
            )
        
        st.markdown(f"""
            <div style="background: #1E3A8A; border-radius: 12px; padding: 1rem; margin-top: 1rem;">
                <div style="color: white; font-weight: 600;">🤖 AI Response:</div>
                <div style="color: white; margin-top: 0.5rem;">{response}</div>
            </div>
        """, unsafe_allow_html=True)
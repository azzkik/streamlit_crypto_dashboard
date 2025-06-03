import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="🚀 Crypto Forecast Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ ADVANCED STYLING ------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .crypto-header {
        background: linear-gradient(90deg, #00f5ff, #ff00ff, #ffff00);
        background-size: 200% 200%;
        animation: gradient 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.1), rgba(255, 0, 255, 0.1));
        border: 2px solid;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.3);
    }
    
    .prediction-table {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        overflow: hidden;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', monospace;
        color: #00f5ff;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(0, 245, 255, 0.3);
    }
    
    .indicator-positive {
        color: #00ff88;
        font-weight: bold;
    }
    
    .indicator-negative {
        color: #ff4444;
        font-weight: bold;
    }
    
    .indicator-neutral {
        color: #ffaa00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ HELPER FUNCTIONS ------------------
def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
    avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = pd.Series(prices).ewm(span=fast).mean()
    exp2 = pd.Series(prices).ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd.values, signal_line.values, histogram.values

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = pd.Series(prices).rolling(window=window).mean()
    std = pd.Series(prices).rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band.values, sma.values, lower_band.values

def get_risk_level(change_percent):
    """Determine risk level based on volatility"""
    abs_change = abs(change_percent)
    if abs_change > 10:
        return "HIGH", "#ff4444"
    elif abs_change > 5:
        return "MEDIUM", "#ffaa00"
    else:
        return "LOW", "#00ff88"

def calculate_confidence_interval(prediction, confidence=0.95):
    """Calculate confidence interval for predictions"""
    std_dev = np.std(prediction) * 0.1  # Simplified confidence calculation
    margin = std_dev * 1.96  # 95% confidence
    upper = prediction + margin
    lower = prediction - margin
    return upper, lower

# ------------------ SIDEBAR CONTROLS ------------------
st.sidebar.markdown("## 🎛️ Control Panel")

crypto_choice = st.sidebar.selectbox(
    "🪙 Select Cryptocurrency:",
    ("KCS", "BTC"),
    index=0
)

# Advanced Parameters
st.sidebar.markdown("### 📊 Analysis Parameters")
forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10)
confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95)
rsi_period = st.sidebar.slider("RSI Period", 10, 30, 14)

# Risk Management
st.sidebar.markdown("### ⚠️ Risk Management")
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 5.0)
take_profit_pct = st.sidebar.slider("Take Profit (%)", 5.0, 50.0, 15.0)

# Model paths
if crypto_choice == "KCS":
    model_path = "lstm_model_kcs.keras"
    file_path = "kcs_price.csv"
else:
    model_path = "lstm_model_btc.keras"
    file_path = "btc_price.csv"

# ------------------ MAIN DASHBOARD ------------------
st.markdown('<h1 class="crypto-header">🚀 CRYPTO AI FORECAST HUB</h1>', unsafe_allow_html=True)

# ------------------ LOAD MODEL & DATA ------------------
@st.cache_resource
def load_model(path):
    try:
        return tf.keras.models.load_model(path)
    except:
        st.error(f"Model file '{path}' not found!")
        return None

model = load_model(model_path)

if model is None:
    st.stop()

# Load data
try:
    data = pd.read_csv(file_path)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
except FileNotFoundError:
    st.error(f"File '{file_path}' tidak ditemukan!")
    st.stop()

if "close" not in data.columns:
    st.error("CSV must have 'close' column!")
    st.stop()

# ------------------ DATA PROCESSING ------------------
close_data = data["close"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

if len(scaled_data) < 30:
    st.warning("Insufficient data (need at least 30 points)")
    st.stop()

# Get predictions
last_30_days = scaled_data[-30:]
input_seq = last_30_days.reshape(1, 30, 1)

# Generate predictions for specified days
predictions = []
current_seq = input_seq.copy()

for _ in range(forecast_days):
    pred = model.predict(current_seq, verbose=0)
    predictions.append(pred[0, 0])
    # Update sequence for next prediction
    current_seq = np.roll(current_seq, -1, axis=1)
    current_seq[0, -1, 0] = pred[0, 0]

predictions = np.array(predictions).reshape(-1, 1)
predictions = scaler.inverse_transform(predictions).flatten()

# Calculate confidence intervals
upper_bound, lower_bound = calculate_confidence_interval(predictions, confidence_level/100)

# ------------------ METRICS SECTION ------------------
col1, col2, col3, col4 = st.columns(4)

current_price = close_data[-1, 0]
predicted_price = predictions[-1]
change_percent = (predicted_price - current_price) / current_price * 100
risk_level, risk_color = get_risk_level(change_percent)

with col1:
    st.metric(
        label="💰 Current Price",
        value=f"${current_price:,.2f}",
        delta=None
    )

with col2:
    st.metric(
        label=f"🔮 {forecast_days}D Forecast",
        value=f"${predicted_price:,.2f}",
        delta=f"{change_percent:+.2f}%"
    )

with col3:
    st.metric(
        label="📈 Potential Gain",
        value=f"{abs(change_percent):.2f}%",
        delta=None
    )

with col4:
    st.metric(
        label="⚠️ Risk Level",
        value=risk_level,
        delta=None
    )

# ------------------ TECHNICAL INDICATORS ------------------
st.markdown("## 📊 Technical Analysis")

# Calculate indicators
prices = close_data[-100:].flatten()  # Last 100 days for indicators
rsi = calculate_rsi(prices, rsi_period)
macd, macd_signal, macd_hist = calculate_macd(prices)
bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)

# Display current indicator values
col1, col2, col3 = st.columns(3)

with col1:
    current_rsi = rsi[-1] if len(rsi) > 0 else 50
    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
    rsi_color = "red" if current_rsi > 70 else "green" if current_rsi < 30 else "orange"
    st.markdown(f"""
    <div class="metric-card">
        <h4>RSI ({rsi_period})</h4>
        <h2 style="color: {rsi_color}">{current_rsi:.1f}</h2>
        <p>{rsi_status}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    current_macd = macd[-1] if len(macd) > 0 else 0
    macd_trend = "Bullish" if current_macd > 0 else "Bearish"
    macd_color = "green" if current_macd > 0 else "red"
    st.markdown(f"""
    <div class="metric-card">
        <h4>MACD</h4>
        <h2 style="color: {macd_color}">{current_macd:.2f}</h2>
        <p>{macd_trend}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    price_position = "Upper" if current_price > bb_upper[-1] else "Lower" if current_price < bb_lower[-1] else "Middle"
    bb_color = "red" if price_position == "Upper" else "green" if price_position == "Lower" else "orange"
    st.markdown(f"""
    <div class="metric-card">
        <h4>Bollinger Bands</h4>
        <h2 style="color: {bb_color}">{price_position}</h2>
        <p>Band Position</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------ ADVANCED CHARTS ------------------
st.markdown("## 📈 Advanced Price Analysis")

# Create subplots
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=('Price & Predictions', 'Volume (Simulated)', 'RSI', 'MACD'),
    row_heights=[0.5, 0.2, 0.15, 0.15]
)

# Historical data
historical_data = close_data[-60:].flatten()
if "date" in data.columns:
    historical_dates = data["date"].values[-60:]
    forecast_dates = pd.date_range(historical_dates[-1] + pd.Timedelta(days=1), periods=forecast_days)
else:
    historical_dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
    forecast_dates = pd.date_range(pd.Timestamp.today() + pd.Timedelta(days=1), periods=forecast_days)

# Main price chart with Bollinger Bands
fig.add_trace(
    go.Scatter(x=historical_dates[-30:], y=bb_upper[-30:], fill=None, mode='lines', 
               line_color='rgba(255,255,255,0.2)', name='BB Upper', showlegend=False),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=historical_dates[-30:], y=bb_lower[-30:], fill='tonexty', mode='lines',
               line_color='rgba(255,255,255,0.2)', name='BB Lower', 
               fillcolor='rgba(0,245,255,0.1)', showlegend=False),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=historical_dates[-30:], y=bb_middle[-30:], mode='lines',
               line=dict(color='rgba(255,255,255,0.5)', dash='dash'), name='BB Middle'),
    row=1, col=1
)

# Historical prices
fig.add_trace(
    go.Scatter(x=historical_dates, y=historical_data, mode='lines',
               line=dict(color='#00f5ff', width=2), name='Historical Price'),
    row=1, col=1
)

# Predictions with confidence interval
fig.add_trace(
    go.Scatter(x=forecast_dates, y=upper_bound, fill=None, mode='lines',
               line_color='rgba(255,0,255,0.3)', name='Upper Bound', showlegend=False),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=forecast_dates, y=lower_bound, fill='tonexty', mode='lines',
               line_color='rgba(255,0,255,0.3)', name='Confidence Interval',
               fillcolor='rgba(255,0,255,0.2)'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=forecast_dates, y=predictions, mode='lines+markers',
               line=dict(color='#ff00ff', width=3), name='AI Prediction'),
    row=1, col=1
)

# Simulated volume
volume_sim = np.random.normal(1000000, 200000, len(historical_dates))
volume_sim = np.abs(volume_sim)
fig.add_trace(
    go.Bar(x=historical_dates, y=volume_sim, name='Volume', 
           marker_color='rgba(0,245,255,0.6)'),
    row=2, col=1
)

# RSI
rsi_dates = historical_dates[-len(rsi):]
fig.add_trace(
    go.Scatter(x=rsi_dates, y=rsi, mode='lines', line=dict(color='#ffaa00', width=2), name='RSI'),
    row=3, col=1
)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# MACD
macd_dates = historical_dates[-len(macd):]
fig.add_trace(
    go.Scatter(x=macd_dates, y=macd, mode='lines', line=dict(color='#00ff88', width=2), name='MACD'),
    row=4, col=1
)
fig.add_trace(
    go.Scatter(x=macd_dates, y=macd_signal, mode='lines', 
               line=dict(color='#ff4444', width=2), name='Signal'),
    row=4, col=1
)

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title=f'{crypto_choice} - Advanced Technical Analysis',
    title_font=dict(size=20, color='#00f5ff')
)

fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

st.plotly_chart(fig, use_container_width=True)

# ------------------ PREDICTION TABLE ------------------
st.markdown("## 🔮 Detailed Predictions")

pred_df = pd.DataFrame({
    "Day": np.arange(1, forecast_days + 1),
    "Date": forecast_dates.strftime('%Y-%m-%d'),
    "Predicted Price": predictions,
    "Upper Bound": upper_bound,
    "Lower Bound": lower_bound,
    "Daily Change %": np.concatenate([[0], np.diff(predictions)/predictions[:-1]*100])
})

st.dataframe(
    pred_df.style.format({
        "Predicted Price": "${:.2f}",
        "Upper Bound": "${:.2f}",
        "Lower Bound": "${:.2f}",
        "Daily Change %": "{:.2f}%"
    }),
    use_container_width=True
)

# ------------------ AI RECOMMENDATION SYSTEM ------------------
st.markdown("## 🤖 AI Trading Recommendation")

# Determine recommendation based on multiple factors
factors = {
    "price_trend": change_percent,
    "rsi": current_rsi,
    "macd": current_macd,
    "bb_position": price_position
}

# Scoring system
score = 0
if change_percent > 2: score += 2
elif change_percent < -2: score -= 2

if current_rsi < 30: score += 1
elif current_rsi > 70: score -= 1

if current_macd > 0: score += 1
else: score -= 1

if price_position == "Lower": score += 1
elif price_position == "Upper": score -= 1

# Final recommendation
if score >= 3:
    recommendation = "STRONG BUY"
    rec_color = "#00ff88"
    rec_emoji = "🚀"
elif score >= 1:
    recommendation = "BUY"
    rec_color = "#88ff88"
    rec_emoji = "📈"
elif score <= -3:
    recommendation = "STRONG SELL"
    rec_color = "#ff4444"
    rec_emoji = "🔻"
elif score <= -1:
    recommendation = "SELL"
    rec_color = "#ff8888"
    rec_emoji = "📉"
else:
    recommendation = "HOLD"
    rec_color = "#ffaa00"
    rec_emoji = "⏸️"

st.markdown(f"""
<div class="recommendation-card" style="border-color: {rec_color};">
    <h2>{rec_emoji} AI RECOMMENDATION</h2>
    <h1 style="color: {rec_color}; font-size: 3rem;">{recommendation}</h1>
    <p style="font-size: 1.2rem;">Confidence Score: {abs(score)}/5</p>
</div>
""", unsafe_allow_html=True)

# ------------------ RISK MANAGEMENT CALCULATOR ------------------
st.markdown("## ⚠️ Risk Management")

col1, col2 = st.columns(2)

with col1:
    investment_amount = st.number_input("💰 Investment Amount ($)", min_value=100, value=1000, step=100)
    
    stop_loss_price = current_price * (1 - stop_loss_pct/100)
    max_loss = investment_amount * (stop_loss_pct/100)
    
    st.markdown(f"""
    <div class="metric-card">
        <h4>🛑 Stop Loss</h4>
        <p>Price: <strong>${stop_loss_price:.2f}</strong></p>
        <p>Max Loss: <strong style="color: #ff4444;">${max_loss:.2f}</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    take_profit_price = current_price * (1 + take_profit_pct/100)
    potential_profit = investment_amount * (take_profit_pct/100)
    
    st.markdown(f"""
    <div class="metric-card">
        <h4>🎯 Take Profit</h4>
        <p>Price: <strong>${take_profit_price:.2f}</strong></p>
        <p>Potential Profit: <strong style="color: #00ff88;">${potential_profit:.2f}</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Risk-Reward Ratio
risk_reward_ratio = potential_profit / max_loss if max_loss > 0 else 0
st.markdown(f"""
<div class="metric-card">
    <h4>📊 Risk-Reward Ratio</h4>
    <h2 style="color: {'#00ff88' if risk_reward_ratio >= 2 else '#ffaa00' if risk_reward_ratio >= 1 else '#ff4444'}">
        1:{risk_reward_ratio:.2f}
    </h2>
    <p>{'Excellent' if risk_reward_ratio >= 3 else 'Good' if risk_reward_ratio >= 2 else 'Fair' if risk_reward_ratio >= 1 else 'Poor'} Risk-Reward</p>
</div>
""", unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 0.9rem;">
    <strong>Disclaimer:</strong> This is for educational purposes. Always do your own research before trading.
</div>
""", unsafe_allow_html=True)
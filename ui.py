import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# --- 1. CSS STYLING ---
def apply_styling():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
        
        :root {
            --bg-color: #0E1117;
            --card-bg: #161B22;
            --accent-cyan: #00F0FF;
            --accent-orange: #FF9900;
            --text-primary: #E6E6E6;
            --text-secondary: #8B949E;
            --success-green: #238636;
            --danger-red: #DA3633;
        }

        .stApp { background-color: var(--bg-color); font-family: 'Inter', sans-serif; }
        
        /* Header Styles */
        .header-container {
            display: flex; justify-content: space-between; align-items: center;
            padding: 1rem 0; border-bottom: 1px solid #30363D; margin-bottom: 2rem;
        }
        .app-title {
            font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700;
            background: -webkit-linear-gradient(45deg, var(--accent-orange), var(--accent-cyan));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .live-indicator {
            font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
            color: var(--success-green); border: 1px solid var(--success-green);
            padding: 2px 8px; border-radius: 4px; text-transform: uppercase;
        }

        /* KPI Cards */
        .kpi-card {
            background: rgba(22, 27, 34, 0.7); backdrop-filter: blur(10px);
            border: 1px solid #30363D; border-radius: 8px; padding: 20px;
            text-align: left; transition: transform 0.2s;
        }
        .kpi-card:hover { border-color: var(--accent-cyan); transform: translateY(-2px); }
        .kpi-label {
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
            color: var(--text-secondary); text-transform: uppercase; margin-bottom: 5px;
        }
        .kpi-value {
            font-family: 'JetBrains Mono', monospace; font-size: 1.6rem;
            font-weight: 700; color: var(--text-primary);
        }
        .kpi-delta { font-size: 0.8rem; font-weight: 600; margin-left: 5px; }
        .delta-pos { color: var(--success-green); }
        .delta-neg { color: var(--danger-red); }
        
        /* Button Styling */
        .stButton > button {
            background-color: var(--accent-orange);
            color: black;
            font-weight: 700;
            border: none;
            border-radius: 4px;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'JetBrains Mono', monospace;
        }
        .stButton > button:hover {
            background-color: #CC7A00;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 2. COMPONENT FUNCTIONS ---
def render_header(ticker):
    st.markdown(f"""
    <div class="header-container">
        <div>
            <div class="app-title">QUANTUM {ticker}</div>
            <div style="color: #888; font-size: 0.9rem;">NEURAL NETWORK PREDICTION PROTOCOL</div>
        </div>
        <div class="live-indicator">● SYSTEM ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

def display_kpi(col, label, value, delta=None):
    delta_html = ""
    if delta is not None:
        color_class = "delta-pos" if delta > 0 else "delta-neg"
        sign = "+" if delta > 0 else ""
        delta_html = f'<span class="kpi-delta {color_class}">{sign}{delta}%</span>'
    
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value} {delta_html}</div>
    </div>
    """, unsafe_allow_html=True)

from plotly.subplots import make_subplots

def plot_chart(df, pred_df=None):
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price Action', 'Volume'),
                        row_heights=[0.7, 0.3])

    # 1. Historical Candlestick
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='OHLC'
    ), row=1, col=1)

    # 2. Moving Averages (50 & 200)
    # Calculate MAs if enough data
    if len(df) > 50:
        ma50 = df['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=df['Date'], y=ma50,
            mode='lines', name='MA50',
            line=dict(color='rgba(255, 165, 0, 0.6)', width=1)
        ), row=1, col=1)
        
    if len(df) > 200:
        ma200 = df['Close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(
            x=df['Date'], y=ma200,
            mode='lines', name='MA200',
            line=dict(color='rgba(0, 240, 255, 0.6)', width=1)
        ), row=1, col=1)

    # 3. Prediction Line
    if pred_df is not None:
        # Determine color based on trend
        start_price = pred_df['Predicted_Close'].iloc[0]
        end_price = pred_df['Predicted_Close'].iloc[-1]
        line_color = '#00FF00' if end_price >= start_price else '#FF0055'
        
        # Add connection line from last historical point
        last_hist_date = df['Date'].iloc[-1]
        last_hist_close = df['Close'].iloc[-1]
        
        # Combine last point with prediction for seamless line
        pred_x = [last_hist_date] + list(pred_df['Date'])
        pred_y = [last_hist_close] + list(pred_df['Predicted_Close'])
        
        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y,
            mode='lines', name='AI Forecast',
            line=dict(color=line_color, width=3, dash='dot')
        ), row=1, col=1)
        
        # Add final price annotation
        fig.add_annotation(
            x=pred_df['Date'].iloc[-1], y=end_price,
            text=f"${end_price:,.2f}",
            showarrow=True, arrowhead=1,
            row=1, col=1,
            font=dict(color=line_color, size=12)
        )

    # 4. Volume Bar Chart
    colors = ['red' if row['Open'] - row['Close'] >= 0 
              else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'],
        name='Volume', marker_color=colors, opacity=0.5
    ), row=2, col=1)

    # Layout Updates
    fig.update_layout(
        template="plotly_dark", height=700,
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(14, 17, 23, 0.5)',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor='rgba(0,0,0,0)')
    )
    
    # Hide x-axis range slider (we have our own volume chart)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) # Hide weekends
    
    st.plotly_chart(fig, use_container_width=True)



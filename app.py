import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import model_engine as engine
import ui
from stock_search import StockSearchEngine

# USAGE NOTES:
# ============
# This app now uses ENHANCED FEATURES including:
# - Fuzzy stock search (100+ stocks)
# - Ensemble predictions (Linear Regression + Random Forest + LSTM)
# - Confidence intervals
# - Risk classification
# - Natural language explanations
# - Warning system for unreliable predictions

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="STOCK PREDICTOR", layout="wide")

# Constants
START_DATE = (date.today() - timedelta(days=15*365)).strftime("%Y-%m-%d") # 15 years history
TODAY = date.today().strftime("%Y-%m-%d")

# Apply CSS
ui.apply_styling()

# Initialize search engine
search_engine = StockSearchEngine()

# --- Sidebar Inputs ---
st.sidebar.title("TERMINAL ACCESS")
st.sidebar.markdown("---")


# Main stock selection - Display company names instead of tickers
all_tickers = search_engine.get_all_tickers()
default_tickers = all_tickers[:11]  # Original 11 stocks

# Create a mapping of company names to tickers
company_to_ticker = {}
ticker_to_company = {}
for ticker in default_tickers:
    info = search_engine.get_stock_info(ticker)
    if info:
        company_name = f"{info['name']} ({ticker})"
        company_to_ticker[company_name] = ticker
        ticker_to_company[ticker] = company_name

# Display company names in dropdown
company_names = list(company_to_ticker.keys())
selected_company = st.sidebar.selectbox("SELECT ASSET", company_names)

# Get the ticker from selected company name
selected_ticker = company_to_ticker[selected_company]

# Get stock info
stock_info = search_engine.get_stock_info(selected_ticker)
if stock_info:
    st.sidebar.info(f"{stock_info['sector']} Sector")

initial_capital = st.sidebar.number_input("INITIAL CAPITAL ($)", value=10000, step=1000)

forecast_options = ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "5 Years"]
forecast_horizon = st.sidebar.selectbox("FORECAST HORIZON", forecast_options)

# Enhanced features toggle
# Enhanced features toggle replaced with fixed single model usage
use_ensemble = False # Forced to False as per user request to disable ensemble
# st.sidebar.checkbox("Use Ensemble Models", value=False, help="Combine LSTM with Linear Regression and Random Forest")

st.sidebar.markdown("---")
run_prediction = st.sidebar.button("RUN PREDICTION MODEL")

# --- Main Dashboard ---
# Load Data
with st.spinner('ESTABLISHING DATALINK...'):
    stock_data, sp500_data = engine.get_stock_data(selected_ticker)

if stock_data is not None and not stock_data.empty:
    # Render header with selected ticker
    ui.render_header(selected_ticker)
    
    # Calculate Historical Metrics immediately
    metrics = engine.calculate_metrics(stock_data, sp500_data)
    
    # Create engine instance for enhanced features
    engine_obj = engine.StockModelEngine(selected_ticker)
    engine_obj.stock_data = stock_data
    engine_obj.market_data = sp500_data
    engine_obj.current_price = stock_data['Close'].iloc[-1]
    engine_obj.metrics_calculator = engine.MetricsCalculator(stock_data, sp500_data)
    
    # Display risk profile immediately
    risk_profile = engine_obj.get_risk_profile()
    risk_level = risk_profile['risk_level']
    
    # Display risk badge
    risk_colors = {'Low': '#238636', 'Medium': '#FF9900', 'High': '#DA3633'}
    risk_color = risk_colors.get(risk_level, '#FF9900')
    
    st.markdown(f"""
    <div style="background: {risk_color}20; border: 2px solid {risk_color}; border-radius: 8px; padding: 12px; margin-bottom: 20px;">
        <span style="color: {risk_color}; font-weight: bold; font-size: 1.1rem;">
            RISK PROFILE: {risk_level.upper()}
        </span>
        <p style="margin-top: 8px; color: #E6E6E6; font-size: 0.9rem;">{risk_profile['explanation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display warnings if any
    if risk_profile['warnings']:
        st.warning("**PREDICTION WARNINGS**")
        for warning in risk_profile['warnings']:
            st.markdown(f"• {warning}")
        st.markdown("---")
    
    if run_prediction:
        with st.spinner("EXECUTING NEURAL NETWORK PROTOCOLS..."):
            # Map horizon to business days
            horizon_map = {
                "1 Day": 1,
                "1 Week": 5,
                "1 Month": 21,
                "3 Months": 63,
                "6 Months": 126,
                "1 Year": 252,
                "5 Years": 1260
            }
            future_steps = horizon_map[forecast_horizon]
            
            # Run enhanced prediction
            results = engine_obj.run_ensemble_prediction(
                forecast_days=future_steps,
                initial_capital=initial_capital,
                use_ensemble=use_ensemble
            )
            
            if results['trained_new']:
                st.success("MODEL RETRAINED AND SAVED.")
            else:
                st.info("LOADED EXISTING MODEL.")
            
            # Extract results
            current_price = results['current_price']
            final_price = results['final_price']
            confidence = results['confidence']
            scenarios = results['investment_scenarios']
            explanation = results['explanation']
            
            # --- Enhanced KPI Cards with Confidence ---
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            ui.display_kpi(kpi1, "CURRENT PRICE", f"${current_price:,.2f}")
            
            # Show future value with confidence range
            confidence_range = f"±${(confidence['confidence_high'] - confidence['confidence_low'])/2:.2f}"
            ui.display_kpi(
                kpi2,
                f"FUTURE VALUE ({confidence['confidence_level']} Confidence)",
                f"${scenarios['expected']:,.2f}",
                delta=round(scenarios['expected_return_pct'], 2)
            )
            
            ui.display_kpi(kpi3, "VOLATILITY", f"{metrics['Volatility']:.2%}")
            ui.display_kpi(kpi4, "SHARPE RATIO", f"{metrics['Sharpe']:.2f}")
            
            st.markdown("---")
            
            # --- Investment Scenarios ---
            st.subheader(">> INVESTMENT SCENARIOS")
            
            scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
            
            with scenario_col1:
                st.markdown(f"""
                <div style="background: rgba(218, 54, 51, 0.1); border: 1px solid #DA3633; border-radius: 6px; padding: 15px; text-align: center;">
                    <div style="color: #8B949E; font-size: 0.75rem; margin-bottom: 5px;">WORST CASE</div>
                    <div style="color: #DA3633; font-size: 1.4rem; font-weight: bold;">${scenarios['worst_case']:,.2f}</div>
                    <div style="color: #DA3633; font-size: 0.9rem;">{scenarios['worst_case_return_pct']:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with scenario_col2:
                st.markdown(f"""
                <div style="background: rgba(0, 240, 255, 0.1); border: 1px solid #00F0FF; border-radius: 6px; padding: 15px; text-align: center;">
                    <div style="color: #8B949E; font-size: 0.75rem; margin-bottom: 5px;">EXPECTED</div>
                    <div style="color: #00F0FF; font-size: 1.4rem; font-weight: bold;">${scenarios['expected']:,.2f}</div>
                    <div style="color: #00F0FF; font-size: 0.9rem;">{scenarios['expected_return_pct']:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with scenario_col3:
                st.markdown(f"""
                <div style="background: rgba(35, 134, 54, 0.1); border: 1px solid #238636; border-radius: 6px; padding: 15px; text-align: center;">
                    <div style="color: #8B949E; font-size: 0.75rem; margin-bottom: 5px;">BEST CASE</div>
                    <div style="color: #238636; font-size: 1.4rem; font-weight: bold;">${scenarios['best_case']:,.2f}</div>
                    <div style="color: #238636; font-size: 0.9rem;">{scenarios['best_case_return_pct']:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- Explanation Section ---
            st.subheader(">> WHY THIS PREDICTION?")
            with st.expander("View Detailed Explanation", expanded=True):
                st.markdown(explanation)
            
            # --- Model Performance (if ensemble used) ---
            if 'model_performance' in results:
                st.subheader(">> MODEL PERFORMANCE")
                perf_data = results['model_performance']
                
                perf_df = pd.DataFrame(perf_data).T
                perf_df['Model'] = ['Linear Regression', 'Random Forest']
                perf_df = perf_df[['Model', 'MAE', 'RMSE']]
                
                st.dataframe(perf_df, use_container_width=True)
                
                # Show ensemble weights
                if 'ensemble_details' in results:
                    weights = results['ensemble_details']['model_weights']
                    st.markdown(f"""
                    **Model Weights:**
                    - Linear Regression: {weights.get('linear', 0)*100:.1f}%
                    - Random Forest: {weights.get('random_forest', 0)*100:.1f}%
                    - LSTM (Deep Learning): {weights.get('lstm', 0)*100:.1f}%
                    """)
            
            st.markdown("---")
            
            # --- Price Trajectory Analysis ---
            st.subheader(">> PRICE TRAJECTORY ANALYSIS")
            
            # Add confidence visualization
            pred_df = results['predictions']
            ui.plot_chart(stock_data, pred_df)
            
            # Add confidence range annotation
            st.info(f"**Confidence Interval:** ${confidence['confidence_low']:,.2f} - ${confidence['confidence_high']:,.2f} ({confidence['confidence_level']} confidence)")
    
    else:
        # Show basic data if prediction not run
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        current_price = stock_data['Close'].iloc[-1]
        ui.display_kpi(kpi1, "CURRENT PRICE", f"${current_price:,.2f}")
        ui.display_kpi(kpi2, "VOLATILITY", f"{metrics['Volatility']:.2%}")
        ui.display_kpi(kpi3, "BETA", f"{metrics['Beta']:.2f}")
        ui.display_kpi(kpi4, "SHARPE", f"{metrics['Sharpe']:.2f}")
        
        st.markdown("---")
        ui.plot_chart(stock_data)
        st.info("INITIATE PREDICTION MODEL FROM SIDEBAR FOR FORECASTS")

else:
    st.error('Unable to fetch data for the selected stock. The ticker might be delisted, the API limit reached, or there is no data available. Please select another stock.')

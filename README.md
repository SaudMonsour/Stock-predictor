# Stock Predictor
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://xvaxbbzdgb2rhdxylpn9ez.streamlit.app/)<--- Click here to open it up

I built this project to help predict where stock prices might go over the next 5 years. Instead of just looking at the stock price history, it actually looks at the bigger picture—things like interest rates, inflation, and GDP—to make smarter forecasts.

## What it does

*   **Long-Term Forecasting**: It generates a 5-year price target, not just a prediction for tomorrow.
*   **Real-Time Data**: It fetches the latest stock prices and economic data up to today.
*   **Advanced AI**: It uses a mix of deep learning models (LSTM, CNN) to find patterns that are hard to see with the naked eye.
*   **Risk Assessment**: It gives you a clear "Best Case" and "Worst Case" scenario so you know the risks.

## The "Secret Sauce" (Features)

The model doesn't just guess; it learns from 11 specific features:
1.  **Stock Price** (Close): The main trend.
2.  **Interest Rates**: Because high rates usually hurt stocks.
3.  **Inflation**: To see if your money is losing value.
4.  **GDP**: The health of the entire economy.
5.  **Unemployment**: How strong the job market is.
6.  **Market Fear (VIX)**: Are investors panicked or calm?
7.  **S&P 500 Performance**: How the general market is doing.
*(Plus 4 other calculated financial metrics)*

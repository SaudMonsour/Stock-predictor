# Model Features

The Stock Predictor model learns from the following **11 features** to make its 5-year forecasts:

## 1. Core Stock Data
*   **Close**: The daily closing price of the stock.
    *   *Note*: The model uses the **past 60 days** of closing prices to predict the future. This is the most important feature.
    *   (Open, High, Low, and Volume are currently **excluded** to reduce noise for long-term 5-year forecasting).

## 2. Economic Indicators (from FRED)
*   **interest_rate** (`FEDFUNDS`): The Federal Funds Rate. Controls the cost of borrowing money. Higher rates usually hurt growth stocks.
*   **treasury_10y** (`DGS10`): 10-Year Treasury Yield. The benchmark for "risk-free" return. Used to value companies.
*   **inflation** (`CPIAUCSL`): Consumer Price Index. Measures how fast prices are rising in the economy.
*   **gdp** (`GDP`): Gross Domestic Product. The total health and output of the US economy.
*   **unemployment** (`UNRATE`): The percentage of the labor force without a job. Indicates economic strength.

## 3. Market Sentiment (from Yahoo Finance)
*   **vix** (`^VIX`): The "Fear Gauge". Measures expected market volatility. High VIX = High Fear.
*   **market_return** (`^GSPC`): Daily returns of the S&P 500 index. Shows how the overall market is performing.

## 4. Derived Intelligence (Calculated)
*   **inflation_change**: The year-over-year change in inflation. Helps detect if inflation is speeding up or slowing down.
*   **real_interest_rate**: The "true" cost of money (Interest Rate minus Inflation).
*   **high_volatility**: A flag (0 or 1) indicating if the market is currently in a high-risk state (VIX > 20).

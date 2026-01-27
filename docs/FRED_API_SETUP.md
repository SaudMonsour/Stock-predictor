# FRED API Key Setup Guide

## What is FRED?

FRED (Federal Reserve Economic Data) is a free database maintained by the Federal Reserve Bank of St. Louis. It contains over 800,000 economic time series from various sources.

## Getting Your Free API Key

### Step 1: Create Account
1. Go to: https://fred.stlouisfed.org/
2. Click "My Account" (top right)
3. Create a free account (takes 1 minute)

### Step 2: Request API Key
1. After logging in, go to: https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request API Key"
3. Fill out simple form (name, email, purpose: "Research/Educational")
4. API key appears immediately - copy it!

### Step 3: Set Up in Your Project

#### Option A: Environment Variable (Recommended)
**Windows (PowerShell)**:
```powershell
setx FRED_API_KEY "your_api_key_here"
```

**Windows (Command Prompt)**:
```cmd
setx FRED_API_KEY "your_api_key_here"
```

Then restart your terminal/IDE.

#### Option B: Pass Directly in Code
```python
fetcher = EconomicDataFetcher(fred_api_key="your_api_key_here")
```

#### Option C: Config File
Create `.env` file:
```
FRED_API_KEY=your_api_key_here
```

Then in code:
```python
from dotenv import load_dotenv
load_dotenv()
# Now os.getenv('FRED_API_KEY') will work
```

## Install Required Package

```bash
pip install fredapi
```

## Available Economic Indicators

The fetcher automatically retrieves:

| Indicator | FRED Code | Description |
|-----------|-----------|-------------|
| Interest Rate | FEDFUNDS | Federal Funds Effective Rate |
| 10Y Treasury | DGS10 | 10-Year Treasury Constant Maturity |
| Inflation | CPIAUCSL | Consumer Price Index |
| GDP | GDP | Real Gross Domestic Product |
| Unemployment | UNRATE | Unemployment Rate |
| Sentiment | UMCSENT | Consumer Sentiment Index |
| VIX | ^VIX | CBOE Volatility Index (from Yahoo) |

## Quick Test

After setting up your API key:

```bash
cd c:\Users\saudc\Downloads\Project
python economic_data_fetcher.py
```

This will:
1. Fetch all indicators for 2010-2021
2. Download AAPL stock data
3. Align economic data with trading days
4. Create derived features
5. Save sample CSV file

## Troubleshooting

**Problem**: "FRED API not initialized"
- **Solution**: Make sure FRED_API_KEY environment variable is set and terminal is restarted

**Problem**: "fredapi not installed"
- **Solution**: Run `pip install fredapi`

**Problem**: API key invalid
- **Solution**: Double-check you copied the full key from FRED website

## Usage in Your Models

```python
from economic_data_fetcher import EconomicDataFetcher

# Initialize
fetcher = EconomicDataFetcher()

# Fetch indicators
indicators = fetcher.fetch_all_indicators("2010-01-01", "2021-12-31")

# Align with your stock data
enhanced_data = fetcher.align_with_trading_days(stock_data, indicators)

# Add derived features
enhanced_data = fetcher.calculate_derived_features(enhanced_data)

# Now use enhanced_data for model training!
```

## Rate Limits

FRED API limits:
- **120 requests per 60 seconds**
- **Unlimited daily requests**

The fetcher caches data, so you won't hit limits in normal usage.

## Alternative: Use Without FRED (Yahoo Finance Only)

If you don't want to set up FRED:

```python
# Fetcher will still work with VIX and S&P 500 data from Yahoo Finance
fetcher = EconomicDataFetcher(fred_api_key=None)
indicators = fetcher.fetch_all_indicators("2010-01-01", "2021-12-31")
# You'll get VIX and market returns, but no FRED indicators
```

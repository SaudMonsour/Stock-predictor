"""
Economic Data Fetcher

Fetches macroeconomic indicators from free sources to enhance stock prediction models.

Data Sources:
- FRED API (Federal Reserve): Interest rates, inflation, GDP, unemployment
- Yahoo Finance: VIX (volatility index), market indices

Setup:
1. Get free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
2. Set environment variable: FRED_API_KEY or pass to constructor
3. pip install fredapi

Usage:
    fetcher = EconomicDataFetcher(fred_api_key="YOUR_KEY")
    indicators = fetcher.fetch_all_indicators("2010-01-01", "2021-12-31")
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("⚠️  fredapi not installed. Install with: pip install fredapi")


class EconomicDataFetcher:
    """Fetches and aligns economic indicators with stock data"""
    
    # FRED series codes
    FRED_SERIES = {
        'interest_rate': 'FEDFUNDS',  # Federal Funds Rate
        'treasury_10y': 'DGS10',       # 10-Year Treasury Yield
        'inflation': 'CPIAUCSL',       # Consumer Price Index
        'gdp': 'GDP',                  # Real GDP
        'unemployment': 'UNRATE',      # Unemployment Rate
        'consumer_sentiment': 'UMCSENT'  # University of Michigan Consumer Sentiment
    }
    
    # Default API Key (User provided for backend use)
    DEFAULT_API_KEY = "e62ee4b2ab2c07f7da73bcb50d273cc9"
    
    def __init__(self, fred_api_key=None):
        """
        Initialize economic data fetcher
        
        Args:
            fred_api_key: FRED API key. If None, tries to read from FRED_API_KEY env variable, then uses default.
        """
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY') or self.DEFAULT_API_KEY
        self.fred = None
        
        if FRED_AVAILABLE and self.fred_api_key:
            try:
                self.fred = Fred(api_key=self.fred_api_key)
                print("✅ FRED API initialized successfully")
            except Exception as e:
                print(f"⚠️  FRED API initialization failed: {e}")
        elif not FRED_AVAILABLE:
            print("⚠️  Install fredapi: pip install fredapi")
        else:
            print("⚠️  No FRED API key provided. Set FRED_API_KEY environment variable")
            print("    Get free key: https://fred.stlouisfed.org/docs/api/api_key.html")
    
    def fetch_fred_series(self, series_id, start_date, end_date):
        """Fetch a single FRED series"""
        if not self.fred:
            print(f"⚠️  Cannot fetch {series_id}: FRED not initialized")
            return None
        
        try:
            data = self.fred.get_series(series_id, start_date, end_date)
            return data
        except Exception as e:
            print(f"❌ Error fetching {series_id}: {e}")
            return None
    
    def fetch_vix(self, start_date, end_date):
        """Fetch VIX (CBOE Volatility Index) from Yahoo Finance"""
        try:
            vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            return vix['Close']
        except Exception as e:
            print(f"❌ Error fetching VIX: {e}")
            return None
    
    def fetch_sp500_returns(self, start_date, end_date):
        """Fetch S&P 500 returns for market context"""
        try:
            sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
            if isinstance(sp500.columns, pd.MultiIndex):
                sp500.columns = sp500.columns.get_level_values(0)
            # Calculate daily returns
            returns = sp500['Close'].pct_change()
            return returns
        except Exception as e:
            print(f"❌ Error fetching S&P 500: {e}")
            return None
    
    def fetch_all_indicators(self, start_date, end_date):
        """
        Fetch all available economic indicators
        
        Returns:
            DataFrame with economic indicators indexed by date
        """
        print(f"\n📊 Fetching economic indicators ({start_date} to {end_date})...")
        
        indicators = pd.DataFrame()
        
        # Fetch FRED data
        if self.fred:
            for name, series_id in self.FRED_SERIES.items():
                print(f"   Fetching {name} ({series_id})...")
                data = self.fetch_fred_series(series_id, start_date, end_date)
                if data is not None:
                    indicators[name] = data
        
        # Fetch VIX
        print(f"   Fetching VIX (volatility)...")
        vix = self.fetch_vix(start_date, end_date)
        if vix is not None:
            indicators['vix'] = vix
        
        # Fetch S&P 500 returns
        print(f"   Fetching S&P 500 returns...")
        sp500_ret = self.fetch_sp500_returns(start_date, end_date)
        if sp500_ret is not None:
            indicators['market_return'] = sp500_ret
        
        if indicators.empty:
            print("❌ No indicators fetched")
            return None
        
        print(f"✅ Fetched {len(indicators.columns)} indicators")
        return indicators
    
    def align_with_trading_days(self, stock_data, indicators):
        """
        Align economic indicators with stock trading days
        
        Economic data is often monthly/quarterly, need to forward-fill for daily stock data
        
        Args:
            stock_data: DataFrame with Date column and stock prices
            indicators: DataFrame with economic indicators
            
        Returns:
            stock_data with economic indicator columns added
        """
        if indicators is None or indicators.empty:
            print("⚠️  No indicators to align")
            return stock_data
        
        print(f"\n🔄 Aligning indicators with trading days...")
        
        # Ensure stock_data has Date as column (not index)
        if 'Date' not in stock_data.columns and stock_data.index.name == 'Date':
            stock_data = stock_data.reset_index()
        
        # Convert Date to datetime
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        # Ensure indicators index is datetime
        indicators.index = pd.to_datetime(indicators.index)
        
        # Merge on date, forward-fill for missing values
        aligned = stock_data.copy()
        
        for col in indicators.columns:
            # Create a series for merging
            indicator_series = indicators[col].copy()
            indicator_series.index.name = 'Date'
            
            # Merge and forward-fill
            merged = pd.merge_asof(
                aligned.sort_values('Date'),
                indicator_series.reset_index().sort_values('Date'),
                on='Date',
                direction='backward'
            )
            
            aligned[col] = merged[col].values
        
        # Fill any remaining NaN with forward fill then backward fill
        for col in indicators.columns:
            aligned[col] = aligned[col].fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Aligned {len(indicators.columns)} indicators")
        print(f"   Resulting dataframe shape: {aligned.shape}")
        print(f"   Columns: {list(aligned.columns)}")
        
        return aligned
    
    def calculate_derived_features(self, df):
        """
        Calculate derived features from economic indicators
        
        Args:
            df: DataFrame with economic indicators
            
        Returns:
            DataFrame with additional derived features
        """
        if 'inflation' in df.columns:
            # Inflation rate of change
            df['inflation_change'] = df['inflation'].pct_change(12)  # YoY change
        
        if 'interest_rate' in df.columns and 'inflation' in df.columns:
            # Real interest rate = nominal rate - inflation
            # Approximate: use interest rate - inflation rate of change
            df['real_interest_rate'] = df['interest_rate'] - df['inflation'].pct_change(12) * 100
        
        if 'gdp' in df.columns:
            # GDP growth rate (YoY)
            df['gdp_growth'] = df['gdp'].pct_change(4) * 100  # Quarterly * 4 periods
        
        if 'vix' in df.columns:
            # VIX categories (fear levels)
            df['high_volatility'] = (df['vix'] > 20).astype(int)
            df['extreme_volatility'] = (df['vix'] > 30).astype(int)
        
        if 'unemployment' in df.columns:
            # Unemployment trend
            df['unemployment_trend'] = df['unemployment'].diff()
        
        return df


def demo_usage():
    """Demonstrate how to use the economic data fetcher"""
    
    print("\n" + "="*60)
    print("  ECONOMIC DATA FETCHER - DEMO")
    print("="*60)
    
    # Initialize fetcher (uses FRED_API_KEY from environment)
    fetcher = EconomicDataFetcher()
    
    # Fetch indicators for 2010-2021 (training period)
    indicators = fetcher.fetch_all_indicators("2010-01-01", "2021-12-31")
    
    if indicators is not None:
        print(f"\n📊 FETCHED INDICATORS:")
        print(indicators.tail())
        print(f"\nShape: {indicators.shape}")
        print(f"Date range: {indicators.index.min()} to {indicators.index.max()}")
        
        # Example: fetch stock data and align
        print(f"\n📈 Fetching AAPL stock data...")
        stock = yf.download("AAPL", start="2010-01-01", end="2021-12-31", progress=False)
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)
        stock.reset_index(inplace=True)
        
        # Align indicators with stock data
        aligned = fetcher.align_with_trading_days(stock, indicators)
        
        # Calculate derived features
        aligned = fetcher.calculate_derived_features(aligned)
        
        print(f"\n✅ FINAL DATASET:")
        print(aligned.tail())
        print(f"\nTotal columns: {len(aligned.columns)}")
        print(f"Columns: {list(aligned.columns)}")
        
        # Save sample
        aligned.to_csv("sample_data_with_economics.csv", index=False)
        print(f"\n💾 Saved sample to: sample_data_with_economics.csv")


if __name__ == "__main__":
    demo_usage()

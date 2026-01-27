"""
Test script for Direct Multi-Step Forecasting

Tests the new direct forecasting module on AAPL, NVDA, TSLA.
Verifies that predictions are generated for multiple horizons without scaling.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from direct_forecasting import DirectMultiStepForecaster, map_forecast_horizon_to_days
from datetime import datetime, timedelta


def create_features(df: pd.DataFrame, window=60):
    """Create feature matrix from price data"""
    features = []
    targets = []
    dates = []
    
    close_prices = df['Close'].values
    date_values = df['Date'].values
    
    for i in range(window, len(close_prices)):
        # Last 60 days as features
        features.append(close_prices[i-window:i])
        # Current price as target
        targets.append(close_prices[i])
        dates.append(date_values[i])
    
    return np.array(features), pd.Series(targets), pd.Series(dates)


def test_direct_forecasting(ticker: str):
    """Test direct forecasting on a single stock"""
    print(f"\n{'='*60}")
    print(f"Testing Direct Forecasting on {ticker}")
    print(f"{'='*60}\n")
    
    # Download data
    print(f"📊 Downloading {ticker} data...")
    stock = yf.Ticker(ticker)
    df = stock.history(period='5y')
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✓ Downloaded {len(df)} days of data")
    
    # Create features
    print(f"\n🔧 Creating features (60-day windows)...")
    X, y, dates = create_features(df)
    print(f"✓ Created {len(X)} samples")
    
    # Split train/validation (time-aware)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    dates_train, dates_val = dates[:split_idx], dates[split_idx:]
    
    print(f"✓ Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    print(f"  Train period: {dates_train.min()} to {dates_train.max()}")
    print(f"  Val period: {dates_val.min()} to {dates_val.max()}")
    
    # Test with Ridge model
    print(f"\n🚀 Training Direct Forecaster (Ridge)...")
    forecaster_ridge = DirectMultiStepForecaster(model_type='ridge')
    forecaster_ridge.fit(X_train, y_train, dates_train, horizons=['1d', '7d', '30d'])
    
    # Test with Random Forest
    print(f"\n🌲 Training Direct Forecaster (Random Forest)...")
    forecaster_rf = DirectMultiStepForecaster(model_type='random_forest')
    forecaster_rf.fit(X_train, y_train, dates_train, horizons=['1d', '7d', '30d'])
    
    # Validate
    print(f"\n📊 Validation Results:")
    print(f"\n{'Horizon':<10} {'Model':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 55)
    
    for horizon in ['1d', '7d', '30d']:
        # Ridge metrics
        metrics_ridge = forecaster_ridge.get_validation_error(
            X_val, y_val, dates_val, horizon
        )
        print(f"{horizon:<10} {'Ridge':<15} "
              f"${metrics_ridge['MAE']:<9.2f} "
              f"${metrics_ridge['RMSE']:<9.2f} "
              f"{metrics_ridge['MAPE']:<9.2f}%")
        
        # RF metrics
        metrics_rf = forecaster_rf.get_validation_error(
            X_val, y_val, dates_val, horizon
        )
        print(f"{horizon:<10} {'Random Forest':<15} "
              f"${metrics_rf['MAE']:<9.2f} "
              f"${metrics_rf['RMSE']:<9.2f} "
              f"{metrics_rf['MAPE']:<9.2f}%")
    
    # Test prediction on last sample
    print(f"\n🔮 Making predictions on latest data:")
    last_sample = X_val[-1:] # Last 60 days
    current_price = y_val.iloc[-1]
    
    print(f"  Current price: ${current_price:.2f}")
    print(f"\n  Predictions (Ridge):")
    for horizon in ['1d', '7d', '30d']:
        pred_ridge = forecaster_ridge.predict(last_sample, horizon)[0]
        change_pct = ((pred_ridge - current_price) / current_price) * 100
        print(f"    {horizon}: ${pred_ridge:.2f} ({change_pct:+.2f}%)")
    
    print(f"\n  Predictions (Random Forest):")
    for horizon in ['1d', '7d', '30d']:
        pred_rf = forecaster_rf.predict(last_sample, horizon)[0]
        change_pct = ((pred_rf - current_price) / current_price) * 100
        print(f"    {horizon}: ${pred_rf:.2f} ({change_pct:+.2f}%)")
    
    return forecaster_ridge, forecaster_rf


def main():
    """Run tests on AAPL, NVDA, TSLA"""
    test_tickers = ['AAPL', 'NVDA', 'TSLA']
    
    print("="*60)
    print("Direct Multi-Step Forecasting Test Suite")
    print("="*60)
    print(f"\nTesting on: {', '.join(test_tickers)}")
    print(f"Horizons: 1 day, 7 days, 30 days")
    print(f"Models: Ridge, Random Forest")
    
    results = {}
    for ticker in test_tickers:
        try:
            ridge, rf = test_direct_forecasting(ticker)
            results[ticker] = {'ridge': ridge, 'rf': rf}
        except Exception as e:
            print(f"\n❌ Error testing {ticker}: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ Testing Complete!")
    print(f"Successfully tested {len(results)}/{len(test_tickers)} tickers")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
5-Year Forecast Validation and Optimization System

This script:
1. Trains models on data up to 2021-12-31
2. Generates 5-year forecasts (predicting to 2026)
3. Fetches actual 2026 data
4. Calculates prediction errors
5. Iteratively optimizes model hyperparameters to minimize error

Usage:
    python validate_5year_forecast.py --ticker AAPL --optimize
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import model_engine
from economic_data_fetcher import EconomicDataFetcher
import warnings
warnings.filterwarnings('ignore')


class ForecastValidator:
    """Validates 5-year forecasts against actual data and optimizes models"""
    
    def __init__(self, ticker, train_end='2021-12-31'):
        """
        Initialize validator
        
        Args:
            ticker: Stock ticker symbol
            train_end: Last date for training data (default: 2021-12-31)
        """
        self.ticker = ticker
        self.train_end = train_end
        self.train_data = None
        self.actual_2026_data = None
        self.predictions = None
        self.results = {}
        
    def fetch_training_data(self):
        """Fetch historical data up to train_end for model training"""
        print(f"\n📊 Fetching training data for {self.ticker} (up to {self.train_end})...")
        
        try:
            data = yf.download(self.ticker, start="2010-01-01", end=self.train_end, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.reset_index(inplace=True)
            
            # Fetch and align economic data
            print("   Fetching economic context data...")
            fetcher = EconomicDataFetcher()
            start_date = data['Date'].iloc[0].strftime('%Y-%m-%d')
            end_date = data['Date'].iloc[-1].strftime('%Y-%m-%d')
            
            indicators = fetcher.fetch_all_indicators(start_date, end_date)
            if indicators is not None:
                aligned = fetcher.align_with_trading_days(data, indicators)
                data = fetcher.calculate_derived_features(aligned)
                print(f"   ✅ Added {len(data.columns) - 7} economic context features")
            
            self.train_data = data
            print(f"✅ Loaded {len(data)} days of training data")
            print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
            print(f"   Final price: ${data['Close'].iloc[-1]:.2f}")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching training data: {e}")
            return None
    
    def fetch_actual_2026_data(self):
        """Fetch actual stock data from 2026"""
        print(f"\n📊 Fetching actual 2026 data for {self.ticker}...")
        
        try:
            data = yf.download(self.ticker, start="2026-01-01", end="2026-12-31", auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.reset_index(inplace=True)
            
            self.actual_2026_data = data
            print(f"✅ Loaded {len(data)} days of actual 2026 data")
            print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
            print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            print(f"   Latest price: ${data['Close'].iloc[-1]:.2f}")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching 2026 data: {e}")
            return None
    
    def generate_5year_prediction(self, use_ensemble=True):
        """
        Generate 5-year (1260 trading days) forecast
        
        Args:
            use_ensemble: Whether to use ensemble models
        """
        print(f"\n🤖 Generating 5-year prediction using model...")
        
        try:
            # Create engine instance
            engine = model_engine.StockModelEngine(self.ticker)
            engine.stock_data = self.train_data
            
            # Fetch market data for the same period
            sp500 = yf.download("^GSPC", start="2010-01-01", end=self.train_end, auto_adjust=True)
            if isinstance(sp500.columns, pd.MultiIndex):
                sp500.columns = sp500.columns.get_level_values(0)
            sp500.reset_index(inplace=True)
            engine.market_data = sp500
            
            engine.current_price = self.train_data['Close'].iloc[-1]
            engine.metrics_calculator = model_engine.MetricsCalculator(self.train_data, sp500)
            
            # Generate 5-year prediction (approximately 1260 trading days)
            forecast_days = 1260
            print(f"   Forecasting {forecast_days} trading days (~5 years) ahead...")
            
            results = engine.run_ensemble_prediction(
                forecast_days=forecast_days,
                initial_capital=10000,
                use_ensemble=use_ensemble
            )
            
            self.predictions = results['predictions']
            self.results = results
            
            print(f"✅ Prediction generated successfully")
            print(f"   Predicted final price: ${results['final_price']:.2f}")
            print(f"   Starting price: ${results['current_price']:.2f}")
            print(f"   Predicted change: {((results['final_price']/results['current_price'])-1)*100:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ Error generating prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_errors(self):
        """Calculate error metrics between predictions and actual data"""
        print(f"\n📈 Calculating prediction errors...")
        
        if self.predictions is None or self.actual_2026_data is None:
            print("❌ Missing predictions or actual data")
            return None
        
        # Align predictions with actual 2026 data
        # Predictions start from last training date (2021-12-31)
        # We need to match them with 2026 actual data
        
        # Get predictions for 2026 (approximately days 1000-1260 from the forecast)
        # Rough estimate: 2022-2025 = ~1000 trading days, so 2026 starts around day 1000
        
        # More accurate: calculate business days between 2021-12-31 and 2026-01-01
        from pandas.tseries.offsets import BDay
        train_end_date = pd.to_datetime(self.train_end)
        year_2026_start = pd.to_datetime('2026-01-01')
        
        # Approximate number of trading days between train_end and 2026
        days_to_2026 = len(pd.bdate_range(train_end_date, year_2026_start)) - 1
        
        print(f"   Training ended: {self.train_end}")
        print(f"   Days to 2026: ~{days_to_2026} trading days")
        
        # Get the portion of predictions corresponding to 2026
        available_2026_days = min(len(self.actual_2026_data), len(self.predictions) - days_to_2026)
        
        if available_2026_days <= 0:
            print("❌ Predictions don't extend far enough into 2026")
            return None
        
        pred_2026 = self.predictions.iloc[days_to_2026:days_to_2026+available_2026_days]['Predicted_Close'].values
        actual_2026 = self.actual_2026_data['Close'].iloc[:available_2026_days].values
        
        # Calculate error metrics
        mae = mean_absolute_error(actual_2026, pred_2026)
        rmse = np.sqrt(mean_squared_error(actual_2026, pred_2026))
        mape = np.mean(np.abs((actual_2026 - pred_2026) / actual_2026)) * 100
        
        # Directional accuracy (did we predict up/down correctly?)
        pred_direction = np.sign(np.diff(pred_2026))
        actual_direction = np.sign(np.diff(actual_2026))
        directional_accuracy = np.mean(pred_direction == actual_direction) * 100
        
        # Price comparison
        pred_final = pred_2026[-1]
        actual_final = actual_2026[-1]
        price_error_pct = ((pred_final - actual_final) / actual_final) * 100
        
        errors = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'predicted_final_price': pred_final,
            'actual_final_price': actual_final,
            'price_error_pct': price_error_pct,
            'days_compared': available_2026_days,
            'predictions_2026': pred_2026,
            'actual_2026': actual_2026
        }
        
        print(f"\n📊 ERROR METRICS:")
        print(f"   MAE (Mean Absolute Error): ${mae:.2f}")
        print(f"   RMSE (Root Mean Squared Error): ${rmse:.2f}")
        print(f"   MAPE (Mean Absolute % Error): {mape:.2f}%")
        print(f"   Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"\n   Predicted 2026 Final Price: ${pred_final:.2f}")
        print(f"   Actual 2026 Final Price: ${actual_final:.2f}")
        print(f"   Price Error: {price_error_pct:+.2f}%")
        
        return errors
    
    def plot_comparison(self, save_path='validation_results.png'):
        """Plot predicted vs actual 2026 prices"""
        if 'predictions_2026' not in self.results or 'actual_2026' not in self.results:
            print("❌ No comparison data to plot")
            return
        
        print(f"\n📊 Generating comparison plot...")
        
        pred_2026 = self.results['predictions_2026']
        actual_2026 = self.results['actual_2026']
        days = np.arange(len(pred_2026))
        
        plt.figure(figsize=(14, 7))
        plt.plot(days, actual_2026, label='Actual 2026 Prices', color='#00F0FF', linewidth=2)
        plt.plot(days, pred_2026, label='Predicted Prices', color='#FF9900', linewidth=2, linestyle='--')
        
        plt.fill_between(days, pred_2026, actual_2026, alpha=0.2, color='red')
        
        plt.title(f'{self.ticker} - 5-Year Forecast Validation (Trained on ≤2021, Predicted 2026)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Trading Days in 2026')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
        
        return plt
    
    def run_validation(self, use_ensemble=True, plot=True):
        """Run complete validation workflow"""
        print(f"\n{'='*60}")
        print(f"  5-YEAR FORECAST VALIDATION: {self.ticker}")
        print(f"{'='*60}")
        
        # Step 1: Fetch training data
        if self.fetch_training_data() is None:
            return None
        
        # Step 2: Generate prediction
        if self.generate_5year_prediction(use_ensemble=use_ensemble) is None:
            return None
        
        # Step 3: Fetch actual 2026 data
        if self.fetch_actual_2026_data() is None:
            return None
        
        # Step 4: Calculate errors
        errors = self.calculate_errors()
        if errors is None:
            return None
        
        self.results.update(errors)
        
        # Step 5: Plot comparison
        if plot:
            self.plot_comparison(f'{self.ticker}_validation.png')
        
        print(f"\n{'='*60}")
        print(f"  VALIDATION COMPLETE")
        print(f"{'='*60}\n")
        
        return self.results


def run_multi_stock_validation(tickers=['AAPL', 'TSLA', 'NVDA'], use_ensemble=True):
    """Run validation on multiple stocks and compare results"""
    print(f"\n🚀 Running validation on {len(tickers)} stocks...\n")
    
    all_results = {}
    
    for ticker in tickers:
        validator = ForecastValidator(ticker)
        results = validator.run_validation(use_ensemble=use_ensemble)
        
        if results:
            all_results[ticker] = {
                'mae': results['mae'],
                'rmse': results['rmse'],
                'mape': results['mape'],
                'directional_accuracy': results['directional_accuracy'],
                'price_error_pct': results['price_error_pct']
            }
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"  VALIDATION SUMMARY - ALL STOCKS")
    print(f"{'='*80}")
    print(f"{'Ticker':<10} {'MAE ($)':<12} {'RMSE ($)':<12} {'MAPE (%)':<12} {'Dir. Acc (%)':<15} {'Price Err (%)':<15}")
    print(f"{'-'*80}")
    
    for ticker, results in all_results.items():
        print(f"{ticker:<10} {results['mae']:<12.2f} {results['rmse']:<12.2f} "
              f"{results['mape']:<12.2f} {results['directional_accuracy']:<15.2f} "
              f"{results['price_error_pct']:<+15.2f}")
    
    print(f"{'='*80}\n")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate 5-year stock price forecasts')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--multi', action='store_true', help='Run on multiple stocks')
    parser.add_argument('--no-ensemble', action='store_true', help='Use LSTM only (no ensemble)')
    
    args = parser.parse_args()
    
    if args.multi:
        run_multi_stock_validation(use_ensemble=not args.no_ensemble)
    else:
        validator = ForecastValidator(args.ticker)
        validator.run_validation(use_ensemble=not args.no_ensemble)

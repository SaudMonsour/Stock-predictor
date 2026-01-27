"""
Example: Using the New OOP Interface for Stock Prediction
===========================================================

This file demonstrates how to use the refactored OOP interface
in model_engine.py for easy maintenance and future extensions.
"""

import model_engine as engine
from datetime import timedelta


# ==============================================================================
# EXAMPLE 1: Basic Usage - Simple Prediction
# ==============================================================================
def example_basic_prediction():
    """Basic usage example - get predictions for a stock"""
    
    # 1. Create engine instance
    stock_engine = engine.StockModelEngine('NVDA')
    
    # 2. Load data
    print("Loading data...")
    stock_data, market_data = stock_engine.load_data()
    
    # 3. Get financial metrics
    print("Calculating metrics...")
    metrics = stock_engine.get_metrics()
    print(f"CAGR: {metrics['CAGR']:.2%}")
    print(f"Volatility: {metrics['Volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe']:.2f}")
    
    # 4. Run prediction
    print("Running prediction...")
    results = stock_engine.run_prediction(
        forecast_days=252,  # 1 year
        initial_capital=10000
    )
    
    # 5. Display results
    print(f"Current Price: ${results['current_price']:.2f}")
    print(f"Predicted Price (1Y): ${results['final_price']:.2f}")
    print(f"Final Wealth: ${results['final_wealth']:.2f}")
    print(f"Projected CAGR: {results['projected_cagr']:.2%}")
    
    return results


# ==============================================================================
# EXAMPLE 2: Using Individual Components
# ==============================================================================
def example_individual_components():
    """Example showing how to use individual components separately"""
    
    ticker = 'AAPL'
    
    # Use StockDataManager independently
    data_manager = engine.StockDataManager(ticker)
    stock_data, market_data = data_manager.load_all_data()
    
    # Use MetricsCalculator independently
    metrics_calc = engine.MetricsCalculator(stock_data, market_data)
    cagr = metrics_calc.calculate_cagr()
    volatility = metrics_calc.calculate_volatility()
    beta = metrics_calc.calculate_beta()
    
    print(f"{ticker} Metrics:")
    print(f"  CAGR: {cagr:.2%}")
    print(f"  Volatility: {volatility:.2%}")
    print(f"  Beta: {beta:.2f}")
    
    return metrics_calc.get_all_metrics()


# ==============================================================================
# EXAMPLE 3: Custom Model Training
# ==============================================================================
def example_custom_training():
    """Example showing custom model training with specific parameters"""
    
    ticker = 'MSFT'
    
    # Load data
    data_manager = engine.StockDataManager(ticker)
    stock_data, _ = data_manager.load_all_data()
    
    # Create prediction engine
    pred_engine = engine.PredictionEngine(ticker)
    
    # Train with custom parameters
    print("Training model with custom parameters...")
    model = pred_engine.train(
        data=stock_data,
        epochs=15,  # More epochs for better training
        batch_size=32  # Smaller batch size
    )
    
    print("Model trained successfully!")
    
    return pred_engine


# ==============================================================================
# EXAMPLE 4: Advanced - Signal Generation
# ==============================================================================
def example_signal_generation():
    """Example showing how to generate trading signals"""
    
    # Create engine and run prediction
    stock_engine = engine.StockModelEngine('GOOGL')
    stock_engine.load_data()
    
    results = stock_engine.run_prediction(
        forecast_days=252,
        initial_capital=10000
    )
    
    # Get trading signals
    signals = results['signals']
    
    print("\nTrading Signals:")
    print("-" * 50)
    for period, (action, change) in signals.items():
        print(f"{period:12s}: {action:6s} ({change:+.2%})")
    
    return signals


# ==============================================================================
# EXAMPLE 5: Checking Model Persistence
# ==============================================================================
def example_model_persistence():
    """Example showing model persistence features"""
    
    ticker = 'AMD'
    
    # Create persistence manager
    persistence = engine.ModelPersistence(ticker)
    
    # Check if model exists
    if persistence.model_exists():
        print(f"Model exists for {ticker}")
        
        # Check age
        age = persistence.get_training_age()
        if age is not None:
            print(f"Model age: {age} days")
            print(f"Should retrain: {age >= 49}")
        
        # Load model
        model = persistence.load_model()
        scaler = persistence.load_scaler()
        metadata = persistence.load_metadata()
        
        print(f"Metadata: {metadata}")
    else:
        print(f"No saved model found for {ticker}")
    
    return persistence


# ==============================================================================
# EXAMPLE 6: Data Preprocessing
# ==============================================================================
def example_data_preprocessing():
    """Example showing data preprocessing capabilities"""
    
    # Create preprocessor
    preprocessor = engine.DataPreprocessor(time_step=60)
    
    # Load some data
    data_manager = engine.StockDataManager('NVDA')
    stock_data, _ = data_manager.load_all_data()
    
    # Preprocess
    df_close = stock_data[['Close']]
    scaled_data = preprocessor.scale_data(df_close)
    
    # Create sequences
    X, y = preprocessor.create_sequences(scaled_data)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Sequences created: {len(X)}")
    
    return preprocessor


# ==============================================================================
# EXAMPLE 7: Building Custom Models
# ==============================================================================
def example_custom_model():
    """Example showing how to build custom model architectures"""
    
    # Create model builder
    builder = engine.ModelBuilder(input_shape=(60, 1))
    
    # Build model
    model = builder.build_model()
    
    # Display model summary
    print("Model Architecture:")
    model.summary()
    
    return model


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("OOP Interface Examples for Stock Prediction Engine")
    print("=" * 70)
    
    # Run examples (comment out as needed)
    print("\n[EXAMPLE 1: Basic Prediction]")
    # example_basic_prediction()
    
    print("\n[EXAMPLE 2: Individual Components]")
    # example_individual_components()
    
    print("\n[EXAMPLE 3: Custom Training]")
    # example_custom_training()
    
    print("\n[EXAMPLE 4: Signal Generation]")
    # example_signal_generation()
    
    print("\n[EXAMPLE 5: Model Persistence]")
    # example_model_persistence()
    
    print("\n[EXAMPLE 6: Data Preprocessing]")
    # example_data_preprocessing()
    
    print("\n[EXAMPLE 7: Custom Model Building]")
    # example_custom_model()
    
    print("\n" + "=" * 70)
    print("Uncomment the examples you want to run!")
    print("=" * 70)

"""
Stock Prediction Model Engine - OOP Architecture
Comprehensive object-oriented design for maintainability and extensibility
"""

import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Bidirectional, 
                                     LSTM, Dense, Dropout, Activation, Multiply, 
                                     Permute, Lambda, RepeatVector, Flatten)
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta, datetime
import os
import pickle
import json
import streamlit as st
import joblib
from economic_data_fetcher import EconomicDataFetcher


# ============================================================================
# CONSTANTS
# ============================================================================
TICKERS = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMD', 'ASML', 'TSM', 'WMT', 'JPM', 'MCD', 'TM']
MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


# ============================================================================
# CLASS: StockDataManager
# ============================================================================
class StockDataManager:
    """Manages stock data fetching and caching"""
    
    def __init__(self, ticker, cache_ttl=3600):
        """
        Initialize data manager
        
        Args:
            ticker: Stock ticker symbol
            cache_ttl: Cache time-to-live in seconds
        """
        self.ticker = ticker
        self.cache_ttl = cache_ttl
        self.stock_data = None
        self.market_data = None
        self.economic_fetcher = EconomicDataFetcher()
    
    @st.cache_data(ttl=3600)
    def fetch_data(_self, ticker):
        """
        Fetch historical stock data with caching
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with historical stock data
        """
        try:
            # Fetch data up to today
            end_date = datetime.now().strftime('%Y-%m-%d')
            data = yf.download(ticker, start="2010-01-01", end=end_date, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            st.error(f"Error fetching stock data for {ticker}: {e}")
            return None
    
    @st.cache_data(ttl=3600)
    def fetch_market_data(_self):
        """
        Fetch S&P 500 data for beta calculation
        
        Returns:
            DataFrame with S&P 500 historical data
        """
        try:
            sp500 = yf.download("^GSPC", start="2010-01-01", end="2021-12-31", auto_adjust=True)
            if isinstance(sp500.columns, pd.MultiIndex):
                sp500.columns = sp500.columns.get_level_values(0)
            sp500.reset_index(inplace=True)
            return sp500
        except Exception as e:
            st.error(f"Error fetching market data: {e}")
            return None
    
    def load_all_data(self):
        """Load both stock and market data, plus economic indicators"""
        # 1. Fetch basic data
        self.stock_data = self.fetch_data(self.ticker)
        self.market_data = self.fetch_market_data()
        
        # 2. Add economic context
        if self.stock_data is not None:
            try:
                start_date = self.stock_data['Date'].iloc[0].strftime('%Y-%m-%d')
                end_date = self.stock_data['Date'].iloc[-1].strftime('%Y-%m-%d')
                
                indicators = self.economic_fetcher.fetch_all_indicators(start_date, end_date)
                
                if indicators is not None:
                    # Align and calculate derived features
                    aligned_data = self.economic_fetcher.align_with_trading_days(self.stock_data, indicators)
                    self.stock_data = self.economic_fetcher.calculate_derived_features(aligned_data)
                    print(f"✅ Enhanced data with {len(self.stock_data.columns) - 7} economic features")
            except Exception as e:
                print(f"⚠️ Economic data integration failed: {e}")
        
        return self.stock_data, self.market_data
    
    def get_dataframe(self):
        """Get the stock data DataFrame"""
        return self.stock_data


# ============================================================================
# CLASS: MetricsCalculator
# ============================================================================
class MetricsCalculator:
    """Calculates financial metrics for stock analysis"""
    
    def __init__(self, stock_data, market_data):
        """
        Initialize metrics calculator
        
        Args:
            stock_data: Historical stock price data
            market_data: Market (S&P 500) data for beta calculation
        """
        self.stock_data = stock_data
        self.market_data = market_data
        self.metrics = {}
    
    def calculate_cagr(self):
        """Calculate Compound Annual Growth Rate"""
        if self.stock_data is None:
            return 0.0
        
        df = self._prepare_dataframe(self.stock_data)
        if df.empty:
            return 0.0
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        cagr = (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (1 / years) - 1
        return cagr
    
    def calculate_volatility(self):
        """Calculate annualized volatility"""
        if self.stock_data is None:
            return 0.0
        
        df = self._prepare_dataframe(self.stock_data)
        if df.empty:
            return 0.0
        df['Daily_Return'] = df['Close'].pct_change()
        volatility = df['Daily_Return'].std() * np.sqrt(252)
        return volatility
    
    def calculate_beta(self):
        """Calculate beta coefficient vs market"""
        if self.stock_data is None or self.market_data is None:
            return 1.0
        
        df = self._prepare_dataframe(self.stock_data)
        sp_df = self._prepare_dataframe(self.market_data)
        
        if df.empty or sp_df.empty:
            return 1.0
        
        df['Daily_Return'] = df['Close'].pct_change()
        sp_df['Daily_Return'] = sp_df['Close'].pct_change()
        
        aligned_data = pd.concat([df['Daily_Return'], sp_df['Daily_Return']], axis=1).dropna()
        aligned_data.columns = ['Stock', 'Market']
        
        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data['Market'].var()
        beta = covariance / market_variance
        return beta
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.04):
        """
        Calculate Sharpe ratio
        
        Args:
            risk_free_rate: Annual risk-free rate (default 4%)
            
        Returns:
            Sharpe ratio value
        """
        cagr = self.calculate_cagr()
        volatility = self.calculate_volatility()
        
        if volatility == 0:
            return 0.0
        
        sharpe = (cagr - risk_free_rate) / volatility
        return sharpe
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        if self.stock_data is None:
            return 0.0
        
        df = self._prepare_dataframe(self.stock_data)
        if df.empty:
            return 0.0
        roll_max = df['Close'].cummax()
        daily_drawdown = df['Close'] / roll_max - 1.0
        mdd = daily_drawdown.min()
        return mdd
    
    def get_all_metrics(self):
        """
        Calculate and return all financial metrics
        
        Returns:
            Dictionary containing all metrics
        """
        if self.stock_data is None or self.market_data is None:
            return {
                "CAGR": 0.0,
                "Volatility": 0.0,
                "Beta": 0.0,
                "Sharpe": 0.0,
                "MDD": 0.0
            }
        
        self.metrics = {
            "CAGR": self.calculate_cagr(),
            "Volatility": self.calculate_volatility(),
            "Beta": self.calculate_beta(),
            "Sharpe": self.calculate_sharpe_ratio(),
            "MDD": self.calculate_max_drawdown()
        }
        return self.metrics
    
    @staticmethod
    def _prepare_dataframe(data):
        """Prepare DataFrame with Date as index"""
        df = data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df


# ============================================================================
# CLASS: DataPreprocessor
# ============================================================================
class DataPreprocessor:
    """Handles data preprocessing for model training"""
    
    def __init__(self, time_step=60):
        """
        Initialize data preprocessor
        
        Args:
            time_step: Number of time steps for sequence creation
        """
        self.time_step = time_step
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def scale_data(self, data):
        """
        Normalize data using MinMaxScaler
        
        Args:
            data: DataFrame or array to scale
            
        Returns:
            Scaled data array
        """
        scaled = self.scaler.fit_transform(data)
        return scaled
    
    def transform_data(self, data):
        """
        Transform data using existing scaler
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        return self.scaler.transform(data)
    
    def inverse_transform(self, predictions):
        """
        Convert scaled predictions back to original scale
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            Original scale predictions
        """
        return self.scaler.inverse_transform(predictions)
    
    def create_sequences(self, scaled_data):
        """
        Create X, Y sequences for LSTM training
        
        Args:
            scaled_data: Scaled input data
            
        Returns:
            Tuple of (X, y) arrays
        """
        dataX, dataY = [], []
        n_features = scaled_data.shape[1]
        for i in range(len(scaled_data) - self.time_step - 1):
            a = scaled_data[i:(i + self.time_step), :]
            dataX.append(a)
            dataY.append(scaled_data[i + self.time_step, 0]) # Predict only 'Close' price
        
        X = np.array(dataX)
        y = np.array(dataY)
        # X is already in (samples, time_step, features) shape
        return X, y
    
    def get_last_sequence(self, data):
        """
        Get last N time steps for prediction
        
        Args:
            data: Full dataset
            
        Returns:
            Last sequence reshaped for model input
        """
        last_sequence = data[-self.time_step:]
        return last_sequence.reshape(1, self.time_step, data.shape[1])
    
    def get_scaler(self):
        """Return the scaler object"""
        return self.scaler
    
    def set_scaler(self, scaler):
        """Set an existing scaler"""
        self.scaler = scaler


# ============================================================================
# CLASS: ModelBuilder
# ============================================================================
class ModelBuilder:
    """Builds CNN-BiLSTM-Attention neural network architecture"""
    
    def __init__(self, input_shape):
        """
        Initialize model builder
        
        Args:
            input_shape: Tuple of (time_steps, features)
        """
        self.input_shape = input_shape
        self.model = None
    
    def build_model(self):
        """
        Build the complete hybrid model architecture
        Implementation of CNN-BiLSTM-Attention
        Reference: Lu et al., "A CNN-BiLSTM-AM Method for Stock Price Prediction"
        
        Returns:
            Compiled Keras model
        """
        # Input Layer
        inputs = Input(shape=self.input_shape)
        
        # CNN Block - Feature Extraction
        x = self._add_cnn_layer(inputs)
        
        # Bi-Directional LSTM Block - Temporal Dependencies
        x = self._add_bilstm_layer(x)
        
        # Attention Mechanism
        sent_representation = self._add_attention_mechanism(x)
        
        # Output Block
        outputs = self._add_output_layer(sent_representation)
        
        # Create and compile model
        self.model = Model(inputs=inputs, outputs=outputs)
        self.compile_model()
        
        return self.model
    
    def _add_cnn_layer(self, inputs):
        """Add CNN feature extraction layer"""
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        return x
    
    def _add_bilstm_layer(self, x):
        """Add Bidirectional LSTM layer"""
        x = Bidirectional(LSTM(units=128, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        return x
    
    def _add_attention_mechanism(self, x):
        """Add attention mechanism layer"""
        # Compute attention weights
        attention = Dense(1, activation='tanh')(x)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(256)(attention)  # 2*128 for BiLSTM
        attention = Permute([2, 1])(attention)
        
        # Apply attention weights
        sent_representation = Multiply()([x, attention])
        sent_representation = Lambda(self._attention_sum)(sent_representation)
        
        return sent_representation
    
    def _add_output_layer(self, x):
        """Add final dense output layers"""
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)  # Linear activation for regression
        return outputs
    
    @staticmethod
    def _attention_sum(x):
        """Custom layer for attention summation"""
        return tf.reduce_sum(x, axis=1)
    
    def compile_model(self, optimizer='adam', loss='mean_squared_error'):
        """
        Compile the model
        
        Args:
            optimizer: Optimizer to use
            loss: Loss function
        """
        if self.model is not None:
            self.model.compile(optimizer=optimizer, loss=loss)
    
    def get_model(self):
        """Return the built model"""
        return self.model


# ============================================================================
# CLASS: ModelPersistence
# ============================================================================
class ModelPersistence:
    """Handles saving and loading of model artifacts"""
    
    def __init__(self, ticker, model_dir=MODEL_DIR):
        """
        Initialize persistence manager
        
        Args:
            ticker: Stock ticker symbol
            model_dir: Directory for saving models
        """
        self.ticker = ticker
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, f"{ticker}_model.keras")
        self.scaler_path = os.path.join(model_dir, f"{ticker}_scaler.pkl")
        self.metadata_path = os.path.join(model_dir, f"{ticker}_metadata.json")
    
    def save_model(self, model):
        """Save Keras model"""
        try:
            model.save(self.model_path)
        except Exception as e:
            st.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load Keras model"""
        if not self.model_exists():
            return None
        
        try:
            model = load_model(self.model_path, 
                             custom_objects={'_attention_sum': ModelBuilder._attention_sum},
                             safe_mode=False)
            return model
        except Exception as e:
            st.warning(f"Could not load model: {e}")
            return None
    
    def save_scaler(self, scaler):
        """Persist scaler object"""
        try:
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        except Exception as e:
            st.error(f"Error saving scaler: {e}")
    
    def load_scaler(self):
        """Load scaler object"""
        if not os.path.exists(self.scaler_path):
            return None
        
        try:
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            return scaler
        except Exception as e:
            st.warning(f"Could not load scaler: {e}")
            return None
    
    def save_metadata(self, info):
        """
        Save training metadata
        
        Args:
            info: Dictionary of metadata to save
        """
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(info, f)
        except Exception as e:
            st.error(f"Error saving metadata: {e}")
    
    def load_metadata(self):
        """Load training metadata"""
        if not os.path.exists(self.metadata_path):
            return None
        
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            st.warning(f"Could not load metadata: {e}")
            return None
    
    def model_exists(self):
        """Check if model artifacts exist"""
        return (os.path.exists(self.model_path) and 
                os.path.exists(self.scaler_path))
    
    def get_training_age(self):
        """
        Get age of current model in days
        
        Returns:
            Days since training, or None if no metadata
        """
        metadata = self.load_metadata()
        if metadata is None:
            return None
        
        training_date_str = metadata.get("training_date")
        if training_date_str:
            training_date = datetime.strptime(training_date_str, "%Y-%m-%d")
            age = datetime.now() - training_date
            return age.days
        
        return None


# ============================================================================
# CLASS: PredictionEngine
# ============================================================================
class PredictionEngine:
    """Handles model training and predictions"""
    
    def __init__(self, ticker, model_dir=MODEL_DIR):
        """
        Initialize prediction engine
        
        Args:
            ticker: Stock ticker symbol
            model_dir: Directory for model storage
        """
        self.ticker = ticker
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = DataPreprocessor(time_step=60)
        self.persistence = ModelPersistence(ticker, model_dir)
    
    def train(self, data, epochs=10, batch_size=64):
        """
        Train a new model
        
        Args:
            data: Stock price data
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Trained model
        """
        # Filter last 5 years of data
        five_years_ago = data['Date'].iloc[-1] - timedelta(days=5*365)
        df_filtered = data[data['Date'] > five_years_ago].copy()
        
        # Prepare data - select appropriate features
        # Always put 'Close' first so it's at index 0
        feature_cols = ['Close']
        
        # Add economic features if available
        potential_features = [
            'interest_rate', 'treasury_10y', 'inflation', 'gdp', 
            'unemployment', 'vix', 'market_return',
            'inflation_change', 'real_interest_rate', 'high_volatility'
        ]
        
        for col in potential_features:
            if col in df_filtered.columns:
                feature_cols.append(col)
        
        print(f"🧠 Training with features: {feature_cols}")
        
        df_features = df_filtered[feature_cols]
        scaled_data = self.preprocessor.scale_data(df_features)
        X, y = self.preprocessor.create_sequences(scaled_data)
        
        # Build and train model
        # Input shape is (time_steps, n_features)
        n_features = X.shape[2]
        builder = ModelBuilder(input_shape=(X.shape[1], n_features))
        self.model = builder.build_model()
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
        
        # Save artifacts
        self.persistence.save_model(self.model)
        self.persistence.save_scaler(self.preprocessor.get_scaler())
        feature_info = {
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "features": feature_cols
        }
        self.persistence.save_metadata(feature_info)
        
        return self.model
    
    def load_existing_model(self):
        """
        Load saved model if exists
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.model = self.persistence.load_model()
            scaler = self.persistence.load_scaler()
            
            if self.model is not None and scaler is not None:
                self.preprocessor.set_scaler(scaler)
                return True
            
            return False
        except Exception as e:
            # If loading fails for any reason, return False to trigger retraining
            print(f"Failed to load existing model: {e}. Will retrain.")
            return False
    
    def predict_future(self, days, initial_sequence):
        """
        Generate future predictions recursively
        
        Args:
            days: Number of days to predict
            initial_sequence: Starting sequence for prediction
            
        Returns:
            Array of predicted prices
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        future_predictions = []
        current_sequence = initial_sequence.copy()
        n_features = current_sequence.shape[2]
        
        for _ in range(days):
            # Predict next step with optimized inference
            next_step = self.model(current_sequence, training=False).numpy()
            predicted_price = next_step[0, 0]
            future_predictions.append(predicted_price)
            
            # Construct next input vector
            if n_features > 1:
                # If we have multiple features (Price + Economics),
                # use predicted price for feature 0, and repeat last values for others (Persistence)
                last_exog = current_sequence[0, -1, 1:] # Get last known exogenous values
                next_vector = np.concatenate([[predicted_price], last_exog])
                next_vector = next_vector.reshape(1, 1, n_features)
            else:
                next_vector = next_step.reshape(1, 1, 1)
                
            # Update sequence: remove first, add new prediction
            current_sequence = np.append(current_sequence[:, 1:, :], 
                                        next_vector, axis=1)
        
        # Inverse transform to original scale
        # Create dummy array with same shape as training data for inverse transform
        # We only care about the first column (Close price)
        dummy_array = np.zeros((len(future_predictions), n_features))
        dummy_array[:, 0] = np.array(future_predictions).flatten()
        
        # Inverse transform and extract just the price column
        rescalsed = self.preprocessor.scaler.inverse_transform(dummy_array)
        final_predictions = rescalsed[:, 0].reshape(-1, 1)
        
        return final_predictions
    
    def should_retrain(self, max_age_days=49):
        """
        Determine if model should be retrained
        
        Args:
            max_age_days: Maximum age before retraining (default 7 weeks)
            
        Returns:
            True if should retrain, False otherwise
        """
        if not self.persistence.model_exists():
            return True
        
        # Try to load the model - if it fails, we need to retrain
        try:
            test_model = self.persistence.load_model()
            if test_model is None:
                return True
        except:
            # If loading fails (e.g., Lambda layer issues), retrain
            return True
        
        age = self.persistence.get_training_age()
        if age is None or age >= max_age_days:
            return True
        
        return False
    
    def get_model(self):
        """Return the current model"""
        return self.model
    
    def get_preprocessor(self):
        """Return the preprocessor"""
        return self.preprocessor


# ============================================================================
# CLASS: SignalGenerator
# ============================================================================
class SignalGenerator:
    """Generates trading signals based on predictions"""
    
    def __init__(self, current_price):
        """
        Initialize signal generator
        
        Args:
            current_price: Current stock price
        """
        self.current_price = current_price
        self.signals = {}
    
    def generate_signals(self, future_prices):
        """
        Create buy/sell/hold signals for various timeframes
        
        Args:
            future_prices: Array of predicted future prices
            
        Returns:
            Dictionary of signals by timeframe
        """
        intervals = {
            "1 Day": 1,
            "5 Days": 5,
            "1 Month": 21,
            "6 Months": 126,
            "1 Year": 252
        }
        
        for label, days in intervals.items():
            if days < len(future_prices):
                future_price = future_prices[days-1][0]
                change = (future_price - self.current_price) / self.current_price
                action = self._determine_action(change)
                self.signals[label] = (action, change)
            else:
                self.signals[label] = ("N/A", 0.0)
        
        return self.signals
    
    def get_signal_for_period(self, days):
        """
        Get signal for specific number of days
        
        Args:
            days: Number of days ahead
            
        Returns:
            Tuple of (action, change_percentage)
        """
        period_labels = {
            1: "1 Day",
            5: "5 Days",
            21: "1 Month",
            126: "6 Months",
            252: "1 Year"
        }
        
        label = period_labels.get(days)
        if label and label in self.signals:
            return self.signals[label]
        
        return ("N/A", 0.0)
    
    @staticmethod
    def _determine_action(change):
        """
        Determine buy/sell/hold action based on price change
        
        Args:
            change: Percentage change in price
            
        Returns:
            "BUY", "SELL", or "HOLD"
        """
        if change > 0.02:  # More than 2% increase
            return "BUY"
        elif change < -0.02:  # More than 2% decrease
            return "SELL"
        else:
            return "HOLD"


# ============================================================================
# MAIN FACADE CLASS: StockModelEngine
# ============================================================================
class StockModelEngine:
    """
    Main interface combining all components
    Provides a clean API for stock prediction workflow
    """
    
    def __init__(self, ticker, model_dir=MODEL_DIR):
        """
        Initialize the complete stock model engine
        
        Args:
            ticker: Stock ticker symbol
            model_dir: Directory for model storage
        """
        self.ticker = ticker
        self.model_dir = model_dir
        
        # Initialize components
        self.data_manager = StockDataManager(ticker)
        self.metrics_calculator = None
        self.prediction_engine = PredictionEngine(ticker, model_dir)
        self.signal_generator = None
        
        # Data storage
        self.stock_data = None
        self.market_data = None
        self.current_price = None
    
    def load_data(self):
        """
        Fetch stock and market data
        
        Returns:
            Tuple of (stock_data, market_data)
        """
        self.stock_data, self.market_data = self.data_manager.load_all_data()
        
        if self.stock_data is not None:
            self.current_price = self.stock_data['Close'].iloc[-1]
            self.metrics_calculator = MetricsCalculator(self.stock_data, self.market_data)
        
        return self.stock_data, self.market_data
    
    def get_metrics(self):
        """
        Calculate all financial metrics
        
        Returns:
            Dictionary of metrics
        """
        if self.metrics_calculator is None:
            return {}
        
        return self.metrics_calculator.get_all_metrics()
    
    def run_prediction(self, forecast_days, initial_capital=10000):
        """
        Execute complete prediction pipeline
        
        Args:
            forecast_days: Number of days to forecast
            initial_capital: Initial investment amount
            
        Returns:
            Dictionary containing prediction results
        """
        if self.stock_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Train or load model
        if self.prediction_engine.should_retrain():
            self.prediction_engine.train(self.stock_data)
            trained_new = True
        else:
            self.prediction_engine.load_existing_model()
            trained_new = False
        
        # Prepare data for prediction
        # Select appropriate features matching training logic
        feature_cols = ['Close']
        potential_features = [
            'interest_rate', 'treasury_10y', 'inflation', 'gdp', 
            'unemployment', 'vix', 'market_return',
            'inflation_change', 'real_interest_rate', 'high_volatility'
        ]
        
        # Check which features exist in valid columns
        # Note: We should ideally save the exact feature list in metadata
        # giving best-effort here based on available columns
        for col in potential_features:
            if col in self.stock_data.columns:
                feature_cols.append(col)
                
        # If we loaded an existing model, verify features match what scaler expects
        if not trained_new:
            # Try to check metadata for exact features
            metadata = self.prediction_engine.persistence.load_metadata()
            if metadata and "features" in metadata:
                saved_features = metadata["features"]
                # Ensure we have all needed features
                missing = [f for f in saved_features if f not in self.stock_data.columns]
                if not missing:
                    feature_cols = saved_features
                else:
                    print(f"⚠️ Missing features for loaded model: {missing}. Force retraining.")
                    # Force retrain if data shape doesn't match
                    self.prediction_engine.train(self.stock_data)
                    trained_new = True
                    # Re-determine feature cols after training
                    feature_cols = ['Close']
                    for col in potential_features:
                        if col in self.stock_data.columns:
                            feature_cols.append(col)
        
        df_features = self.stock_data[feature_cols]
        preprocessor = self.prediction_engine.get_preprocessor()
        
        # Ensure scaler is fitted with current data
        if not trained_new:
            try:
                scaled_data = preprocessor.transform_data(df_features)
            except Exception as e:
                print(f"⚠️ Transform failed: {e}. Refitting scaler.")
                scaled_data = preprocessor.scale_data(df_features)
        else:
            scaled_data = preprocessor.transform_data(df_features)
        
        last_sequence = preprocessor.get_last_sequence(scaled_data)
        
        # CRITICAL CHECK: Verify model input shape matches data shape
        # This prevents the "expected axis -1 of input shape to have value 1, but received 11" error
        model = self.prediction_engine.get_model()
        if model is not None:
            # Model input shape is typically (None, time_steps, n_features)
            # We check the last dimension (n_features)
            expected_features = model.input_shape[-1]
            actual_features = last_sequence.shape[-1]
            
            if expected_features != actual_features:
                print(f"⚠️ Shape mismatch! Model expects {expected_features} features, but data has {actual_features}.")
                print("🔄 Forcing model retraining with correct features...")
                
                # Force retrain
                self.prediction_engine.train(self.stock_data)
                trained_new = True
                
                # Re-process data with new model/scaler
                # The scaler would have been updated during training
                scaled_data = preprocessor.transform_data(df_features)
                last_sequence = preprocessor.get_last_sequence(scaled_data)
        
        # Generate predictions
        future_prices = self.prediction_engine.predict_future(forecast_days, last_sequence)

        
        # Create prediction DataFrame
        last_date = self.stock_data['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=len(future_prices), freq='B')
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_prices.flatten()
        })
        
        # Calculate results
        final_price = pred_df['Predicted_Close'].iloc[-1]
        growth_factor = final_price / self.current_price
        final_wealth = initial_capital * growth_factor
        
        # Calculate projected CAGR
        years_fraction = forecast_days / 252
        if years_fraction > 0:
            projected_cagr = (final_price / self.current_price) ** (1/years_fraction) - 1
        else:
            projected_cagr = 0.0
        
        # Generate trading signals
        self.signal_generator = SignalGenerator(self.current_price)
        signals = self.signal_generator.generate_signals(future_prices)
        
        return {
            "trained_new": trained_new,
            "predictions": pred_df,
            "current_price": self.current_price,
            "final_price": final_price,
            "final_wealth": final_wealth,
            "projected_cagr": projected_cagr,
            "signals": signals
        }
    
    def get_current_price(self):
        """Get latest stock price"""
        return self.current_price
    
    def get_signals(self):
        """Get trading signals"""
        if self.signal_generator:
            return self.signal_generator.signals
        return {}
    
    def get_stock_data(self):
        """Get stock data DataFrame"""
        return self.stock_data
    
    def get_market_data(self):
        """Get market data DataFrame"""
        return self.market_data
    
    # ========================================================================
    # ENHANCED FEATURES - New capabilities
    # ========================================================================
    
    def get_confidence_interval(self, predicted_price: float, days_ahead: int):
        """
        Calculate confidence interval for prediction
        
        Args:
            predicted_price: Model's predicted price
            days_ahead: Forecast horizon in days
            
        Returns:
            Dictionary with confidence bounds and classification
        """
        from confidence_calculator import ConfidenceCalculator
        
        metrics = self.get_metrics()
        volatility = metrics.get('Volatility', 0.25)
        
        calculator = ConfidenceCalculator()
        return calculator.calculate_from_volatility(
            predicted_price, volatility, days_ahead
        )
    
    def get_risk_profile(self):
        """
        Get risk classification for this stock
        
        Returns:
            Dictionary with risk level and explanation
        """
        from risk_analyzer import RiskAnalyzer
        
        metrics = self.get_metrics()
        analyzer = RiskAnalyzer()
        
        risk_level = analyzer.classify_risk(metrics)
        warnings = analyzer.detect_warnings(self.stock_data, metrics)
        is_reliable, reason = analyzer.is_prediction_reliable(self.stock_data, metrics)
        explanation = analyzer.get_risk_explanation(risk_level, metrics)
        
        return {
            'risk_level': risk_level,
            'warnings': warnings,
            'is_reliable': is_reliable,
            'reliability_reason': reason,
            'explanation': explanation
        }
    
    def get_prediction_explanation(self, prediction_change: float, confidence_level: str):
        """
        Generate natural language explanation for prediction
        
        Args:
            prediction_change: Predicted price change percentage
            confidence_level: Confidence classification
            
        Returns:
            Human-readable explanation string
        """
        from explanation_generator import ExplanationGenerator
        
        metrics = self.get_metrics()
        generator = ExplanationGenerator()
        
        return generator.generate_explanation(
            prediction_change, self.stock_data, metrics, confidence_level
        )
    
    def run_ensemble_prediction(self, forecast_days, initial_capital=10000, use_ensemble=False):
        """
        Execute prediction with ensemble models and enhanced features
        
        Args:
            forecast_days: Number of days to forecast
            initial_capital: Initial investment amount
            use_ensemble: Whether to use ensemble (True) or just LSTM (False)
            
        Returns:
            Enhanced dictionary with confidence, risk, and explanations
        """
        # Get basic LSTM prediction first
        basic_results = self.run_prediction(forecast_days, initial_capital)
        
        # Get LSTM predicted price
        lstm_price = basic_results['final_price']
        
        # If ensemble requested, combine with other models
        if use_ensemble:
            try:
                from prediction_ensemble import EnsemblePredictor
                
                # Train ensemble models
                ensemble = EnsemblePredictor(self.ticker)
                ensemble.train_all_models(self.stock_data)
                
                # Get ensemble prediction
                ensemble_result = ensemble.predict_ensemble(
                    self.stock_data, lstm_price, forecast_days
                )
                
                final_price = ensemble_result['ensemble_prediction']
                basic_results['ensemble_details'] = ensemble_result
                basic_results['model_performance'] = ensemble.get_model_performance()
            except Exception as e:
                # Fallback to LSTM if ensemble fails
                print(f"Ensemble prediction failed: {e}. Using LSTM only.")
                final_price = lstm_price
        else:
            final_price = lstm_price
        
        # Calculate confidence interval
        confidence_data = self.get_confidence_interval(final_price, forecast_days)
        
        # Get risk profile
        risk_profile = self.get_risk_profile()
        
        # Calculate prediction change
        prediction_change = (final_price - self.current_price) / self.current_price
        
        # Generate explanation
        explanation = self.get_prediction_explanation(
            prediction_change,
            confidence_data['confidence_level']
        )
        
        # Override final price with Best Case (Upper Confidence Bound) as requested
        # "always the best case is the correct one"
        final_price = confidence_data['confidence_high']
        
        # Recalculate wealth based on Best Case
        growth_factor = final_price / self.current_price
        final_wealth = initial_capital * growth_factor
        
        # Calculate best/worst case scenarios for the results dictionary
        best_case_wealth = initial_capital * (confidence_data['confidence_high'] / self.current_price)
        worst_case_wealth = initial_capital * (confidence_data['confidence_low'] / self.current_price)
        
        # Update results with enhanced features
        basic_results.update({
            'final_price': final_price,  # Now using Best Case
            'final_wealth': final_wealth,
            'confidence': confidence_data,
            'risk_profile': risk_profile,
            'explanation': explanation,
            'investment_scenarios': {
                'expected': final_wealth,
                'best_case': best_case_wealth,
                'worst_case': worst_case_wealth,
                'expected_return_pct': prediction_change * 100,
                'best_case_return_pct': ((confidence_data['confidence_high'] - self.current_price) / self.current_price) * 100,
                'worst_case_return_pct': ((confidence_data['confidence_low'] - self.current_price) / self.current_price) * 100
            }
        })
        
        return basic_results




# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS (for existing app.py)
# ============================================================================
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    """Legacy function for backward compatibility"""
    manager = StockDataManager(ticker)
    return manager.load_all_data()


def calculate_metrics(data, sp500):
    """Legacy function for backward compatibility"""
    calculator = MetricsCalculator(data, sp500)
    metrics = calculator.get_all_metrics()
    # Ensure we never return empty dict for legacy calls
    if not metrics:
        return {
            "CAGR": 0.0,
            "Volatility": 0.0,
            "Beta": 0.0,
            "Sharpe": 0.0,
            "MDD": 0.0
        }
    return metrics


def train_or_load_model(ticker, data):
    """Legacy function for backward compatibility"""
    engine = PredictionEngine(ticker)
    
    # Check if we should retrain (based on age or missing files)
    should_train = engine.should_retrain()
    
    # If we think we shouldn't retrain, try loading
    if not should_train:
        if not engine.load_existing_model():
            # If loading fails (e.g. corrupted files), force retraining
            should_train = True
    
    if should_train:
        model = engine.train(data)
        scaler = engine.get_preprocessor().get_scaler()
        return model, scaler, True
    else:
        model = engine.get_model()
        scaler = engine.get_preprocessor().get_scaler()
        return model, scaler, False


def predict_future(model, last_sequence, future_steps, scaler):
    """Legacy function for backward compatibility"""
    # Create temporary engine
    temp_engine = PredictionEngine("TEMP")
    temp_engine.model = model
    temp_engine.preprocessor.set_scaler(scaler)
    
    return temp_engine.predict_future(future_steps, last_sequence)

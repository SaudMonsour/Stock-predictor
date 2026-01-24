"""
Ensemble Prediction System
Combines multiple models for improved forecast accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


class EnsemblePredictor:
    """Multi-model ensemble for stock price prediction"""
    
    def __init__(self, ticker: str):
        """
        Initialize ensemble predictor
        
        Args:
            ticker: Stock ticker symbol
        """
        self.ticker = ticker
        self.models = {}
        self.weights = {}
        self.performance = {}
        self.trained = False
    
    def train_all_models(
        self,
        stock_data: pd.DataFrame,
        time_steps: int = 60
    ) -> Dict:
        """
        Train all models in the ensemble
        
        Args:
            stock_data: Historical stock price data
            time_steps: Lookback window for features
            
        Returns:
            Training performance metrics
        """
        # Prepare training data
        X, y = self._prepare_features(stock_data, time_steps)
        
        if len(X) < 100:
            raise ValueError("Insufficient data for training ensemble models")
        
        # Split train/validation (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train Linear Regression
        print("Training Linear Regression...")
        lr_model =LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_val)
        lr_mae = mean_absolute_error(y_val, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_val, lr_pred))
        
        self.models['linear'] = lr_model
        self.performance['linear'] = {
            'MAE': lr_mae,
            'RMSE': lr_rmse
        }
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_val)
        rf_mae = mean_absolute_error(y_val, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
        
        self.models['random_forest'] = rf_model
        self.performance['random_forest'] = {
            'MAE': rf_mae,
            'RMSE': rf_rmse
        }
        
        # Calculate weights inversely proportional to MAE
        # Better models (lower MAE) get higher weights
        mae_values = [lr_mae, rf_mae]
        inverse_mae = [1/mae if mae > 0 else 1 for mae in mae_values]
        total_inverse = sum(inverse_mae)
        
        self.weights['linear'] = inverse_mae[0] / total_inverse
        self.weights['random_forest'] = inverse_mae[1] / total_inverse
        
        # Reserve weight for LSTM (will be combined later)
        # For now, traditional models get 40% weight, LSTM will get 60%
        self.weights['linear'] *= 0.4
        self.weights['random_forest'] *= 0.4
        self.weights['lstm'] = 0.6  # Placeholder for deep learning model
        
        self.trained = True
        
        return self.performance
    
    def predict_ensemble(
        self,
        stock_data: pd.DataFrame,
        lstm_prediction: float,
        days_ahead: int = 30,
        time_steps: int = 60
    ) -> Dict:
        """
        Generate ensemble prediction
        
        Args:
            stock_data: Historical data
            lstm_prediction: Prediction from existing LSTM model
            days_ahead: Forecast horizon
            time_steps: Feature window
            
        Returns:
            Dictionary with ensemble and individual predictions
        """
        if not self.trained:
            raise ValueError("Models must be trained first")
        
        # Prepare features from recent data
        X_latest = self._prepare_single_sequence(stock_data, time_steps)
        
        # Get predictions from each model
        lr_pred = self.models['linear'].predict(X_latest)[0]
        rf_pred = self.models['random_forest'].predict(X_latest)[0]
        
        # Scale predictions by forecast horizon
        # Models predict next-step; scale for multi-day forecast
        current_price = stock_data['Close'].iloc[-1]
        
        lr_change = (lr_pred - current_price) / current_price
        rf_change = (rf_pred - current_price) / current_price
        lstm_change = (lstm_prediction - current_price) / current_price
        
        # Dampen changes for longer horizons (realism adjustment)
        horizon_factor = min(1.0 + (days_ahead / 252) * 0.5, 2.0)
        
        lr_final = current_price * (1 + lr_change * horizon_factor)
        rf_final = current_price * (1 + rf_change * horizon_factor)
        lstm_final = lstm_prediction
        
        # Calculate weighted ensemble
        ensemble_pred = (
            lr_final * self.weights['linear'] +
            rf_final * self.weights['random_forest'] +
            lstm_final * self.weights['lstm']
        )
        
        return {
            'ensemble_prediction': round(ensemble_pred, 2),
            'individual_predictions': {
                'linear_regression': round(lr_final, 2),
                'random_forest': round(rf_final, 2),
                'lstm': round(lstm_final, 2)
            },
            'model_weights': self.weights.copy(),
            'days_ahead': days_ahead
        }
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all models"""
        return self.performance.copy()
    
    def get_weights(self) -> Dict:
        """Get current model weights"""
        return self.weights.copy()
    
    def get_best_model(self) -> Tuple[str, Dict]:
        """
        Identify best performing model
        
        Returns:
            (model_name, performance_metrics) tuple
        """
        if not self.performance:
            return None, {}
        
        # Find model with lowest MAE
        best_model = min(
            self.performance.items(),
            key=lambda x: x[1]['MAE']
        )
        
        return best_model
    
    # Private helper methods
    
    def _prepare_features(
        self,
        stock_data: pd.DataFrame,
        time_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target for sklearn models
        
        Args:
            stock_data: Historical data
            time_steps: Number of previous days to use as features
            
        Returns:
            (X, y) arrays for training
        """
        prices = stock_data['Close'].values
        
        X, y = [], []
        for i in range(time_steps, len(prices)):
            # Features: previous N days of prices
            X.append(prices[i-time_steps:i])
            # Target: next day's price
            y.append(prices[i])
        
        return np.array(X), np.array(y)
    
    def _prepare_single_sequence(
        self,
        stock_data: pd.DataFrame,
        time_steps: int
    ) -> np.ndarray:
        """Prepare single sequence for prediction"""
        prices = stock_data['Close'].values[-time_steps:]
        return prices.reshape(1, -1)


# Convenience function

def create_ensemble(
    ticker: str,
    stock_data: pd.DataFrame
) -> EnsemblePredictor:
    """
    Create and train ensemble predictor
    
    Args:
        ticker: Stock ticker
        stock_data: Historical data
        
    Returns:
        Trained EnsemblePredictor instance
    """
    predictor = EnsemblePredictor(ticker)
    predictor.train_all_models(stock_data)
    return predictor

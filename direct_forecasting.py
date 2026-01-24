"""
Direct Multi-Step Forecasting Module

Implements direct forecasting for multiple time horizons without naive scaling.
Each horizon gets its own trained model to avoid error compounding.

Author: Stock Prediction Dashboard
Created: 2026-01-24
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


class DirectMultiStepForecaster:
    """
    Trains separate models for each forecast horizon to eliminate naive scaling.
    
    Horizons supported: 1d, 7d, 30d, 180d, 365d
    
    Example:
        forecaster = DirectMultiStepForecaster()
        forecaster.fit(X_train, y_train, dates_train)
        predictions = forecaster.predict(X_test, horizon='30d')
    """
    
    SUPPORTED_HORIZONS = {
        '1d': 1,
        '7d': 7,
        '30d': 30,
        '180d': 180,
        '365d': 365
    }
    
    def __init__(self, model_type='ridge'):
        """
        Initialize direct forecaster
        
        Args:
            model_type: 'ridge' or 'random_forest'
        """
        self.model_type = model_type
        self.models = {}  # One model per horizon
        self.scalers = {}  # One scaler per horizon
        self.trained_horizons = set()
        
    def _create_model(self):
        """Create a new model instance"""
        if self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_horizon_targets(
        self, 
        y: pd.Series, 
        dates: pd.Series, 
        horizon_days: int
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Create target shifted by horizon days
        
        Args:
            y: Target series (prices)
            dates: Date series aligned with y
            horizon_days: Number of days to shift
            
        Returns:
            Tuple of (valid_y, horizon_targets)
        """
        # Convert to DataFrame for easier handling
        df = pd.DataFrame({
            'date': dates,
            'price': y
        })
        df = df.sort_values('date').reset_index(drop=True)
        
        # Shift target by horizon
        df['target_h'] = df['price'].shift(-horizon_days)
        
        # Drop NaN values
        df_valid = df.dropna()
        
        return df_valid['price'], df_valid['target_h']
    
    def fit(
        self, 
        X: np.ndarray, 
        y: pd.Series, 
        dates: pd.Series,
        horizons: List[str] = None
    ):
        """
        Train models for each horizon
        
        Args:
            X: Feature matrix (last 60 days)
            y: Target series (current prices)
            dates: Date series aligned with X and y
            horizons: List of horizons to train (e.g., ['1d', '7d', '30d'])
                     If None, trains all supported horizons
        """
        if horizons is None:
            horizons = list(self.SUPPORTED_HORIZONS.keys())
        
        for horizon_name in horizons:
            if horizon_name not in self.SUPPORTED_HORIZONS:
                raise ValueError(f"Unsupported horizon: {horizon_name}")
            
            horizon_days = self.SUPPORTED_HORIZONS[horizon_name]
            
            # Create horizon-specific targets
            y_current, y_horizon = self._create_horizon_targets(
                y, dates, horizon_days
            )
            
            # Align X with valid indices
            valid_indices = y_current.index
            X_valid = X[valid_indices] if isinstance(X, np.ndarray) else X.iloc[valid_indices]
            
            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_valid)
            
            # Train model
            model = self._create_model()
            model.fit(X_scaled, y_horizon)
            
            # Store
            self.models[horizon_name] = model
            self.scalers[horizon_name] = scaler
            self.trained_horizons.add(horizon_name)
            
            print(f"✓ Trained {self.model_type} for horizon {horizon_name} "
                  f"({len(X_scaled)} samples)")
    
    def predict(
        self, 
        X: np.ndarray, 
        horizon: str = '30d'
    ) -> np.ndarray:
        """
        Predict for specified horizon
        
        Args:
            X: Feature matrix
            horizon: Horizon to predict (e.g., '30d')
            
        Returns:
            Array of predictions
        """
        if horizon not in self.trained_horizons:
            raise ValueError(
                f"Model not trained for horizon {horizon}. "
                f"Trained horizons: {self.trained_horizons}"
            )
        
        # Scale features using horizon-specific scaler
        X_scaled = self.scalers[horizon].transform(X)
        
        # Predict
        predictions = self.models[horizon].predict(X_scaled)
        
        return predictions
    
    def get_validation_error(
        self,
        X_val: np.ndarray,
        y_val: pd.Series,
        dates_val: pd.Series,
        horizon: str
    ) -> Dict[str, float]:
        """
        Compute validation metrics for a horizon
        
        Returns:
            Dict with MAE, RMSE, MAPE
        """
        if horizon not in self.trained_horizons:
            raise ValueError(f"Model not trained for horizon {horizon}")
        
        horizon_days = self.SUPPORTED_HORIZONS[horizon]
        
        # Create horizon targets for validation
        y_current, y_horizon_true = self._create_horizon_targets(
            y_val, dates_val, horizon_days
        )
        
        # Align X
        valid_indices = y_current.index
        X_val_aligned = X_val[valid_indices] if isinstance(X_val, np.ndarray) else X_val.iloc[valid_indices]
        
        # Predict
        y_pred = self.predict(X_val_aligned, horizon)
        
        # Calculate metrics
        mae = np.mean(np.abs(y_horizon_true - y_pred))
        rmse = np.sqrt(np.mean((y_horizon_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_horizon_true - y_pred) / y_horizon_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def save(self, save_dir: str, ticker: str):
        """Save all models and scalers"""
        os.makedirs(save_dir, exist_ok=True)
        
        for horizon in self.trained_horizons:
            model_path = os.path.join(save_dir, f"{ticker}_{horizon}_{self.model_type}.pkl")
            scaler_path = os.path.join(save_dir, f"{ticker}_{horizon}_scaler.pkl")
            
            joblib.dump(self.models[horizon], model_path)
            joblib.dump(self.scalers[horizon], scaler_path)
        
        print(f"✓ Saved {len(self.trained_horizons)} models for {ticker}")
    
    def load(self, save_dir: str, ticker: str, horizons: List[str] = None):
        """Load models and scalers"""
        if horizons is None:
            horizons = list(self.SUPPORTED_HORIZONS.keys())
        
        for horizon in horizons:
            model_path = os.path.join(save_dir, f"{ticker}_{horizon}_{self.model_type}.pkl")
            scaler_path = os.path.join(save_dir, f"{ticker}_{horizon}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[horizon] = joblib.load(model_path)
                self.scalers[horizon] = joblib.load(scaler_path)
                self.trained_horizons.add(horizon)
        
        print(f"✓ Loaded {len(self.trained_horizons)} models for {ticker}")


def train_direct_forecaster(
    model_instance,
    X: np.ndarray,
    y: pd.Series,
    dates: pd.Series,
    horizon: int,
    fit_kwargs: dict = None
):
    """
    Helper function to train a single model for a specific horizon
    
    Args:
        model_instance: Sklearn model instance
        X: Features
        y: Target prices
        dates: Dates aligned with data
        horizon: Number of days ahead to predict
        fit_kwargs: Additional kwargs for model.fit()
        
    Returns:
        Fitted model
    """
    # Create horizon target
    df = pd.DataFrame({
        'date': dates,
        'price': y
    })
    df = df.sort_values('date').reset_index(drop=True)
    df['target_h'] = df['price'].shift(-horizon)
    df_valid = df.dropna()
    
    # Align X
    X_valid = X[df_valid.index] if isinstance(X, np.ndarray) else X.iloc[df_valid.index]
    y_horizon = df_valid['target_h']
    
    # Fit
    if fit_kwargs:
        model_instance.fit(X_valid, y_horizon, **fit_kwargs)
    else:
        model_instance.fit(X_valid, y_horizon)
    
    return model_instance


def map_forecast_horizon_to_days(horizon_str: str) -> int:
    """
    Convert horizon string to integer days
    
    Examples:
        '1 Month' → 30
        '3 Months' → 90
        '6 Months' → 180
        '1 Year' → 365
    """
    horizon_mapping = {
        '1 Month': 30,
        '3 Months': 90,
        '6 Months': 180,
        '1 Year': 365,
        '2 Years': 730
    }
    
    return horizon_mapping.get(horizon_str, 30)  # Default to 30 days

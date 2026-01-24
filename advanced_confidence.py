"""
Advanced Confidence Interval Calculation

Uses bootstrap sampling from historical residuals to compute empirical confidence intervals.
Replaces volatility-based assumptions with actual prediction error distributions.

Author: Stock Prediction Dashboard
Created: 2026-01-24
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


class ResidualBootstrapCI:
    """
    Calculate confidence intervals using bootstrap sampling from residuals
    
    More accurate than volatility-based CI because it uses actual model errors.
    
    Example:
        calculator = ResidualBootstrapCI()
        calculator.fit(y_true, y_pred)  # Save residuals
        ci = calculator.calculate_ci(prediction=150.50, n_sims=2000)
        # Returns: {'low': 145.20, 'high': 156.80, 'level': 'Medium'}
    """
    
    def __init__(self, percentiles: Tuple[float, float] = (10, 90)):
        """
        Initialize CI calculator
        
        Args:
            percentiles: Tuple of (lower, upper) percentiles for CI
                        Default (10, 90) gives 80% confidence interval
        """
        self.percentiles = percentiles
        self.residuals = []
        self.fitted = False
        
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute and store residuals from validation data
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        """
        residuals = y_true - y_pred
        self.residuals = residuals[~np.isnan(residuals)]  # Remove NaN
        self.fitted = True
        
        # Store statistics
        self.mean_residual = np.mean(self.residuals)
        self.std_residual = np.std(self.residuals)
        self.mae = np.mean(np.abs(self.residuals))
        
    def calculate_ci(
        self,
        prediction: float,
        n_sims: int = 2000,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate confidence interval using bootstrap
        
        Args:
            prediction: Point prediction to compute CI around
            n_sims: Number of bootstrap samples
            seed: Random seed for reproducibility
            
        Returns:
            Dict with 'predicted_price', 'confidence_low', 'confidence_high',
            'confidence_percent', 'confidence_level'
        """
        if not self.fitted:
            raise ValueError("Must call fit() before calculate_ci()")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Bootstrap: sample residuals with replacement
        sampled_residuals = np.random.choice(
            self.residuals,
            size=n_sims,
            replace=True
        )
        
        # Simulate prices
        simulated_prices = prediction + sampled_residuals
        
        # Calculate percentiles
        ci_low = np.percentile(simulated_prices, self.percentiles[0])
        ci_high = np.percentile(simulated_prices, self.percentiles[1])
        median = np.median(simulated_prices)
        
        # Calculate confidence level classification
        range_pct = ((ci_high - ci_low) / prediction) * 100
        level = self._classify_confidence(range_pct)
        
        return {
            'predicted_price': prediction,
            'confidence_low': ci_low,
            'confidence_high': ci_high,
            'median': median,
            'confidence_percent': self.percentiles[1] - self.percentiles[0],
            'range_percent': range_pct,
            'confidence_level': level,
            'mae': self.mae
        }
    
    def _classify_confidence(self, range_percent: float) -> str:
        """
        Classify confidence level based on CI range
        
        Args:
            range_percent: Percentage width of CI relative to prediction
            
        Returns:
            'High', 'Medium', or 'Low'
        """
        if range_percent < 10:
            return 'High'
        elif range_percent < 25:
            return 'Medium'
        else:
            return 'Low'
    
    def get_coverage(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        n_sims: int = 1000
    ) -> float:
        """
        Compute empirical coverage of confidence intervals
        
        Args:
            y_true: Actual values
            predictions: Predicted values
            n_sims: Bootstrap samples per prediction
            
        Returns:
            Coverage percentage (should be ~80% for 10-90% CI)
        """
        if not self.fitted:
            raise ValueError("Must call fit() before get_coverage()")
        
        coverage_count = 0
        
        for true_val, pred_val in zip(y_true, predictions):
            ci = self.calculate_ci(pred_val, n_sims=n_sims)
            
            if ci['confidence_low'] <= true_val <= ci['confidence_high']:
                coverage_count += 1
        
        coverage = (coverage_count / len(y_true)) * 100
        return coverage


def residual_bootstrap_ci(
    prediction: np.ndarray,
    historical_residuals: np.ndarray,
    n_sims: int = 2000,
    percentiles: Tuple[float, float] = (10, 90)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standalone function for residual-based CI
    
    Args:
        prediction: Point predictions (array)
        historical_residuals: Historical errors from validation
        n_sims: Number of bootstrap samples
        percentiles: CI percentiles
        
    Returns:
        Tuple of (median, lower_bound, upper_bound)
    """
    # Handle single prediction
    if np.isscalar(prediction):
        prediction = np.array([prediction])
    
    results_median = []
    results_low = []
    results_high = []
    
    for pred in prediction:
        # Bootstrap residuals
        sampled = np.random.choice(historical_residuals, size=n_sims, replace=True)
        simulated = pred + sampled
        
        median = np.median(simulated)
        low = np.percentile(simulated, percentiles[0])
        high = np.percentile(simulated, percentiles[1])
        
        results_median.append(median)
        results_low.append(low)
        results_high.append(high)
    
    return (
        np.array(results_median),
        np.array(results_low),
        np.array(results_high)
    )


class QuantileConfidenceCalculator:
    """
    Alternative: Use quantile regression for direct CI estimation
    
    Trains models to predict specific quantiles (e.g., 10th, 50th, 90th percentile)
    instead of just the mean.
    """
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        """
        Initialize quantile calculator
        
        Args:
            quantiles: List of quantiles to predict
        """
        self.quantiles = quantiles
        self.models = {}
        
    def fit(self, X, y):
        """
        Train quantile regression models
        
        Requires: from sklearn.ensemble import GradientBoostingRegressor
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        for q in self.quantiles:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            model.fit(X, y)
            self.models[q] = model
    
    def predict(self, X) -> Dict[str, np.ndarray]:
        """
        Predict all quantiles
        
        Returns:
            Dict with keys like 'q_0.1', 'q_0.5', 'q_0.9'
        """
        predictions = {}
        for q, model in self.models.items():
            predictions[f'q_{q}'] = model.predict(X)
        
        return predictions


def combine_ensemble_ci(
    predictions: Dict[str, float],
    residuals_dict: Dict[str, np.ndarray],
    weights: Dict[str, float],
    n_sims: int = 2000
) -> Dict[str, float]:
    """
    Combine confidence intervals from multiple models in an ensemble
    
    Args:
        predictions: {model_name: prediction} dict
        residuals_dict: {model_name: residuals_array} dict
        weights: {model_name: weight} dict
        n_sims: Bootstrap samples
        
    Returns:
        Combined CI dict
    """
    all_simulations = []
    
    for model_name, prediction in predictions.items():
        weight = weights.get(model_name, 0)
        residuals = residuals_dict.get(model_name, np.array([0]))
        
        # Bootstrap this model's residuals
        if len(residuals) > 0:
            sampled = np.random.choice(residuals, size=n_sims, replace=True)
            simulated = prediction + sampled
            
            # Weight the simulations
            all_simulations.extend(simulated * weight)
    
    all_simulations = np.array(all_simulations)
    
    # Compute combined statistics
    ensemble_pred = sum(pred * weights.get(name, 0) 
                       for name, pred in predictions.items())
    
    return {
        'predicted_price': ensemble_pred,
        'confidence_low': np.percentile(all_simulations, 10),
        'confidence_high': np.percentile(all_simulations, 90),
        'median': np.median(all_simulations),
        'std': np.std(all_simulations)
    }


def monte_carlo_dropout_ci(
    model,
    X: np.ndarray,
    n_samples: int = 100
) -> Dict[str, np.ndarray]:
    """
    Calculate CI using Monte Carlo Dropout for neural networks
    
    Requires: TensorFlow/Keras model with Dropout layers
    
    Args:
        model: Keras model with dropout
        X: Input features
        n_samples: Number of MC samples with dropout active
        
    Returns:
        Dict with 'mean', 'std', 'p10', 'p90'
    """
    predictions = []
    
    for _ in range(n_samples):
        # Run prediction with dropout active (training=True)
        pred = model(X, training=True)
        predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    
    return {
        'mean': np.mean(predictions, axis=0),
        'std': np.std(predictions, axis=0),
        'p10': np.percentile(predictions, 10, axis=0),
        'p50': np.percentile(predictions, 50, axis=0),
        'p90': np.percentile(predictions, 90, axis=0)
    }


def conformal_prediction_ci(
    calibration_residuals: np.ndarray,
    prediction: float,
    alpha: float = 0.1
) -> Tuple[float, float]:
    """
    Conformal prediction for distribution-free CI
    
    Args:
        calibration_residuals: Absolute residuals from calibration set
        prediction: Point prediction
        alpha: Significance level (0.1 for 90% CI)
        
    Returns:
        (lower_bound, upper_bound)
    """
    # Calculate the (1-alpha) quantile of absolute residuals
    abs_residuals = np.abs(calibration_residuals)
    quantile = np.quantile(abs_residuals, 1 - alpha)
    
    # Prediction interval
    lower = prediction - quantile
    upper = prediction + quantile
    
    return lower, upper

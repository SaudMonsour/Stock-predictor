"""
Confidence Calculator for Stock Predictions
Calculates prediction intervals using historical volatility and statistical methods
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats


class ConfidenceCalculator:
    """Calculate confidence intervals for stock price predictions"""
    
    def __init__(self):
        """Initialize confidence calculator"""
        self.confidence_levels = {
            'Low': (0, 50),
            'Medium': (50, 75),
            'High': (75, 100)
        }
    
    def calculate_from_volatility(
        self,
        predicted_price: float,
        volatility: float,
        days_ahead: int,
        confidence_percent: float = 68.0
    ) -> Dict:
        """
        Calculate confidence interval using volatility
        
        Args:
            predicted_price: Model's predicted price
            volatility: Annual volatility (e.g., 0.30 for 30%)
            days_ahead: Number of trading days in forecast
            confidence_percent: Confidence level (68% = 1 std, 95% = 2 std)
            
        Returns:
            Dictionary with confidence bounds and classification
        """
        # Convert annual volatility to forecast period volatility
        # Volatility scales with sqrt(time)
        years_fraction = days_ahead / 252.0
        period_volatility = volatility * np.sqrt(years_fraction)
        
        # Determine z-score for confidence level
        # 68% ≈ 1σ, 95% ≈ 2σ, 99% ≈ 3σ
        if confidence_percent >= 99:
            z_score = 2.576  # 99%
        elif confidence_percent >= 95:
            z_score = 1.96   # 95%
        else:
            z_score = 1.0    # 68%
        
        # Calculate bounds assuming log-normal distribution
        log_std = period_volatility
        
        confidence_low = predicted_price * np.exp(-z_score * log_std)
        confidence_high = predicted_price * np.exp(z_score * log_std)
        
        # Calculate relative confidence (narrower range = higher confidence)
        range_pct = (confidence_high - confidence_low) / predicted_price * 100
        
        # Classify confidence level
        if range_pct < 10:
            confidence_level = 'High'
        elif range_pct < 25:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        return {
            'predicted_price': round(predicted_price, 2),
            'confidence_low': round(confidence_low, 2),
            'confidence_high': round(confidence_high, 2),
            'confidence_percent': int(confidence_percent),
            'confidence_level': confidence_level,
            'range_percent': round(range_pct, 1)
        }
    
    def calculate_from_residuals(
        self,
        predicted_price: float,
        historical_errors: np.ndarray,
        confidence_percent: float = 68.0
    ) -> Dict:
        """
        Calculate confidence interval using model's historical prediction errors
        
        Args:
            predicted_price: Model's predicted price
            historical_errors: Array of past prediction errors
            confidence_percent: Confidence level
            
        Returns:
            Dictionary with confidence bounds
        """
        if len(historical_errors) == 0:
            # Fallback to wide bounds if no historical data
            return {
                'predicted_price': round(predicted_price, 2),
                'confidence_low': round(predicted_price * 0.85, 2),
                'confidence_high': round(predicted_price * 1.15, 2),
                'confidence_percent': int(confidence_percent),
                'confidence_level': 'Low',
                'range_percent': 30.0
            }
        
        # Calculate standard deviation of errors
        error_std = np.std(historical_errors)
        error_mean = np.mean(historical_errors)
        
        # Determine z-score for confidence level
        if confidence_percent >= 95:
            z_score = 1.96
        else:
            z_score = 1.0
        
        # Calculate bounds
        margin = z_score * error_std
        confidence_low = predicted_price - margin
        confidence_high = predicted_price + margin
        
        # Ensure non-negative prices
        confidence_low = max(0, confidence_low)
        
        # Classify confidence
        range_pct = (confidence_high - confidence_low) / predicted_price * 100
        
        if range_pct < 10:
            confidence_level = 'High'
        elif range_pct < 25:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        return {
            'predicted_price': round(predicted_price, 2),
            'confidence_low': round(confidence_low, 2),
            'confidence_high': round(confidence_high, 2),
            'confidence_percent': int(confidence_percent),
            'confidence_level': confidence_level,
            'range_percent': round(range_pct, 1)
        }
    
    def monte_carlo_simulation(
        self,
        current_price: float,
        expected_return: float,
        volatility: float,
        days_ahead: int,
        num_simulations: int = 1000
    ) -> Dict:
        """
        Use Monte Carlo simulation for long-term predictions
        
        Args:
            current_price: Current stock price
            expected_return: Annual expected return
            volatility: Annual volatility
            days_ahead: Trading days to simulate
            num_simulations: Number of Monte Carlo paths
            
        Returns:
            Confidence bounds from simulation percentiles
        """
        # Convert to daily parameters
        daily_return = expected_return / 252
        daily_vol = volatility / np.sqrt(252)
        
        # Run simulations
        final_prices = []
        
        for _ in range(num_simulations):
            price = current_price
            
            for _ in range(days_ahead):
                # Geometric Brownian Motion
                random_shock = np.random.normal(0, 1)
                price = price * np.exp(daily_return + daily_vol * random_shock)
            
            final_prices.append(price)
        
        final_prices = np.array(final_prices)
        
        # Calculate percentiles
        median_price = np.median(final_prices)
        low_bound = np.percentile(final_prices, 16)   # 1σ lower
        high_bound = np.percentile(final_prices, 84)  # 1σ upper
        
        range_pct = (high_bound - low_bound) / median_price * 100
        
        if range_pct < 15:
            confidence_level = 'High'
        elif range_pct < 30:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        return {
            'predicted_price': round(median_price, 2),
            'confidence_low': round(low_bound, 2),
            'confidence_high': round(high_bound, 2),
            'confidence_percent': 68,
            'confidence_level': confidence_level,
            'range_percent': round(range_pct, 1),
            'simulations_run': num_simulations
        }
    
    def get_explanation(self, confidence_data: Dict) -> str:
        """
        Generate human-readable explanation of confidence level
        
        Args:
            confidence_data: Confidence calculation result
            
        Returns:
            Plain English explanation
        """
        level = confidence_data['confidence_level']
        range_pct = confidence_data['range_percent']
        low = confidence_data['confidence_low']
        high = confidence_data['confidence_high']
        pred = confidence_data['predicted_price']
        
        explanations = {
            'High': f"High confidence prediction. The model expects the price to be ${pred:.2f}, with a narrow range between ${low:.2f} and ${high:.2f} (±{range_pct/2:.1f}%). This suggests stable, predictable behavior.",
            
            'Medium': f"Moderate confidence prediction. The price is expected around ${pred:.2f}, but could range from ${low:.2f} to ${high:.2f} (±{range_pct/2:.1f}%). Some uncertainty exists due to market volatility.",
            
            'Low': f"Low confidence prediction. While the model predicts ${pred:.2f}, there's significant uncertainty with a wide range from ${low:.2f} to ${high:.2f} (±{range_pct/2:.1f}%). Proceed with caution due to high volatility or limited data."
        }
        
        return explanations.get(level, "Confidence level unavailable.")


# Convenience function
def calculate_confidence(
    predicted_price: float,
    volatility: float,
    days_ahead: int,
    confidence_percent: float = 68.0
) -> Dict:
    """
    Quick confidence calculation using volatility method
    
    Args:
        predicted_price: Model's predicted price
        volatility: Annual volatility
        days_ahead: Trading days ahead
        confidence_percent: Confidence level (default 68%)
        
    Returns:
        Confidence interval dictionary
    """
    calculator = ConfidenceCalculator()
    return calculator.calculate_from_volatility(
        predicted_price, volatility, days_ahead, confidence_percent
    )

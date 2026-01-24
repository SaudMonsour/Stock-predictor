"""
Risk Analyzer for Stock Predictions
Classifies risk levels and detects warning conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import timedelta


class RiskAnalyzer:
    """Analyze risk profile and detect warning conditions for predictions"""
    
    # Risk classification thresholds
    RISK_THRESHOLDS = {
        'volatility': {
            'low': 0.20,      # < 20% annual volatility
            'high': 0.40      # > 40% annual volatility
        },
        'beta': {
            'low': 0.8,       # < 0.8 beta
            'high': 1.5       # > 1.5 beta
        },
        'max_drawdown': {
            'low': -0.15,     # > -15% max drawdown
            'high': -0.30     # < -30% max drawdown
        }
    }
    
    def __init__(self):
        """Initialize risk analyzer"""
        pass
    
    def classify_risk(self, metrics: Dict) -> str:
        """
        Classify stock risk as Low, Medium, or High
        
        Args:
            metrics: Dictionary with Volatility, Beta, MDD
            
        Returns:
            Risk classification string
        """
        volatility = metrics.get('Volatility', 0.25)
        beta = metrics.get('Beta', 1.0)
        max_drawdown = metrics.get('MDD', -0.20)
        
        # Count high-risk indicators
        risk_score = 0
        
        # Volatility check
        if volatility > self.RISK_THRESHOLDS['volatility']['high']:
            risk_score += 2
        elif volatility > self.RISK_THRESHOLDS['volatility']['low']:
            risk_score += 1
        
        # Beta check
        if beta > self.RISK_THRESHOLDS['beta']['high']:
            risk_score += 2
        elif beta > self.RISK_THRESHOLDS['beta']['low']:
            risk_score += 1
        
        # Max Drawdown check (more negative = worse)
        if max_drawdown < self.RISK_THRESHOLDS['max_drawdown']['high']:
            risk_score += 2
        elif max_drawdown < self.RISK_THRESHOLDS['max_drawdown']['low']:
            risk_score += 1
        
        # Classify based on total score
        if risk_score >= 5:
            return 'High'
        elif risk_score >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def detect_warnings(self, stock_data: pd.DataFrame, metrics: Dict) -> List[str]:
        """
        Detect conditions that make predictions unreliable
        
        Args:
            stock_data: Historical stock price data
            metrics: Financial metrics dictionary
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check 1: Insufficient data
        if len(stock_data) < 500:  # Less than ~2 years of daily data
            years = len(stock_data) / 252
            warnings.append(
                f"Limited historical data ({years:.1f} years). "
                "Predictions may be less reliable."
            )
        
        # Check 2: Extreme volatility
        volatility = metrics.get('Volatility', 0)
        if volatility > 0.60:  # > 60% annual volatility
            warnings.append(
                f"Extremely high volatility ({volatility:.1%}). "
                "Price movements are highly unpredictable."
            )
        
        # Check 3: Recent volatility spike
        if self._detect_volatility_spike(stock_data):
            warnings.append(
                "Recent volatility spike detected. "
                "Market conditions are unstable."
            )
        
        # Check 4: Abnormal price movements
        if self._detect_abnormal_prices(stock_data):
            warnings.append(
                "Abnormal price movements detected. "
                "Recent events may have disrupted normal patterns."
            )
        
        # Check 5: Severe max drawdown
        max_drawdown = metrics.get('MDD', 0)
        if max_drawdown < -0.50:  # > 50% drawdown
            warnings.append(
                f"Severe historical drawdown ({max_drawdown:.1%}). "
                "Stock has experienced major crashes."
            )
        
        # Check 6: Low trading volume (if available)
        if 'Volume' in stock_data.columns:
            if self._detect_low_volume(stock_data):
                warnings.append(
                    "Low trading volume detected. "
                    "Prices may be less stable and harder to predict."
                )
        
        return warnings
    
    def is_prediction_reliable(
        self,
        stock_data: pd.DataFrame,
        metrics: Dict
    ) -> Tuple[bool, str]:
        """
        Determine if predictions are reliable
        
        Args:
            stock_data: Historical data
            metrics: Financial metrics
            
        Returns:
            (is_reliable, reason) tuple
        """
        warnings = self.detect_warnings(stock_data, metrics)
        
        # Critical conditions that make predictions unreliable
        critical_warnings = [
            w for w in warnings
            if any(keyword in w.lower() for keyword in [
                'extremely high volatility',
                'abnormal price movements',
                'limited historical data'
            ])
        ]
        
        if len(critical_warnings) >= 2:
            return False, "Multiple critical warning conditions detected."
        
        if len(warnings) >= 4:
            return False, "Too many warning indicators present."
        
        # Check for extreme volatility alone
        volatility = metrics.get('Volatility', 0)
        if volatility > 0.70:
            return False, "Volatility exceeds safe prediction threshold."
        
        return True, "Prediction conditions are acceptable."
    
    def get_risk_explanation(self, risk_level: str, metrics: Dict) -> str:
        """
        Generate human-readable risk explanation
        
        Args:
            risk_level: Low, Medium, or High
            metrics: Financial metrics
            
        Returns:
            Plain English explanation
        """
        volatility = metrics.get('Volatility', 0)
        beta = metrics.get('Beta', 1.0)
        max_drawdown = metrics.get('MDD', 0)
        
        explanations = {
            'Low': (
                f"Low Risk Stock: This stock shows stable behavior with "
                f"{volatility:.1%} volatility and {beta:.2f} beta. "
                f"Suitable for conservative investors."
            ),
            'Medium': (
                f"Medium Risk Stock: Moderate volatility ({volatility:.1%}) "
                f"and beta ({beta:.2f}). Balanced risk-reward profile. "
                f"Suitable for most investors."
            ),
            'High': (
                f"High Risk Stock: Elevated volatility ({volatility:.1%}), "
                f"beta ({beta:.2f}), and historical drawdown ({max_drawdown:.1%}). "
                f"Only suitable for risk-tolerant investors."
            )
        }
        
        return explanations.get(risk_level, "Risk level uncertain.")
    
    # Private helper methods
    
    def _detect_volatility_spike(self, stock_data: pd.DataFrame) -> bool:
        """Detect recent volatility spike"""
        if len(stock_data) < 100:
            return False
        
        # Calculate rolling volatility
        returns = stock_data['Close'].pct_change()
        recent_vol = returns.tail(30).std() * np.sqrt(252)
        historical_vol = returns.iloc[-252:-30].std() * np.sqrt(252)
        
        # Spike if recent volatility is 2x historical
        return recent_vol > historical_vol * 2.0
    
    def _detect_abnormal_prices(self, stock_data: pd.DataFrame) -> bool:
        """Detect abnormal price movements (outliers)"""
        if len(stock_data) < 60:
            return False
        
        # Calculate daily returns
        returns = stock_data['Close'].pct_change().dropna()
        
        # Check for returns beyond 3 standard deviations in last 30 days
        recent_returns = returns.tail(30)
        mean = returns.mean()
        std = returns.std()
        
        outliers = recent_returns[np.abs(recent_returns - mean) > 3 * std]
        
        return len(outliers) > 0
    
    def _detect_low_volume(self, stock_data: pd.DataFrame) -> bool:
        """Detect low trading volume"""
        if 'Volume' not in stock_data.columns or len(stock_data) < 60:
            return False
        
        # Compare recent volume to historical average
        recent_volume = stock_data['Volume'].tail(30).mean()
        historical_volume = stock_data['Volume'].iloc[-252:-30].mean()
        
        # Low if recent volume is less than 30% of historical
        return recent_volume < historical_volume * 0.3


# Convenience functions

def classify_risk(metrics: Dict) -> str:
    """Classify stock risk level"""
    analyzer = RiskAnalyzer()
    return analyzer.classify_risk(metrics)


def get_warnings(stock_data: pd.DataFrame, metrics: Dict) -> List[str]:
    """Get list of warning messages"""
    analyzer = RiskAnalyzer()
    return analyzer.detect_warnings(stock_data, metrics)


def is_reliable(stock_data: pd.DataFrame, metrics: Dict) -> Tuple[bool, str]:
    """Check if prediction is reliable"""
    analyzer = RiskAnalyzer()
    return analyzer.is_prediction_reliable(stock_data, metrics)

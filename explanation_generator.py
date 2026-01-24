"""
Explanation Generator for Stock Predictions
Generates natural language explanations in simple English
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class ExplanationGenerator:
    """Generate human-readable explanations for stock predictions"""
    
    def __init__(self):
        """Initialize explanation generator"""
        pass
    
    def generate_explanation(
        self,
        prediction_change: float,
        stock_data: pd.DataFrame,
        metrics: Dict,
        confidence_level: str
    ) -> str:
        """
        Generate comprehensive explanation for prediction
        
        Args:
            prediction_change: Predicted price change (percentage)
            stock_data: Historical stock data
            metrics: Financial metrics
            confidence_level: Prediction confidence level
            
        Returns:
            Human-readable explanation string
        """
        # Determine prediction direction
        direction = "rise" if prediction_change > 0 else "fall"
        direction_adj = "bullish" if prediction_change > 0 else "bearish"
        
        # Analyze indicators
        trend = self._analyze_trend(stock_data)
        momentum = self._analyze_momentum(stock_data)
        volume_signal = self._analyze_volume(stock_data)
        volatility_signal = self._analyze_volatility(metrics)
        
        # Build explanation
        reasons = []
        
        # Add trend reason
        if (trend == "upward" and prediction_change > 0) or \
           (trend == "downward" and prediction_change < 0):
            reasons.append(f"{trend.capitalize()} price trend over recent months")
        
        # Add momentum reason
        if momentum != "neutral":
            reasons.append(f"{momentum.capitalize()} momentum signals")
        
        # Add volume reason
        if volume_signal:
            reasons.append(volume_signal)
        
        # Add volatility reason
        if volatility_signal:
            reasons.append(volatility_signal)
        
        # Construct final explanation
        if len(reasons) == 0:
            reasons.append("mixed technical indicators")
        
        explanation = (
            f"This stock is predicted to {direction} {abs(prediction_change):.1%} because:\n"
            f"• {reasons[0]}"
        )
        
        for reason in reasons[1:]:
            explanation += f"\n• {reason}"
        
        # Add confidence context
        confidence_context = {
            'High': "The prediction model shows high confidence in this forecast.",
            'Medium': "The prediction has moderate confidence due to some market uncertainty.",
            'Low': "Caution advised: prediction confidence is low due to high volatility or limited data."
        }
        
        explanation += f"\n\n{confidence_context.get(confidence_level, '')}"
        
        return explanation
    
    def generate_simple_explanation(
        self,
        prediction_change: float,
        trend: str,
        volatility: float
    ) -> str:
        """
        Generate a simplified one-line explanation
        
        Args:
            prediction_change: Predicted percentage change
            trend: "up", "down", or "sideways"
            volatility: Volatility metric
            
        Returns:
            One-line explanation
        """
        direction = "rise" if prediction_change > 0 else "fall"
        
        # Determine volatility description
        if volatility < 0.20:
            vol_desc = "low volatility suggests stable growth"
        elif volatility < 0.40:
            vol_desc = "moderate volatility indicates normal market conditions"
        else:
            vol_desc = "high volatility means increased uncertainty"
        
        # Determine trend description
        trend_desc = {
            "up": "strong upward momentum",
            "down": "downward pressure",
            "sideways": "mixed signals"
        }.get(trend.lower(), "unclear trend")
        
        return f"Expected to {direction} due to {trend_desc} and {vol_desc}."
    
    # Private analysis methods
    
    def _analyze_trend(self, stock_data: pd.DataFrame) -> str:
        """Analyze price trend"""
        if len(stock_data) < 60:
            return "neutral"
        
        # Compare recent prices to older prices
        recent_avg = stock_data['Close'].tail(30).mean()
        older_avg = stock_data['Close'].iloc[-90:-60].mean()
        
        change = (recent_avg - older_avg) / older_avg
        
        if change > 0.05:
            return "upward"
        elif change < -0.05:
            return "downward"
        else:
            return "sideways"
    
    def _analyze_momentum(self, stock_data: pd.DataFrame) -> str:
        """Analyze momentum using simple moving average crossover"""
        if len(stock_data) < 50:
            return "neutral"
        
        # Calculate short and long moving averages
        ma_short = stock_data['Close'].rolling(window=20).mean().iloc[-1]
        ma_long = stock_data['Close'].rolling(window=50).mean().iloc[-1]
        current_price = stock_data['Close'].iloc[-1]
        
        # Determine momentum
        if ma_short > ma_long and current_price > ma_short:
            return "positive"
        elif ma_short < ma_long and current_price < ma_short:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_volume(self, stock_data: pd.DataFrame) -> str:
        """Analyze trading volume"""
        if 'Volume' not in stock_data.columns or len(stock_data) < 30:
            return None
        
        recent_volume = stock_data['Volume'].tail(10).mean()
        average_volume = stock_data['Volume'].tail(60).mean()
        
        if recent_volume > average_volume * 1.3:
            return "Above-average trading volume shows strong investor interest"
        elif recent_volume < average_volume * 0.7:
            return "Lower trading volume suggests reduced market activity"
        else:
            return None
    
    def _analyze_volatility(self, metrics: Dict) -> str:
        """Analyze volatility implications"""
        volatility = metrics.get('Volatility', 0.25)
        
        if volatility < 0.15:
            return "Low volatility indicates stable, predictable price movements"
        elif volatility > 0.45:
            return "High volatility creates significant price uncertainty"
        else:
            return None
    
    def explain_risk(self, risk_level: str, reasons: List[str]) -> str:
        """
        Explain why a stock has a certain risk level
        
        Args:
            risk_level: Low, Medium, or High
            reasons: List of risk factors
            
        Returns:
            Risk explanation
        """
        intro = {
            'Low': "This stock is classified as low risk.",
            'Medium': "This stock has a medium risk profile.",
            'High': "⚠️ This is a high-risk stock."
        }.get(risk_level, "Risk level unknown.")
        
        if len(reasons) == 0:
            return intro
        
        explanation = intro + " Key factors:\n"
        for reason in reasons:
            explanation += f"• {reason}\n"
        
        return explanation.strip()
    
    def explain_confidence(self, confidence_data: Dict) -> str:
        """
        Explain confidence level in simple terms
        
        Args:
            confidence_data: Confidence calculation results
            
        Returns:
            Confidence explanation
        """
        level = confidence_data.get('confidence_level', 'Medium')
        range_pct = confidence_data.get('range_percent', 20)
        
        explanations = {
            'High': f"High confidence (±{range_pct/2:.1f}%): The model's predictions have been consistently accurate for this stock.",
            'Medium': f"Moderate confidence (±{range_pct/2:.1f}%): Predictions are reasonable but some uncertainty exists.",
            'Low': f"Low confidence (±{range_pct/2:.1f}%): Significant uncertainty due to volatility or limited historical patterns."
        }
        
        return explanations.get(level, "Confidence level unavailable.")


# Convenience functions

def explain_prediction(
    prediction_change: float,
    stock_data: pd.DataFrame,
    metrics: Dict,
    confidence_level: str
) -> str:
    """Generate prediction explanation"""
    generator = ExplanationGenerator()
    return generator.generate_explanation(
        prediction_change, stock_data, metrics, confidence_level
    )


def simple_explanation(prediction_change: float, trend: str, volatility: float) -> str:
    """Generate simple one-line explanation"""
    generator = ExplanationGenerator()
    return generator.generate_simple_explanation(prediction_change, trend, volatility)

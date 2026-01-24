"""
Comprehensive Backtesting Module

Provides extensive evaluation metrics beyond simple point errors.
Includes directional accuracy, economic metrics, and baseline comparisons.

Author: Stock Prediction Dashboard
Created: 2026-01-24
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BacktestResults:
    """Container for backtest metrics"""
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    sharpe_ratio: float
    cumulative_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'MAE': self.mae,
            'RMSE': self.rmse,
            'MAPE': self.mape,
            'Directional_Accuracy': self.directional_accuracy,
            'Sharpe_Ratio': self.sharpe_ratio,
            'Cumulative_Return': self.cumulative_return,
            'Max_Drawdown': self.max_drawdown,
            'Win_Rate': self.win_rate,
            'Profit_Factor': self.profit_factor
        }


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_prices: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Comprehensive forecast evaluation
    
    Args:
        y_true: Actual future values
        y_pred: Predicted future values
        current_prices: Current prices (for directional accuracy)
                       If None, uses previous values in y_true
        
    Returns:
        Dict with comprehensive metrics
    """
    # Basic error metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    
    # Directional accuracy
    if current_prices is None:
        # Use lagged y_true as current prices
        current_prices = np.concatenate([[y_true[0]], y_true[:-1]])
    
    actual_direction = np.sign(y_true - current_prices)
    pred_direction = np.sign(y_pred - current_prices)
    directional_acc = np.mean(actual_direction == pred_direction) * 100
    
    # Economic metrics (simple long/short strategy)
    returns = (y_true - current_prices) / (current_prices + 1e-9)
    signals = np.sign(y_pred - current_prices)  # 1 for long, -1 for short, 0 for neutral
    strategy_returns = returns * signals
    
    # Sharpe ratio (annualized, assuming daily returns)
    if np.std(strategy_returns) > 0:
        sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Cumulative return
    cumulative_return = np.sum(strategy_returns) * 100
    
    # Max drawdown
    cumulative_wealth = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / (running_max + 1e-9)
    max_drawdown = np.min(drawdown) * 100
    
    # Win rate
    winning_trades = np.sum(strategy_returns > 0)
    total_trades = len(strategy_returns)
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Profit factor
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = gross_profit / (gross_loss + 1e-9)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Directional_Accuracy': directional_acc,
        'Sharpe_Ratio': sharpe,
        'Cumulative_Return': cumulative_return,
        'Max_Drawdown': max_drawdown,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor
    }


def baseline_naive_forecast(
    y_true: np.ndarray,
    method: str = 'last_price'
) -> np.ndarray:
    """
    Generate naive baseline predictions
    
    Args:
        y_true: Actual values
        method: 'last_price', 'mean', or 'trend'
        
    Returns:
        Baseline predictions
    """
    if method == 'last_price':
        # Random walk: next value = current value
        return np.concatenate([[y_true[0]], y_true[:-1]])
    
    elif method == 'mean':
        # Historical mean
        return np.full_like(y_true, np.mean(y_true))
    
    elif method == 'trend':
        # Linear trend extrapolation
        x = np.arange(len(y_true))
        coeffs = np.polyfit(x, y_true, deg=1)
        return np.polyval(coeffs, x)
    
    else:
        raise ValueError(f"Unknown baseline method: {method}")


def compare_to_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_method: str = 'last_price'
) -> Dict[str, any]:
    """
    Compare model predictions to naive baseline
    
    Args:
        y_true: Actual values
        y_pred: Model predictions
        baseline_method: Baseline strategy
        
    Returns:
        Dict with model metrics, baseline metrics, and improvement
    """
    baseline_pred = baseline_naive_forecast(y_true, method=baseline_method)
    
    model_metrics = evaluate_forecast(y_true, y_pred)
    baseline_metrics = evaluate_forecast(y_true, baseline_pred)
    
    # Calculate improvements
    improvements = {}
    for key in model_metrics:
        baseline_val = baseline_metrics[key]
        model_val = model_metrics[key]
        
        if baseline_val != 0:
            if key in ['MAE', 'RMSE', 'MAPE', 'Max_Drawdown']:
                # Lower is better
                improvement = ((baseline_val - model_val) / abs(baseline_val)) * 100
            else:
                # Higher is better
                improvement = ((model_val - baseline_val) / abs(baseline_val)) * 100
        else:
            improvement = 0.0
        
        improvements[f'{key}_Improvement'] = improvement
    
    return {
        'model_metrics': model_metrics,
        'baseline_metrics': baseline_metrics,
        'improvements': improvements,
        'beats_baseline': model_metrics['RMSE'] < baseline_metrics['RMSE']
    }


class BacktestEngine:
    """
    Full backtesting engine with walk-forward validation
    
    Example:
        engine = BacktestEngine()
        results = engine.backtest(
            model=my_model,
            X=features,
            y=targets,
            dates=date_series
        )
        engine.print_report()
    """
    
    def __init__(self):
        self.results = []
        self.fold_metrics = []
        
    def backtest(
        self,
        model,
        X: np.ndarray,
        y: pd.Series,
        dates: pd.Series,
        n_splits: int = 5
    ) -> List[Dict]:
        """
        Run walk-forward backtest
        
        Args:
            model: Model with fit() and predict() methods
            X: Features
            y: Targets
            dates: Dates
            n_splits: Number of CV folds
            
        Returns:
            List of fold results
        """
        from walk_forward_validation import walk_forward_split
        
        self.results = []
        
        for fold, (train_idx, test_idx) in enumerate(walk_forward_split(dates, n_splits), 1):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            metrics = evaluate_forecast(
                y_test.values,
                y_pred,
                current_prices=y_test.values  # For directional accuracy
            )
            
            metrics['fold'] = fold
            metrics['train_size'] = len(train_idx)
            metrics['test_size'] = len(test_idx)
            
            self.results.append(metrics)
        
        return self.results
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Aggregate metrics across all folds
        
        Returns:
            Dict with mean and std of each metric
        """
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        summary = {}
        for col in df.columns:
            if col not in ['fold', 'train_size', 'test_size']:
                summary[f'{col}_mean'] = df[col].mean()
                summary[f'{col}_std'] = df[col].std()
        
        return summary
    
    def print_report(self):
        """Print formatted backtest report"""
        if not self.results:
            print("No backtest results available")
            return
        
        print(f"\n{'='*70}")
        print(f"Backtest Report")
        print(f"{'='*70}\n")
        
        summary = self.get_summary_metrics()
        
        print(f"{'Metric':<25} {'Mean':<15} {'Std':<15}")
        print("-" * 55)
        
        key_metrics = [
            ('MAE', '$'),
            ('RMSE', '$'),
            ('MAPE', '%'),
            ('Directional_Accuracy', '%'),
            ('Sharpe_Ratio', ''),
            ('Cumulative_Return', '%'),
            ('Max_Drawdown', '%'),
            ('Win_Rate', '%'),
            ('Profit_Factor', 'x')
        ]
        
        for metric_name, unit in key_metrics:
            mean_val = summary.get(f'{metric_name}_mean', 0)
            std_val = summary.get(f'{metric_name}_std', 0)
            
            if unit == '$':
                print(f"{metric_name:<25} ${mean_val:<14.2f} ${std_val:<14.2f}")
            elif unit in ['%', 'x']:
                print(f"{metric_name:<25} {mean_val:<14.2f}{unit} {std_val:<14.2f}{unit}")
            else:
                print(f"{metric_name:<25} {mean_val:<14.2f} {std_val:<14.2f}")
        
        print(f"\n{'='*70}\n")


def generate_backtest_report(
    results: List[Dict],
    ticker: str,
    model_name: str
) -> str:
    """
    Generate markdown backtest report
    
    Args:
        results: List of fold metrics
        ticker: Stock ticker
        model_name: Model name
        
    Returns:
        Markdown formatted report
    """
    df = pd.DataFrame(results)
    
    report = f"""# Backtest Report: {ticker} - {model_name}

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| MAE | ${df['MAE'].mean():.2f} | ${df['MAE'].std():.2f} | ${df['MAE'].min():.2f} | ${df['MAE'].max():.2f} |
| RMSE | ${df['RMSE'].mean():.2f} | ${df['RMSE'].std():.2f} | ${df['RMSE'].min():.2f} | ${df['RMSE'].max():.2f} |
| MAPE | {df['MAPE'].mean():.2f}% | {df['MAPE'].std():.2f}% | {df['MAPE'].min():.2f}% | {df['MAPE'].max():.2f}% |
| Directional Accuracy | {df['Directional_Accuracy'].mean():.2f}% | {df['Directional_Accuracy'].std():.2f}% | {df['Directional_Accuracy'].min():.2f}% | {df['Directional_Accuracy'].max():.2f}% |
| Sharpe Ratio | {df['Sharpe_Ratio'].mean():.2f} | {df['Sharpe_Ratio'].std():.2f} | {df['Sharpe_Ratio'].min():.2f} | {df['Sharpe_Ratio'].max():.2f} |

## Economic Metrics

- **Cumulative Return**: {df['Cumulative_Return'].mean():.2f}%
- **Max Drawdown**: {df['Max_Drawdown'].mean():.2f}%
- **Win Rate**: {df['Win_Rate'].mean():.2f}%
- **Profit Factor**: {df['Profit_Factor'].mean():.2f}x

## Per-Fold Performance

| Fold | MAE | RMSE | Dir. Acc. | Sharpe |
|------|-----|------|-----------|--------|
"""
    
    for _, row in df.iterrows():
        report += f"| {row['fold']} | ${row['MAE']:.2f} | ${row['RMSE']:.2f} | {row['Directional_Accuracy']:.1f}% | {row['Sharpe_Ratio']:.2f} |\n"
    
    return report

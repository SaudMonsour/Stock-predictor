"""
Walk-Forward Cross-Validation Module

Implements time-aware train/test splits to prevent data leakage.
Uses rolling windows where training always ends before testing begins.

Author: Stock Prediction Dashboard
Created: 2026-01-24
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, List
from datetime import timedelta


def walk_forward_split(
    dates: pd.Series,
    n_splits: int = 5,
    test_size_days: int = 60
) -> Generator[Tuple[List[int], List[int]], None, None]:
    """
    Generate time-aware train/test splits for cross-validation
    
    CRITICAL: Ensures train max date < test min date (no data leakage)
    
    Args:
        dates: Series of datetime values aligned with data
        n_splits: Number of cross-validation folds
        test_size_days: Size of each test period in days
        
    Yields:
        (train_indices, test_indices) tuples
        
    Example:
        For 5 years of data with 5 splits and 60-day test periods:
        Split 1: Train [2019-2023.25] → Test [2023.25-2023.50]
        Split 2: Train [2019-2023.50] → Test [2023.50-2023.75]
        Split 3: Train [2019-2023.75] → Test [2023.75-2024.00]
        ...
    """
    # Ensure dates are sorted and datetime
    dates = pd.to_datetime(dates).sort_values()
    dates_df = pd.DataFrame({'date': dates, 'idx': range(len(dates))})
    
    # Get date range
    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days
    
    # Calculate splits working backward from end date
    test_period = pd.Timedelta(days=test_size_days)
    
    for i in range(n_splits):
        # Calculate test period (working backward)
        test_end = max_date - (i * test_period)
        test_start = test_end - test_period
        
        # Ensure test_start is not before minimum date
        if test_start <= min_date:
            continue
        
        # Train period: everything before test_start
        train_end = test_start - pd.Timedelta(days=1)
        
        # Get indices
        train_mask = dates_df['date'] <= train_end
        test_mask = (dates_df['date'] >= test_start) & (dates_df['date'] <= test_end)
        
        train_idx = dates_df[train_mask]['idx'].tolist()
        test_idx = dates_df[test_mask]['idx'].tolist()
        
        # Only yield if we have data in both sets
        if len(train_idx) > 0 and len(test_idx) > 0:
            # Verify no leakage
            assert dates_df.loc[train_idx, 'date'].max() < dates_df.loc[test_idx, 'date'].min(), \
                "Data leakage detected: train dates must be < test dates"
            
            yield train_idx, test_idx


def expanding_window_split(
    dates: pd.Series,
    min_train_days: int = 365,
    test_size_days: int = 60,
    step_size_days: int = 30
) -> Generator[Tuple[List[int], List[int]], None, None]:
    """
    Expanding window cross-validation
    
    Train window grows over time, test window rolls forward.
    More realistic for production scenarios.
    
    Args:
        dates: Series of datetime values
        min_train_days: Minimum training period required
        test_size_days: Size of test period
        step_size_days: How far to step forward each fold
        
    Yields:
        (train_indices, test_indices) tuples
    """
    dates = pd.to_datetime(dates).sort_values()
    dates_df = pd.DataFrame({'date': dates, 'idx': range(len(dates))})
    
    min_date = dates.min()
    max_date = dates.max()
    
    # Start with minimum training period
    train_start = min_date
    test_start = train_start + pd.Timedelta(days=min_train_days)
    
    while test_start + pd.Timedelta(days=test_size_days) <= max_date:
        # Define periods
        train_end = test_start - pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_size_days)
        
        # Get indices
        train_mask = (dates_df['date'] >= train_start) & (dates_df['date'] <= train_end)
        test_mask = (dates_df['date'] >= test_start) & (dates_df['date'] <= test_end)
        
        train_idx = dates_df[train_mask]['idx'].tolist()
        test_idx = dates_df[test_mask]['idx'].tolist()
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            # Verify no leakage
            assert dates_df.loc[train_idx, 'date'].max() < dates_df.loc[test_idx, 'date'].min()
            yield train_idx, test_idx
        
        # Step forward
        test_start += pd.Timedelta(days=step_size_days)


def rolling_window_split(
    dates: pd.Series,
    train_size_days: int = 1260,  # ~5 years
    test_size_days: int = 60,
    step_size_days: int = 30
) -> Generator[Tuple[List[int], List[int]], None, None]:
    """
    Rolling window cross-validation
    
    Both train and test windows roll forward (fixed size).
    Good for non-stationary data.
    
    Args:
        dates: Series of datetime values
        train_size_days: Size of training window
        test_size_days: Size of test window
        step_size_days: Step size for rolling
        
    Yields:
        (train_indices, test_indices) tuples
    """
    dates = pd.to_datetime(dates).sort_values()
    dates_df = pd.DataFrame({'date': dates, 'idx': range(len(dates))})
    
    min_date = dates.min()
    max_date = dates.max()
    
    # Start position
    train_start = min_date
    
    while True:
        train_end = train_start + pd.Timedelta(days=train_size_days)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_size_days)
        
        # Check if we've exceeded data range
        if test_end > max_date:
            break
        
        # Get indices
        train_mask = (dates_df['date'] >= train_start) & (dates_df['date'] <= train_end)
        test_mask = (dates_df['date'] >= test_start) & (dates_df['date'] <= test_end)
        
        train_idx = dates_df[train_mask]['idx'].tolist()
        test_idx = dates_df[test_mask]['idx'].tolist()
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            # Verify no leakage
            assert dates_df.loc[train_idx, 'date'].max() < dates_df.loc[test_idx, 'date'].min()
            yield train_idx, test_idx
        
        # Roll forward
        train_start += pd.Timedelta(days=step_size_days)


class WalkForwardValidator:
    """
    Wrapper class for walk-forward cross-validation
    
    Example:
        validator = WalkForwardValidator(n_splits=5)
        for fold, (train_idx, test_idx) in enumerate(validator.split(dates)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            scores = evaluate(y_test, predictions)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size_days: int = 60,
        method: str = 'walk_forward'
    ):
        """
        Initialize validator
        
        Args:
            n_splits: Number of cross-validation folds
            test_size_days: Size of test period in days
            method: 'walk_forward', 'expanding', or 'rolling'
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.method = method
        
    def split(
        self,
        dates: pd.Series
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Generate train/test splits
        
        Args:
            dates: Series of datetime values
            
        Yields:
            (train_indices, test_indices) tuples
        """
        if self.method == 'walk_forward':
            yield from walk_forward_split(
                dates,
                n_splits=self.n_splits,
                test_size_days=self.test_size_days
            )
        elif self.method == 'expanding':
            yield from expanding_window_split(
                dates,
                test_size_days=self.test_size_days
            )
        elif self.method == 'rolling':
            yield from rolling_window_split(
                dates,
                test_size_days=self.test_size_days
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_n_splits(self, dates: pd.Series) -> int:
        """Get number of splits that will be generated"""
        return sum(1 for _ in self.split(dates))


def test_no_leakage(train_idx: List[int], test_idx: List[int], dates: pd.Series) -> bool:
    """
    Verify that train dates < test dates (no data leakage)
    
    Args:
        train_idx: Training indices
        test_idx: Test indices
        dates: Date series
        
    Returns:
        True if no leakage, False otherwise
    """
    train_dates = dates.iloc[train_idx]
    test_dates = dates.iloc[test_idx]
    
    max_train_date = train_dates.max()
    min_test_date = test_dates.min()
    
    return max_train_date < min_test_date


def visualize_splits(dates: pd.Series, n_splits: int = 5):
    """
    Print a visual representation of the train/test splits
    
    Useful for debugging and understanding the split strategy
    """
    validator = WalkForwardValidator(n_splits=n_splits)
    
    print(f"\n{'='*70}")
    print(f"Walk-Forward Cross-Validation Splits")
    print(f"{'='*70}\n")
    
    for fold, (train_idx, test_idx) in enumerate(validator.split(dates), 1):
        train_dates = dates.iloc[train_idx]
        test_dates = dates.iloc[test_idx]
        
        print(f"Fold {fold}:")
        print(f"  Train: {train_dates.min().date()} to {train_dates.max().date()} "
              f"({len(train_idx)} samples)")
        print(f"  Test:  {test_dates.min().date()} to {test_dates.max().date()} "
              f"({len(test_idx)} samples)")
        print(f"  Gap:   {(test_dates.min() - train_dates.max()).days} days")
        print(f"  Leakage: {'❌ DETECTED' if not test_no_leakage(train_idx, test_idx, dates) else '✓ None'}\n")

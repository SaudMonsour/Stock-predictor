"""
Test script for Walk-Forward Validation

Verifies time-aware cross-validation with no data leakage.
Tests on AAPL, NVDA, TSLA.
"""

import pandas as pd
import yfinance as yf
from walk_forward_validation import (
    WalkForwardValidator,
    walk_forward_split,
    test_no_leakage,
    visualize_splits
)


def test_walk_forward_validation(ticker: str):
    """Test walk-forward validation on a single stock"""
    print(f"\n{'='*70}")
    print(f"Testing Walk-Forward Validation on {ticker}")
    print(f"{'='*70}\n")
    
    # Download data
    print(f"📊 Downloading {ticker} data...")
    stock = yf.Ticker(ticker)
    df = stock.history(period='5y')
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    dates = df['Date']
    
    print(f"✓ Downloaded {len(df)} days of data")
    print(f"  Period: {dates.min().date()} to {dates.max().date()}")
    
    # Test walk-forward splitting
    print(f"\n🔍 Testing Walk-Forward Cross-Validation (5 folds)...")
    validator = WalkForwardValidator(n_splits=5, test_size_days=60)
    
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(validator.split(dates), 1):
        train_dates = dates.iloc[train_idx]
        test_dates = dates.iloc[test_idx]
        
        # Verify no leakage
        no_leakage = test_no_leakage(train_idx, test_idx, dates)
        fold_results.append({
            'fold': fold,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_start': train_dates.min(),
            'train_end': train_dates.max(),
            'test_start': test_dates.min(),
            'test_end': test_dates.max(),
            'no_leakage': no_leakage
        })
    
    # Display results
    print(f"\n✓ Generated {len(fold_results)} folds")
    print(f"\n{'Fold':<6} {'Train Size':<12} {'Test Size':<12} {'No Leakage':<12}")
    print("-" * 42)
    
    all_passed = True
    for result in fold_results:
        status = "✓ PASS" if result['no_leakage'] else "❌ FAIL"
        if not result['no_leakage']:
            all_passed = False
        
        print(f"{result['fold']:<6} {result['train_size']:<12} "
              f"{result['test_size']:<12} {status:<12}")
    
    # Detailed view
    visualize_splits(dates, n_splits=5)
    
    return all_passed


def test_no_leakage_manual():
    """Manual test to verify leakage detection works"""
    print(f"\n{'='*70}")
    print(f"Testing Leakage Detection")
    print(f"{'='*70}\n")
    
    # Create sample dates
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    dates_series = pd.Series(dates)
    
    # Test 1: No leakage (correct)
    train_idx = list(range(0, 1000))
    test_idx = list(range(1000, 1200))
    result1 = test_no_leakage(train_idx, test_idx, dates_series)
    print(f"Test 1 (No leakage): {'✓ PASS' if result1 else '❌ FAIL'}")
    
    # Test 2: With leakage (should fail)
    train_idx = list(range(0, 1100))  # Overlaps with test
    test_idx = list(range(1000, 1200))
    result2 = test_no_leakage(train_idx, test_idx, dates_series)
    print(f"Test 2 (With leakage): {'✓ PASS (correctly detected)' if not result2 else '❌ FAIL (missed leakage)'}")
    
    return result1 and not result2


def main():
    """Run all tests"""
    print("="*70)
    print("Walk-Forward Validation Test Suite")
    print("="*70)
    print(f"\nTesting on: AAPL, NVDA, TSLA")
    print(f"Cross-validation: 5 folds, 60-day test periods")
    
    # Test leakage detection
    print(f"\n[1] Testing leakage detection logic...")
    leakage_test_passed = test_no_leakage_manual()
    
    # Test on stocks
    test_results = {}
    for ticker in ['AAPL', 'NVDA', 'TSLA']:
        try:
            passed = test_walk_forward_validation(ticker)
            test_results[ticker] = passed
        except Exception as e:
            print(f"\n❌ Error testing {ticker}: {e}")
            test_results[ticker] = False
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Test Summary")
    print(f"{'='*70}\n")
    
    print(f"Leakage Detection: {'✓ PASS' if leakage_test_passed else '❌ FAIL'}")
    
    for ticker, passed in test_results.items():
        print(f"{ticker}: {'✓ PASS' if passed else '❌ FAIL'}")
    
    all_passed = leakage_test_passed and all(test_results.values())
    
    print(f"\n{'='*70}")
    if all_passed:
        print(f"✅ All Tests Passed!")
    else:
        print(f"❌ Some Tests Failed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

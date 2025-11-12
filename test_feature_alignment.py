"""
Quick test to verify inference features match training (81 features)
Run this before deploying inference server
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from inference_server import engineer_features_from_ohlcv

def generate_test_ohlcv(n_candles=300):
    """Generate synthetic M5 OHLCV data for testing"""
    dates = pd.date_range(
        end=datetime.now(),
        periods=n_candles,
        freq='5min'
    )
    
    # Random walk price
    np.random.seed(42)
    close = 1.1000 + np.cumsum(np.random.randn(n_candles) * 0.0001)
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n_candles) * 0.0001,
        'high': close + abs(np.random.randn(n_candles) * 0.0002),
        'low': close - abs(np.random.randn(n_candles) * 0.0002),
        'close': close,
        'volume': np.random.randint(100, 1000, n_candles)
    }, index=dates)
    
    return df

def test_feature_count():
    """Test that feature engineering produces exactly 81 features"""
    print("=" * 60)
    print("TESTING FEATURE ALIGNMENT")
    print("=" * 60)
    
    # Generate test data
    print("\n1. Generating test OHLCV data (300 M5 candles)...")
    df_ohlcv = generate_test_ohlcv(300)
    print(f"   ✓ Generated {len(df_ohlcv)} candles")
    
    # Engineer features
    print("\n2. Engineering features...")
    try:
        df_features, feature_cols = engineer_features_from_ohlcv(df_ohlcv, "EUR_USD")
        print(f"   ✓ Feature engineering completed")
    except Exception as e:
        print(f"   ✗ Feature engineering failed: {e}")
        return False
    
    # Check feature count
    print("\n3. Validating feature count...")
    expected_features = 81
    actual_features = len(feature_cols)
    
    print(f"   Expected: {expected_features} features")
    print(f"   Actual:   {actual_features} features")
    
    if actual_features == expected_features:
        print(f"   ✓ Feature count matches! ({actual_features} features)")
    else:
        print(f"   ✗ Feature count mismatch!")
        print(f"   Difference: {actual_features - expected_features}")
        return False
    
    # Check for expected feature types
    print("\n4. Checking feature types...")
    
    # Count base technical features (should have RSI, MACD, etc.)
    base_features = [f for f in feature_cols if not f.endswith('_H1') and not f.endswith('_H4') 
                     and f not in ['tau_pre', 'tau_post', 'weighted_surprise']]
    print(f"   Base technical: {len(base_features)} features")
    
    # Count MTF features (should have _H1 and _H4 suffix)
    mtf_features = [f for f in feature_cols if f.endswith('_H1') or f.endswith('_H4')]
    print(f"   Multi-timeframe: {len(mtf_features)} features")
    
    # Count macro features
    macro_features = [f for f in feature_cols if f in ['tau_pre', 'tau_post', 'weighted_surprise']]
    print(f"   Macro: {len(macro_features)} features")
    
    # Check for sentiment features (should be ZERO)
    sentiment_features = [f for f in feature_cols if 'sentiment' in f.lower() or 'polarity' in f.lower()]
    print(f"   Sentiment: {len(sentiment_features)} features")
    
    if len(sentiment_features) > 0:
        print(f"   ✗ ERROR: Found sentiment features but models not trained with them!")
        print(f"   Sentiment features: {sentiment_features}")
        return False
    
    # Validate breakdown
    print("\n5. Validating feature breakdown...")
    expected_breakdown = {
        'base': 67,
        'mtf': 14,
        'macro': 3,
        'sentiment': 0
    }
    
    actual_breakdown = {
        'base': len(base_features),
        'mtf': len(mtf_features),
        'macro': len(macro_features),
        'sentiment': len(sentiment_features)
    }
    
    all_match = True
    for feature_type, expected_count in expected_breakdown.items():
        actual_count = actual_breakdown[feature_type]
        status = "✓" if actual_count == expected_count else "✗"
        print(f"   {status} {feature_type.capitalize()}: {actual_count}/{expected_count}")
        if actual_count != expected_count:
            all_match = False
    
    if not all_match:
        print("\n   ✗ Feature breakdown doesn't match training!")
        return False
    
    # Success!
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nInference features now match training exactly:")
    print(f"  - {len(base_features)} base technical features (M5)")
    print(f"  - {len(mtf_features)} multi-timeframe features (H1 + H4)")
    print(f"  - {len(macro_features)} macro features")
    print(f"  - {len(sentiment_features)} sentiment features")
    print(f"  = {actual_features} total features ✓")
    print("\nInference server is ready for deployment!")
    
    return True

if __name__ == "__main__":
    success = test_feature_count()
    exit(0 if success else 1)

"""
Simple macro test with ZigZag targets
Load pre-calculated correlations from earlier analysis
"""
import pandas as pd
import numpy as np

print("="*80)
print("MACRO FEATURES WITH ZIGZAG TARGET - SIMPLE TEST")
print("="*80)

# Load the ZigZag data
print("\n[1/2] Loading ZigZag training data...")
df = pd.read_csv('zigzag_training_data.csv', index_col=0, parse_dates=True)
print(f"  ✓ Loaded {len(df):,} samples")

# Load the earlier correlation analysis (from analyze_feature_correlations.py)
print("\n[2/2] Loading earlier macro correlations...")
try:
    old_corrs = pd.read_csv('feature_target_correlations.csv')
    
    # Filter for macro features
    macro_features = old_corrs[old_corrs['feature'].str.contains('vix|rate_|yield|dxy|oil', case=False, na=False)]
    
    print("\n" + "="*80)
    print("MACRO FEATURES (from earlier analysis with OLD target)")
    print("="*80)
    print("\nTop macro features:")
    print(macro_features.sort_values('correlation', ascending=False).head(10).to_string(index=False))
    
    # Now compare with ZigZag base features
    print("\n" + "="*80)
    print("COMPARISON: MACROS vs ZIGZAG BASE FEATURES")
    print("="*80)
    
    print("\nZigZag Base Features (with ZigZag target):")
    print("  RSI:           33.2% correlation (direction)")
    print("  Candle body:    9.9% correlation (direction)")
    print("  MACD diff:      3.5% correlation (direction)")
    
    print("\nBest Macro Features (with OLD binary target):")
    for _, row in macro_features.head(5).iterrows():
        print(f"  {row['feature']:20s}: {row['correlation']*100:5.1f}% correlation")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    best_macro = macro_features['correlation'].max()
    worst_base = 0.0354  # MACD diff with ZigZag
    
    print(f"\nBest macro (old target):  {best_macro*100:.1f}%")
    print(f"Worst base (ZigZag):      {worst_base*100:.1f}%")
    
    # Estimate macro performance with ZigZag target
    # Assumption: If ZigZag improved RSI from 1.3% → 33.2% (25x improvement)
    # Then macros should improve similarly
    improvement_factor = 33.2 / 1.3  # RSI improvement with ZigZag
    
    print(f"\nEstimated macro with ZigZag target:")
    print(f"  Improvement factor: {improvement_factor:.1f}x (based on RSI)")
    
    for _, row in macro_features.head(5).iterrows():
        old_corr = row['correlation']
        estimated_new = old_corr * improvement_factor
        print(f"  {row['feature']:20s}: {old_corr*100:5.1f}% → {estimated_new*100:5.1f}% (estimated)")
    
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    best_macro_estimated = best_macro * improvement_factor
    
    if best_macro_estimated > 0.10:  # 10% threshold
        print(f"\n✓ MACROS MIGHT BE USEFUL")
        print(f"  Best macro estimated: {best_macro_estimated*100:.1f}%")
        print(f"  This beats MACD diff ({worst_base*100:.1f}%)")
        print(f"\n  RECOMMENDATION: Test adding VIX and yield_curve")
    else:
        print(f"\n✗ MACROS LIKELY NOT USEFUL")
        print(f"  Best macro estimated: {best_macro_estimated*100:.1f}%")
        print(f"  This is weaker than all base features")
        print(f"\n  RECOMMENDATION: Stick with engineer's 3 features")
    
except FileNotFoundError:
    print("  ✗ Could not find feature_target_correlations.csv")
    print("  Run analyze_feature_correlations.py first")
    
    # Fallback: Use known values from earlier
    print("\n" + "="*80)
    print("USING KNOWN VALUES FROM EARLIER ANALYSIS")
    print("="*80)
    
    print("\nMacro features (with OLD binary target):")
    print("  VIX:                3.2% correlation")
    print("  Yield curve:        2.6% correlation")
    print("  Rate differential:  1.9% correlation")
    
    print("\nZigZag Base Features:")
    print("  RSI:               33.2% correlation")
    print("  Candle body:        9.9% correlation")
    print("  MACD diff:          3.5% correlation")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # RSI improved 25x with ZigZag (1.3% → 33.2%)
    # If macros improve similarly:
    vix_estimated = 0.032 * 25.5  # 81.6%
    yield_estimated = 0.026 * 25.5  # 66.3%
    rate_estimated = 0.019 * 25.5  # 48.5%
    
    print(f"\nIF macros improve like RSI did (25x):")
    print(f"  VIX:                3.2% → {vix_estimated:.1f}% (unrealistic!)")
    print(f"  Yield curve:        2.6% → {yield_estimated:.1f}% (unrealistic!)")
    print(f"  Rate differential:  1.9% → {rate_estimated:.1f}% (unrealistic!)")
    
    print(f"\nBUT: Macros are fundamentally different from technical indicators")
    print(f"  - Technical indicators (RSI, MACD) react to ZigZag swings")
    print(f"  - Macros (rates, VIX) change slowly, independent of M5 swings")
    
    print(f"\nMore realistic estimate (5x improvement):")
    vix_realistic = 0.032 * 5  # 16%
    yield_realistic = 0.026 * 5  # 13%
    rate_realistic = 0.019 * 5  # 9.5%
    
    print(f"  VIX:                3.2% → {vix_realistic:.1f}%")
    print(f"  Yield curve:        2.6% → {yield_realistic:.1f}%")
    print(f"  Rate differential:  1.9% → {rate_realistic:.1f}%")
    
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    print(f"\nEven with 5x improvement:")
    print(f"  Best macro (VIX): ~16% correlation")
    print(f"  This beats MACD diff (3.5%) and candle body (9.9%)")
    print(f"  But much weaker than RSI (33.2%)")
    
    print(f"\n⚠️  UNCERTAIN - NEED ACTUAL TEST")
    print(f"\nRECOMMENDATION:")
    print(f"  1. Start with engineer's 3 features (proven to work)")
    print(f"  2. Train baseline model → get 60-65% accuracy")
    print(f"  3. Then test adding VIX as 4th feature")
    print(f"  4. If accuracy improves, keep it; otherwise remove")
    
    print(f"\nPRIORITY: Get the baseline working first!")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)

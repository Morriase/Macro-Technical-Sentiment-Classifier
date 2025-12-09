"""
ZigZag Extrema Calculator
Identifies peaks and troughs in price data for target creation

Based on senior engineer's approach from book (65% accuracy)
"""
import numpy as np
import pandas as pd
from loguru import logger


def calculate_zigzag_extrema(df, depth=48, deviation=1, backstep=47):
    """
    Calculate ZigZag extrema (peaks and troughs) for target creation
    
    This identifies significant price reversals by finding local maxima/minima
    that are separated by a minimum distance (backstep).
    
    Args:
        df: DataFrame with OHLC data (must have 'high' and 'low' columns)
        depth: Lookback window in bars (48 = 4 hours on M5)
        deviation: Minimum price change (not used in simple implementation)
        backstep: Minimum bars between extrema (prevents oscillation)
    
    Returns:
        DataFrame with added columns:
        - extremum_price: Price at extremum (NaN for non-extremum bars)
        - extremum_type: 'peak' or 'trough' (None for non-extremum bars)
        - next_extremum_price: Forward-filled next extremum price
        - next_extremum_type: Forward-filled next extremum type
    
    Example:
        >>> df = calculate_zigzag_extrema(df, depth=48, backstep=47)
        >>> # Each bar now knows the price of the NEXT extremum
        >>> # Target = direction and magnitude to next extremum
    """
    df = df.copy()
    
    logger.info(f"Calculating ZigZag extrema (depth={depth}, backstep={backstep})...")
    
    # Find local maxima and minima
    extrema = []
    
    for i in range(depth, len(df) - depth):
        # Check if current bar is a peak (local maximum)
        window_high = df['high'].iloc[i-depth:i+depth+1]
        if df['high'].iloc[i] == window_high.max():
            extrema.append({
                'index': i, 
                'price': df['high'].iloc[i], 
                'type': 'peak'
            })
        
        # Check if current bar is a trough (local minimum)
        window_low = df['low'].iloc[i-depth:i+depth+1]
        if df['low'].iloc[i] == window_low.min():
            extrema.append({
                'index': i, 
                'price': df['low'].iloc[i], 
                'type': 'trough'
            })
    
    logger.info(f"  Found {len(extrema)} raw extrema")
    
    # Filter extrema by backstep (prevent oscillation)
    filtered_extrema = []
    last_idx = -backstep - 1
    
    for ext in extrema:
        if ext['index'] - last_idx >= backstep:
            filtered_extrema.append(ext)
            last_idx = ext['index']
    
    logger.info(f"  Filtered to {len(filtered_extrema)} extrema (backstep={backstep})")
    
    # Create extremum columns
    df['extremum_price'] = np.nan
    df['extremum_type'] = None
    
    for ext in filtered_extrema:
        df.loc[df.index[ext['index']], 'extremum_price'] = ext['price']
        df.loc[df.index[ext['index']], 'extremum_type'] = ext['type']
    
    # Forward fill extremum (each bar knows the NEXT extremum)
    # This is the key: we're predicting the NEXT turning point
    df['next_extremum_price'] = df['extremum_price'].bfill()
    df['next_extremum_type'] = df['extremum_type'].bfill()
    
    # Count valid targets
    valid_targets = df['next_extremum_price'].notna().sum()
    logger.success(f"✓ ZigZag extrema calculated: {valid_targets:,} bars with valid targets")
    
    return df


def create_zigzag_targets(df, pip_multiplier=10000):
    """
    Create dual targets from ZigZag extrema
    
    Args:
        df: DataFrame with next_extremum_price column (from calculate_zigzag_extrema)
        pip_multiplier: Multiplier to convert price to pips (10000 for EUR/USD)
    
    Returns:
        DataFrame with added columns:
        - target_direction: 1 (buy) if next extremum is higher, 0 (sell) otherwise
        - target_magnitude: Distance to next extremum in pips
        - target_magnitude_norm: Normalized magnitude to [-1, 1]
    """
    df = df.copy()
    
    logger.info("Creating ZigZag-based targets...")
    
    # Target 1: Direction (1 = buy, 0 = sell)
    # If next extremum is higher than current close → Buy
    # If next extremum is lower than current close → Sell
    df['target_direction'] = (
        df['next_extremum_price'] > df['close']
    ).astype(int)
    
    # Target 2: Magnitude (distance to next extremum in pips)
    df['target_magnitude'] = (
        (df['next_extremum_price'] - df['close']) * pip_multiplier
    )
    
    # Normalize magnitude to [-1, 1] for neural network
    # Use 3-sigma range (99.7% of data)
    mag_mean = df['target_magnitude'].mean()
    mag_std = df['target_magnitude'].std()
    df['target_magnitude_norm'] = (
        (df['target_magnitude'] - mag_mean) / (mag_std * 3)
    ).clip(-1, 1)
    
    # Drop rows without valid targets
    initial_len = len(df)
    df.dropna(subset=['next_extremum_price'], inplace=True)
    dropped = initial_len - len(df)
    
    # Log statistics
    buy_count = df['target_direction'].sum()
    sell_count = len(df) - buy_count
    
    logger.info(f"  Direction distribution:")
    logger.info(f"    Buy (1):  {buy_count:,} ({buy_count/len(df)*100:.1f}%)")
    logger.info(f"    Sell (0): {sell_count:,} ({sell_count/len(df)*100:.1f}%)")
    logger.info(f"  Magnitude range: {df['target_magnitude'].min():.1f} to {df['target_magnitude'].max():.1f} pips")
    logger.info(f"  Magnitude mean: {df['target_magnitude'].mean():.1f} pips")
    logger.info(f"  Dropped {dropped:,} rows without valid targets")
    
    logger.success(f"✓ ZigZag targets created: {len(df):,} samples ready")
    
    return df


def validate_zigzag_quality(df, min_correlation=0.20):
    """
    Validate ZigZag target quality by checking feature correlations
    
    Args:
        df: DataFrame with features and targets
        min_correlation: Minimum expected correlation (0.20 = 20%)
    
    Returns:
        bool: True if quality is acceptable
    """
    logger.info("Validating ZigZag target quality...")
    
    # Check if required columns exist
    required_cols = ['target_direction', 'target_magnitude_norm']
    feature_cols = ['rsi_norm', 'macd_diff_norm', 'candle_body_norm']
    
    missing = [col for col in required_cols + feature_cols if col not in df.columns]
    if missing:
        logger.warning(f"Missing columns for validation: {missing}")
        return False
    
    # Calculate correlations
    correlations = {}
    for feature in feature_cols:
        corr_dir = np.corrcoef(
            df[feature].dropna().values,
            df.loc[df[feature].notna(), 'target_direction'].values
        )[0, 1]
        
        corr_mag = np.corrcoef(
            df[feature].dropna().values,
            df.loc[df[feature].notna(), 'target_magnitude_norm'].values
        )[0, 1]
        
        correlations[feature] = {
            'direction': abs(corr_dir),
            'magnitude': abs(corr_mag)
        }
    
    # Find best correlation
    best_feature = max(correlations.keys(), 
                      key=lambda k: correlations[k]['direction'])
    best_corr = correlations[best_feature]['direction']
    
    logger.info(f"  Best feature: {best_feature}")
    logger.info(f"  Direction correlation: {best_corr:.3f}")
    
    # Validate
    if best_corr >= min_correlation:
        logger.success(f"✓ ZigZag quality GOOD: {best_corr:.1%} correlation (>= {min_correlation:.1%})")
        return True
    else:
        logger.warning(f"⚠ ZigZag quality LOW: {best_corr:.1%} correlation (< {min_correlation:.1%})")
        logger.warning("  Consider adjusting ZigZag parameters (depth, backstep)")
        return False

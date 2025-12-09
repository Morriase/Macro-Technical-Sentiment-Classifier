"""
ZigZag Extrema Calculator - MQL5 Compatible Implementation
Identifies peaks and troughs in price data for target creation

Based on MQL5 ZigZag indicator logic (resources/zigzag.mq5)
"""
import numpy as np
import pandas as pd
from loguru import logger


def _highest(array: np.ndarray, depth: int, start: int) -> int:
    """
    Find index of highest value looking BACKWARD from start.
    MQL5 equivalent: Highest() function
    
    Args:
        array: Price array (highs)
        depth: How many bars to look back
        start: Current bar index
    
    Returns:
        Index of the highest bar in the lookback window
    """
    if start < 0:
        return 0
    
    max_val = array[start]
    index = start
    
    # Look BACKWARD only: from start-1 down to start-depth
    for i in range(start - 1, max(start - depth, -1), -1):
        if i >= 0 and array[i] > max_val:
            index = i
            max_val = array[i]
    
    return index


def _lowest(array: np.ndarray, depth: int, start: int) -> int:
    """
    Find index of lowest value looking BACKWARD from start.
    MQL5 equivalent: Lowest() function
    
    Args:
        array: Price array (lows)
        depth: How many bars to look back
        start: Current bar index
    
    Returns:
        Index of the lowest bar in the lookback window
    """
    if start < 0:
        return 0
    
    min_val = array[start]
    index = start
    
    # Look BACKWARD only: from start-1 down to start-depth
    for i in range(start - 1, max(start - depth, -1), -1):
        if i >= 0 and array[i] < min_val:
            index = i
            min_val = array[i]
    
    return index



# State machine constants (matching MQL5 EnSearchMode)
EXTREMUM = 0   # Searching for first extremum
PEAK = 1       # Searching for next ZigZag peak
BOTTOM = -1    # Searching for next ZigZag bottom


def calculate_zigzag_extrema(df: pd.DataFrame, depth: int = 48, 
                              deviation: int = 1, backstep: int = 47) -> pd.DataFrame:
    """
    Calculate ZigZag extrema using MQL5-compatible algorithm.
    
    This is a faithful Python port of the MQL5 ZigZag indicator that:
    1. Looks BACKWARD only (not forward) to find extrema
    2. Uses backstep to clear weaker extrema within recent bars
    3. Uses a state machine alternating between Peak and Bottom search
    4. Applies deviation filter for minimum price movement
    
    Args:
        df: DataFrame with OHLC data (must have 'high' and 'low' columns)
        depth: Lookback window in bars (48 = 4 hours on M5)
        deviation: Minimum deviation in points (1 = 1 pip for 5-digit broker)
        backstep: Bars to look back for clearing weaker extrema
    
    Returns:
        DataFrame with ZigZag columns added:
        - zigzag: Non-zero at extrema points (contains the price)
        - zigzag_high: Non-zero at peaks
        - zigzag_low: Non-zero at bottoms
        - extremum_type: 1 for peak, -1 for bottom, 0 otherwise
    """
    df = df.copy()
    high = df['high'].values
    low = df['low'].values
    n = len(df)
    
    logger.info(f"Calculating ZigZag extrema (depth={depth}, deviation={deviation}, backstep={backstep})...")
    
    if n < 100:
        logger.warning(f"Not enough data for ZigZag calculation (need 100, got {n})")
        df['zigzag'] = 0.0
        df['zigzag_high'] = 0.0
        df['zigzag_low'] = 0.0
        df['extremum_type'] = 0
        return df
    
    # Initialize buffers (matching MQL5)
    zigzag_buffer = np.zeros(n)
    high_map = np.zeros(n)
    low_map = np.zeros(n)
    
    # Point value for deviation calculation
    # For 5-digit brokers (EUR/USD), 1 point = 0.00001
    point = 0.00001
    
    last_high = 0.0
    last_low = 0.0
    
    # ========================================
    # PASS 1: Mark potential highs and lows
    # ========================================
    for shift in range(depth, n):
        # --- LOW extremum detection ---
        lowest_idx = _lowest(low, depth, shift)
        val = low[lowest_idx]
        
        if val == last_low:
            val = 0.0
        else:
            last_low = val
            # Check if current bar is far from the lowest (deviation filter)
            if (low[shift] - val) > deviation * point:
                val = 0.0
            else:
                # Backstep: clear weaker lows in recent bars
                for back in range(1, backstep + 1):
                    if shift - back >= 0:
                        res = low_map[shift - back]
                        if res != 0 and res > val:
                            low_map[shift - back] = 0.0
        
        # Mark this bar as low if it IS the lowest
        if low[shift] == val:
            low_map[shift] = val
        else:
            low_map[shift] = 0.0
        
        # --- HIGH extremum detection ---
        highest_idx = _highest(high, depth, shift)
        val = high[highest_idx]
        
        if val == last_high:
            val = 0.0
        else:
            last_high = val
            # Check if current bar is far from the highest (deviation filter)
            if (val - high[shift]) > deviation * point:
                val = 0.0
            else:
                # Backstep: clear weaker highs in recent bars
                for back in range(1, backstep + 1):
                    if shift - back >= 0:
                        res = high_map[shift - back]
                        if res != 0 and res < val:
                            high_map[shift - back] = 0.0
        
        # Mark this bar as high if it IS the highest
        if high[shift] == val:
            high_map[shift] = val
        else:
            high_map[shift] = 0.0

    
    # ========================================
    # PASS 2: Final selection using state machine
    # ========================================
    extreme_search = EXTREMUM
    last_low = 0.0
    last_high = 0.0
    last_low_pos = 0
    last_high_pos = 0
    
    for shift in range(depth, n):
        if extreme_search == EXTREMUM:
            # Looking for first extremum
            if last_low == 0.0 and last_high == 0.0:
                if high_map[shift] != 0:
                    last_high = high[shift]
                    last_high_pos = shift
                    extreme_search = BOTTOM  # Found peak, now search for bottom
                    zigzag_buffer[shift] = last_high
                
                if low_map[shift] != 0.0:
                    last_low = low[shift]
                    last_low_pos = shift
                    extreme_search = PEAK  # Found bottom, now search for peak
                    zigzag_buffer[shift] = last_low
        
        elif extreme_search == PEAK:
            # Searching for next peak (we have a bottom, looking for high)
            # Check if we found a LOWER low (update the bottom)
            if low_map[shift] != 0.0 and low_map[shift] < last_low and high_map[shift] == 0.0:
                # Found a lower low - move the bottom marker
                zigzag_buffer[last_low_pos] = 0.0
                last_low_pos = shift
                last_low = low_map[shift]
                zigzag_buffer[shift] = last_low
            
            # Check if we found a high (transition to searching for bottom)
            if high_map[shift] != 0.0 and low_map[shift] == 0.0:
                last_high = high_map[shift]
                last_high_pos = shift
                zigzag_buffer[shift] = last_high
                extreme_search = BOTTOM
        
        elif extreme_search == BOTTOM:
            # Searching for next bottom (we have a peak, looking for low)
            # Check if we found a HIGHER high (update the peak)
            if high_map[shift] != 0.0 and high_map[shift] > last_high and low_map[shift] == 0.0:
                # Found a higher high - move the peak marker
                zigzag_buffer[last_high_pos] = 0.0
                last_high_pos = shift
                last_high = high_map[shift]
                zigzag_buffer[shift] = last_high
            
            # Check if we found a low (transition to searching for peak)
            if low_map[shift] != 0.0 and high_map[shift] == 0.0:
                last_low = low_map[shift]
                last_low_pos = shift
                zigzag_buffer[shift] = last_low
                extreme_search = PEAK
    
    # ========================================
    # Create output columns
    # ========================================
    df['zigzag'] = zigzag_buffer
    df['zigzag_high'] = high_map
    df['zigzag_low'] = low_map
    
    # Create extremum_type: 1 for peak, -1 for bottom
    extremum_type = np.zeros(n)
    for i in range(n):
        if zigzag_buffer[i] != 0:
            if high_map[i] != 0:
                extremum_type[i] = 1  # Peak
            elif low_map[i] != 0:
                extremum_type[i] = -1  # Bottom
    df['extremum_type'] = extremum_type.astype(int)
    
    # Count extrema
    n_peaks = np.sum(extremum_type == 1)
    n_bottoms = np.sum(extremum_type == -1)
    logger.info(f"ZigZag found {n_peaks} peaks and {n_bottoms} bottoms ({n_peaks + n_bottoms} total extrema)")
    
    return df



def create_zigzag_targets(df: pd.DataFrame, pip_multiplier: int = 10000,
                          depth: int = None, deviation: int = None, 
                          backstep: int = None) -> pd.DataFrame:
    """
    Create training targets based on ZigZag extrema.
    
    For each bar, determines:
    1. Direction to next extremum (1=up to peak, 0=down to bottom)
    2. Magnitude (price distance to next extremum in pips)
    
    Args:
        df: DataFrame with OHLC data (must have 'zigzag' and 'extremum_type' columns
            from calculate_zigzag_extrema, OR will calculate them if depth/deviation/backstep provided)
        pip_multiplier: Multiplier to convert price to pips (10000 for EUR/USD, 100 for JPY pairs)
        depth, deviation, backstep: ZigZag parameters (only used if zigzag not already calculated)
    
    Returns:
        DataFrame with target columns:
        - target_direction: 1 (buy/up) or 0 (sell/down)
        - target_magnitude: Distance to next extremum in pips
        - bars_to_extremum: Number of bars until next extremum
    """
    # Check if ZigZag already calculated, if not calculate it
    if 'zigzag' not in df.columns or 'extremum_type' not in df.columns:
        if depth is None:
            from src.config import ZIGZAG_CONFIG
            depth = ZIGZAG_CONFIG['depth']
            deviation = ZIGZAG_CONFIG['deviation']
            backstep = ZIGZAG_CONFIG['backstep']
        df = calculate_zigzag_extrema(df, depth, deviation, backstep)
    
    n = len(df)
    close = df['close'].values
    zigzag = df['zigzag'].values
    extremum_type = df['extremum_type'].values
    
    # Initialize target arrays
    target_direction = np.full(n, np.nan)
    target_magnitude = np.full(n, np.nan)
    bars_to_extremum = np.full(n, np.nan)
    
    # Find all extremum indices
    extremum_indices = np.where(zigzag != 0)[0]
    
    if len(extremum_indices) < 2:
        logger.warning("Not enough extrema found for target creation")
        df['target_direction'] = target_direction
        df['target_magnitude'] = target_magnitude
        df['bars_to_extremum'] = bars_to_extremum
        return df
    
    logger.info(f"Creating targets from {len(extremum_indices)} extrema...")
    
    # For each bar, find the NEXT extremum and set targets
    ext_ptr = 0  # Pointer to current extremum in the list
    
    for i in range(n):
        # Move pointer to find next extremum after current bar
        while ext_ptr < len(extremum_indices) and extremum_indices[ext_ptr] <= i:
            ext_ptr += 1
        
        if ext_ptr >= len(extremum_indices):
            # No more extrema after this bar
            break
        
        next_ext_idx = extremum_indices[ext_ptr]
        next_ext_type = extremum_type[next_ext_idx]
        next_ext_price = zigzag[next_ext_idx]
        
        # Direction: 1 if next extremum is a peak (price going up), 0 if bottom (going down)
        target_direction[i] = 1 if next_ext_type == 1 else 0
        
        # Magnitude: distance from current close to extremum price (in pips)
        target_magnitude[i] = abs(next_ext_price - close[i]) * pip_multiplier
        
        # Bars until extremum
        bars_to_extremum[i] = next_ext_idx - i
    
    df['target_direction'] = target_direction
    df['target_magnitude'] = target_magnitude
    df['bars_to_extremum'] = bars_to_extremum
    
    # Log statistics
    valid_targets = ~np.isnan(target_direction)
    n_valid = np.sum(valid_targets)
    n_buy = np.sum(target_direction[valid_targets] == 1)
    n_sell = np.sum(target_direction[valid_targets] == 0)
    avg_magnitude = np.nanmean(target_magnitude)
    avg_bars = np.nanmean(bars_to_extremum)
    
    logger.info(f"Target statistics:")
    logger.info(f"  Valid samples: {n_valid} ({100*n_valid/n:.1f}%)")
    logger.info(f"  Buy signals: {n_buy} ({100*n_buy/n_valid:.1f}%)")
    logger.info(f"  Sell signals: {n_sell} ({100*n_sell/n_valid:.1f}%)")
    logger.info(f"  Avg magnitude: {avg_magnitude:.1f} pips")
    logger.info(f"  Avg bars to extremum: {avg_bars:.1f}")
    
    return df



def test_zigzag_quality(df: pd.DataFrame, feature_columns: list = None) -> dict:
    """
    Test the quality of ZigZag targets by computing correlations with features.
    
    This helps validate that the ZigZag implementation is working correctly.
    Good ZigZag targets should show meaningful correlations with technical indicators.
    
    Args:
        df: DataFrame with ZigZag targets and features
        feature_columns: List of feature columns to test (default: RSI, MACD)
    
    Returns:
        Dictionary with correlation statistics
    """
    if feature_columns is None:
        # Default features to test
        feature_columns = []
        for col in df.columns:
            if any(x in col.lower() for x in ['rsi', 'macd', 'ema', 'atr']):
                feature_columns.append(col)
    
    if 'target_direction' not in df.columns:
        logger.error("No target_direction column found. Run create_zigzag_targets first.")
        return {}
    
    results = {}
    valid_mask = ~df['target_direction'].isna()
    
    logger.info(f"\nFeature correlations with ZigZag target:")
    logger.info("-" * 50)
    
    for col in feature_columns:
        if col in df.columns:
            # Compute correlation
            corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, 'target_direction'])
            results[col] = corr
            logger.info(f"  {col}: {corr:.4f} ({abs(corr)*100:.1f}%)")
    
    # Sort by absolute correlation
    sorted_results = dict(sorted(results.items(), key=lambda x: abs(x[1]), reverse=True))
    
    if sorted_results:
        best_feature = list(sorted_results.keys())[0]
        best_corr = sorted_results[best_feature]
        logger.info(f"\nBest feature: {best_feature} with {abs(best_corr)*100:.1f}% correlation")
    
    return sorted_results


def validate_zigzag_quality(df: pd.DataFrame, min_correlation: float = 0.20) -> bool:
    """
    Validate that ZigZag targets have sufficient quality for training.
    
    Checks:
    1. At least one feature has correlation > min_correlation with target
    2. Buy/Sell distribution is reasonably balanced (40-60%)
    3. Sufficient number of extrema found
    
    Args:
        df: DataFrame with ZigZag targets and features
        min_correlation: Minimum required correlation (default 0.20 = 20%)
    
    Returns:
        True if quality checks pass, False otherwise
    """
    if 'target_direction' not in df.columns:
        logger.error("No target_direction column found. Run create_zigzag_targets first.")
        return False
    
    # Check 1: Feature correlations
    feature_columns = []
    for col in df.columns:
        if any(x in col.lower() for x in ['rsi', 'macd', 'norm', 'candle', 'yield', 'dxy']):
            feature_columns.append(col)
    
    correlations = test_zigzag_quality(df, feature_columns)
    
    if not correlations:
        logger.warning("No features found for correlation check")
        return False
    
    best_corr = max(abs(v) for v in correlations.values())
    corr_pass = best_corr >= min_correlation
    
    # Check 2: Buy/Sell balance
    valid_mask = ~df['target_direction'].isna()
    n_valid = valid_mask.sum()
    
    if n_valid == 0:
        logger.error("No valid targets found")
        return False
    
    buy_pct = df.loc[valid_mask, 'target_direction'].mean() * 100
    balance_pass = 40 <= buy_pct <= 60
    
    # Check 3: Extrema count
    n_extrema = (df['zigzag'] != 0).sum() if 'zigzag' in df.columns else 0
    extrema_pass = n_extrema >= 10  # At least 10 extrema
    
    # Log results
    logger.info(f"\nZigZag Quality Validation:")
    logger.info(f"  Best correlation: {best_corr*100:.1f}% {'✓' if corr_pass else '✗'} (need ≥{min_correlation*100:.0f}%)")
    logger.info(f"  Buy/Sell balance: {buy_pct:.1f}% buy {'✓' if balance_pass else '✗'} (need 40-60%)")
    logger.info(f"  Extrema count: {n_extrema} {'✓' if extrema_pass else '✗'} (need ≥10)")
    
    all_pass = corr_pass and balance_pass and extrema_pass
    
    if all_pass:
        logger.success("✓ ZigZag quality validation PASSED")
    else:
        logger.warning("✗ ZigZag quality validation FAILED")
    
    return all_pass


if __name__ == "__main__":
    """
    Test the ZigZag implementation with sample data
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from pathlib import Path
    from src.config import DATA_DIR, ZIGZAG_CONFIG
    
    # Load sample data
    data_file = DATA_DIR / "EUR_USD_M5.csv"
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        sys.exit(1)
    
    logger.info(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Ensure column names are lowercase
    df.columns = df.columns.str.lower()
    
    # Use a subset for testing
    df = df.tail(10000).reset_index(drop=True)
    logger.info(f"Testing with {len(df)} bars")
    
    # Calculate ZigZag targets
    df = create_zigzag_targets(
        df,
        depth=ZIGZAG_CONFIG['depth'],
        deviation=ZIGZAG_CONFIG['deviation'],
        backstep=ZIGZAG_CONFIG['backstep']
    )
    
    # Add RSI for correlation test
    try:
        import talib
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'].values)
        
        # Test quality
        test_zigzag_quality(df, ['rsi', 'macd'])
    except ImportError:
        logger.warning("TA-Lib not installed, skipping correlation test")
    
    # Show sample of results
    logger.info("\nSample ZigZag results:")
    extrema_df = df[df['zigzag'] != 0][['close', 'zigzag', 'extremum_type']].head(10)
    print(extrema_df)

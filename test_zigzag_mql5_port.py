"""
Test the MQL5-compatible ZigZag implementation.

This script validates that:
1. ZigZag algorithm produces reasonable extrema
2. Feature correlations are significantly better than random (>20%)
3. Buy/Sell distribution is balanced
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, ZIGZAG_CONFIG
from src.utils.zigzag import calculate_zigzag_extrema, create_zigzag_targets, test_zigzag_quality
from loguru import logger

def main():
    # Load EUR/USD M5 data (parquet format in kaggle_dataset)
    data_file = DATA_DIR / "kaggle_dataset" / "fx_data" / "EURUSD_M5.parquet"
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return
    
    logger.info(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)
    df.columns = df.columns.str.lower()
    
    # Use last 50,000 bars for testing (about 6 months of M5 data)
    df = df.tail(50000).reset_index(drop=True)
    logger.info(f"Testing with {len(df)} bars")
    
    # Get ZigZag config
    depth = ZIGZAG_CONFIG['depth']
    deviation = ZIGZAG_CONFIG['deviation']
    backstep = ZIGZAG_CONFIG['backstep']
    
    logger.info(f"\nZigZag Parameters:")
    logger.info(f"  Depth: {depth} bars ({depth * 5 / 60:.1f} hours on M5)")
    logger.info(f"  Deviation: {deviation} points")
    logger.info(f"  Backstep: {backstep} bars")
    
    # Calculate ZigZag targets (pip_multiplier=10000 for EUR/USD)
    df = create_zigzag_targets(df, pip_multiplier=10000, depth=depth, deviation=deviation, backstep=backstep)
    
    # Add technical indicators for correlation testing
    try:
        import talib
        
        # RSI - should have strong correlation with ZigZag
        df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'].values, df['low'].values, df['close'].values
        )
        
        # ATR
        df['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # EMA crossover
        df['ema_50'] = talib.EMA(df['close'].values, timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'].values, timeperiod=200)
        df['ema_cross'] = (df['ema_50'] - df['ema_200']) / df['close']
        
        # Bollinger Bands position
        upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20)
        df['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        logger.info("\nTechnical indicators calculated successfully")
        
    except ImportError:
        logger.warning("TA-Lib not installed, using simple indicators")
        # Simple RSI approximation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Test correlations
    feature_cols = [col for col in df.columns if col not in [
        'open', 'high', 'low', 'close', 'volume', 'time', 'datetime',
        'zigzag', 'zigzag_high', 'zigzag_low', 'extremum_type',
        'target_direction', 'target_magnitude', 'bars_to_extremum'
    ]]
    
    correlations = test_zigzag_quality(df, feature_cols)
    
    # Validation checks
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    
    # Check 1: RSI correlation should be > 20%
    rsi_corr = correlations.get('rsi_14', 0)
    rsi_pass = abs(rsi_corr) > 0.20
    logger.info(f"\n1. RSI Correlation: {abs(rsi_corr)*100:.1f}% {'✓ PASS' if rsi_pass else '✗ FAIL (need >20%)'}")
    
    # Check 2: At least one feature > 25%
    best_corr = max(abs(v) for v in correlations.values()) if correlations else 0
    best_pass = best_corr > 0.25
    logger.info(f"2. Best Feature Correlation: {best_corr*100:.1f}% {'✓ PASS' if best_pass else '✗ FAIL (need >25%)'}")
    
    # Check 3: Buy/Sell balance (should be 40-60%)
    valid_mask = ~df['target_direction'].isna()
    buy_pct = df.loc[valid_mask, 'target_direction'].mean() * 100
    balance_pass = 40 <= buy_pct <= 60
    logger.info(f"3. Buy/Sell Balance: {buy_pct:.1f}% buy {'✓ PASS' if balance_pass else '✗ FAIL (need 40-60%)'}")
    
    # Check 4: Reasonable number of extrema
    n_extrema = (df['zigzag'] != 0).sum()
    extrema_per_day = n_extrema / (len(df) / 288)  # 288 M5 bars per day
    extrema_pass = 2 <= extrema_per_day <= 20
    logger.info(f"4. Extrema per Day: {extrema_per_day:.1f} {'✓ PASS' if extrema_pass else '✗ FAIL (need 2-20)'}")
    
    # Check 5: Average magnitude should be meaningful (> 10 pips)
    avg_magnitude = df['target_magnitude'].mean()
    magnitude_pass = avg_magnitude > 10
    logger.info(f"5. Avg Magnitude: {avg_magnitude:.1f} pips {'✓ PASS' if magnitude_pass else '✗ FAIL (need >10)'}")
    
    # Overall result
    all_pass = rsi_pass and best_pass and balance_pass and extrema_pass and magnitude_pass
    logger.info("\n" + "=" * 60)
    if all_pass:
        logger.info("✓ ALL CHECKS PASSED - ZigZag implementation is working correctly!")
    else:
        logger.warning("✗ SOME CHECKS FAILED - Review the implementation")
    logger.info("=" * 60)
    
    # Save sample data for inspection
    sample_file = "zigzag_test_sample.csv"
    extrema_sample = df[df['zigzag'] != 0][['close', 'zigzag', 'extremum_type', 'rsi_14']].head(20)
    extrema_sample.to_csv(sample_file, index=True)
    logger.info(f"\nSample extrema saved to {sample_file}")
    
    return all_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

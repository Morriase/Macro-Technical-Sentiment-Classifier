"""
Test Full ZigZag Pipeline Integration
Test the complete pipeline with ZigZag approach on a small dataset
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from main import ForexClassifierPipeline
from src.config import PRIMARY_PAIR
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING FULL ZIGZAG PIPELINE INTEGRATION")
print("="*80)

try:
    # Initialize pipeline
    logger.info("\n[1/5] Initializing pipeline...")
    pipeline = ForexClassifierPipeline(currency_pair=PRIMARY_PAIR, use_kaggle_data=True)
    logger.success("✓ Pipeline initialized")

    # Fetch data (small sample for testing)
    logger.info("\n[2/5] Fetching data...")
    pipeline.fetch_data()
    
    # Limit to first 10K bars for quick test
    pipeline.df_price = pipeline.df_price.head(10000)
    logger.success(f"✓ Using {len(pipeline.df_price):,} bars for testing")

    # Engineer features (simplified 5 features)
    logger.info("\n[3/5] Engineering simplified features...")
    pipeline.engineer_features()
    logger.success(f"✓ Features engineered: {len(pipeline.df_features):,} samples")
    
    # Verify we have exactly 5 features
    feature_cols = ['rsi_norm', 'macd_diff_norm', 'candle_body_norm', 'yield_curve', 'dxy_index']
    for col in feature_cols:
        if col not in pipeline.df_features.columns:
            raise ValueError(f"Missing feature: {col}")
    logger.success(f"✓ All 5 features present: {', '.join(feature_cols)}")

    # Create ZigZag targets
    logger.info("\n[4/5] Creating ZigZag targets...")
    pipeline.create_target()
    logger.success(f"✓ ZigZag targets created: {len(pipeline.df_features):,} samples")
    
    # Verify targets exist
    required_targets = ['target_direction', 'target_magnitude', 'target_magnitude_norm', 'target_class']
    for col in required_targets:
        if col not in pipeline.df_features.columns:
            raise ValueError(f"Missing target: {col}")
    logger.success(f"✓ All targets present: {', '.join(required_targets)}")

    # Check data quality
    logger.info("\n[5/5] Validating data quality...")
    
    # Check for NaNs
    nan_counts = pipeline.df_features[feature_cols + required_targets].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Found NaNs:\n{nan_counts[nan_counts > 0]}")
    else:
        logger.success("✓ No NaNs in features or targets")
    
    # Check feature ranges (should be normalized to [-1, 1])
    for col in ['rsi_norm', 'macd_diff_norm', 'candle_body_norm']:
        min_val = pipeline.df_features[col].min()
        max_val = pipeline.df_features[col].max()
        logger.info(f"  {col}: [{min_val:.3f}, {max_val:.3f}]")
        if min_val < -1.5 or max_val > 1.5:
            logger.warning(f"  ⚠ {col} outside expected range [-1, 1]")
    
    # Check target distribution
    buy_count = pipeline.df_features['target_direction'].sum()
    sell_count = len(pipeline.df_features) - buy_count
    logger.info(f"  Target distribution: Buy={buy_count:,} ({buy_count/len(pipeline.df_features)*100:.1f}%), Sell={sell_count:,} ({sell_count/len(pipeline.df_features)*100:.1f}%)")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\n✓ ZIGZAG PIPELINE INTEGRATION TEST PASSED")
    print(f"\nData shape: {pipeline.df_features.shape}")
    print(f"Features (5): {', '.join(feature_cols)}")
    print(f"Targets (4): {', '.join(required_targets)}")
    print(f"\nReady for training!")
    
    # Save sample
    sample = pipeline.df_features[['close'] + feature_cols + required_targets].head(100)
    sample.to_csv('zigzag_pipeline_test_sample.csv')
    print(f"\n✓ Saved sample to: zigzag_pipeline_test_sample.csv")
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETE - READY FOR FULL TRAINING")
    print("="*80)
    
except Exception as e:
    print("\n" + "="*80)
    print("✗ TEST FAILED")
    print("="*80)
    logger.error(f"Error: {e}", exc_info=True)
    sys.exit(1)

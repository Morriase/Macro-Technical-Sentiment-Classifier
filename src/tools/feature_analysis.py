"""
Feature Analysis Tool
Fetches data from MT5 (2015+), computes features, and analyzes correlations.
"""
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.data_acquisition.mt5_data import MT5DataAcquisition
    from src.feature_engineering.technical_features import TechnicalFeatureEngineer
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

def analyze_features():
    # 1. Acquire Data
    logger.info("Initializing MT5 Data Acquisition...")
    mt5 = MT5DataAcquisition()
    if not mt5.is_connected:
        logger.error("MT5 not connected. Please ensure MetaTrader 5 terminal is installed and running.")
        logger.info("If you cannot run MT5, we can use OANDA or Kaggle data instead.")
        return

    symbol = "EURUSD"
    # User requested data from 2015
    start_date = datetime(2015, 1, 1)
    end_date = datetime.now()
    
    logger.info(f"Fetching {symbol} data from {start_date} to {end_date}...")
    df_raw = mt5.fetch_historical_data(symbol, start_date=start_date, end_date=end_date, timeframe="M5")
    
    if df_raw.empty:
        logger.error("No data fetched from MT5. Check symbol name or date range.")
        return

    logger.success(f"Fetched {len(df_raw)} rows of raw data.")

    # 2. Engineer Features
    logger.info("Computing technical features (including new indicators: Ichimoku, Keltner, etc)...")
    engineer = TechnicalFeatureEngineer()
    try:
        df_features = engineer.calculate_all_features(df_raw)
        df_features = engineer.calculate_feature_crosses(df_features)
    except Exception as e:
        logger.error(f"Feature calculation failed: {e}")
        return
    
    logger.success(f"Computed {len(df_features.columns)} features.")

    # 3. Correlation Analysis
    logger.info("Analyzing feature correlations...")
    # Select only numeric columns
    numeric_df = df_features.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS REPORT")
    print("="*60)
    print(f"Total Features Analyzed: {len(numeric_df.columns)}")
    print(f"High Correlation Threshold: 0.95")
    print("-" * 60)
    
    if not to_drop:
        print("No highly correlated features found > 0.95.")
    else:
        print(f"FOUND {len(to_drop)} REDUNDANT FEATURES:\n")
        for col in to_drop:
            # Find the feature it is correlated with
            correlated_with = upper[col][upper[col] > 0.95].index.tolist()
            print(f"  [X] {col:<30} correlated with: {correlated_with}")
            
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print(f"Consider dropping these {len(to_drop)} features to reduce model noise and training time.")
    print("="*60)
    
    # Save simple report
    report_path = project_root / "feature_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Feature Analysis Report - {datetime.now()}\n")
        f.write(f"Data: {symbol} (2015-Present)\n")
        f.write(f"Total Features: {len(df_features.columns)}\n\n")
        f.write("Redundant Features (>0.95 correlation):\n")
        for col in to_drop:
            correlated_with = upper[col][upper[col] > 0.95].index.tolist()
            f.write(f"- {col} <-> {correlated_with}\n")

    logger.info(f"Analysis complete. Report saved to {report_path}")
    mt5.close()

if __name__ == "__main__":
    analyze_features()

"""
Simple MT5 Data Download Script
Downloads FX data locally for multiple timeframes, saves to data folder
You upload to Kaggle later
"""
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta
import pytz
from src.data_acquisition.mt5_data import MT5DataAcquisition
import sys
from pathlib import Path

# Add project root to allow imports from src
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import CURRENCY_PAIRS


# Timeframes to download
TIMEFRAMES = ["M5", "H1", "H4"]

# Number of bars to fetch for each timeframe
# Adjust to get roughly the same amount of historical data
# M5: 80k bars ~ 13 months
# H1: 80k/12 ~ 6.7k bars
# H4: 80k/48 ~ 1.7k bars
BARS_PER_TIMEFRAME = {
    "M5": 80000,
    "H1": 7000,
    "H4": 2000 
}

# Output folder
OUTPUT_DIR = Path("data/kaggle_dataset/fx_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Connecting to MT5...")
mt5 = MT5DataAcquisition()

if not mt5.is_connected:
    logger.error("MT5 not connected! Make sure MT5 terminal is running.")
    exit(1)

# Convert "EUR_USD" to "EURUSD" format for MT5
symbols_mt5 = [pair.replace("_", "") for pair in CURRENCY_PAIRS]

logger.info(f"Downloading data for {len(symbols_mt5)} symbols across {len(TIMEFRAMES)} timeframes...")

for symbol in symbols_mt5:
    for timeframe in TIMEFRAMES:
        num_bars = BARS_PER_TIMEFRAME[timeframe]
        logger.info(f"Downloading {symbol} - {timeframe} ({num_bars} bars)...")

        df = mt5.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            count=num_bars
        )

        if df.empty:
            logger.warning(f"No data for {symbol} on {timeframe}")
            continue

        # Save to parquet
        filename = OUTPUT_DIR / f"{symbol}_{timeframe}.parquet"
        df.to_parquet(filename, compression="gzip")

        size_mb = filename.stat().st_size / (1024 * 1024)
        logger.success(f"✓ {symbol} ({timeframe}): {len(df):,} candles saved ({size_mb:.1f} MB)")

mt5.close()

logger.success(f"\n✅ ALL DATA SAVED TO: {OUTPUT_DIR}")
logger.info("Next: Upload this folder to Kaggle Datasets")

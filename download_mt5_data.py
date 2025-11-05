"""
Simple MT5 Data Download Script
Downloads FX data locally, saves to data folder
You upload to Kaggle later
"""
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta
import pytz
from src.data_acquisition.mt5_data import MT5DataAcquisition
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# Major pairs + Gold
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF",
           "AUDUSD", "USDCAD", "NZDUSD", "XAUUSD"]

# Number of bars to fetch (80,000 M5 bars = ~392 days / 13 months of 24/5 FX data)
# This is the maximum available from most MT5 brokers
# 80k bars at M5 = plenty of data for LSTM training
NUM_BARS = 80000

# Output folder
OUTPUT_DIR = Path("data/kaggle_dataset/fx_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Connecting to MT5...")
mt5 = MT5DataAcquisition()

if not mt5.is_connected:
    logger.error("MT5 not connected! Make sure MT5 terminal is running.")
    exit(1)

logger.info(f"Downloading data for {len(SYMBOLS)} symbols...")
logger.info(f"Fetching last {NUM_BARS} M5 bars per symbol")

for symbol in SYMBOLS:
    logger.info(f"Downloading {symbol}...")

    df = mt5.fetch_historical_data(
        symbol=symbol,
        timeframe="M5",
        count=NUM_BARS
    )

    if df.empty:
        logger.warning(f"No data for {symbol}")
        continue

    # Save to parquet
    filename = OUTPUT_DIR / f"{symbol}_M5.parquet"
    df.to_parquet(filename, compression="gzip")

    size_mb = filename.stat().st_size / (1024 * 1024)
    logger.success(f"✓ {symbol}: {len(df):,} candles saved ({size_mb:.1f} MB)")

mt5.close()

logger.success(f"\n✅ ALL DATA SAVED TO: {OUTPUT_DIR}")
logger.info("Next: Upload this folder to Kaggle Datasets")

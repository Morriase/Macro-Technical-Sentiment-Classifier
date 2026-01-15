"""
MetaTrader 5 Data Acquisition Module
Free alternative to OANDA for FX data
Works worldwide - no geographic restrictions
"""
from loguru import logger
from typing import Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


try:
    import MetaTrader5 as mt5
except ImportError:
    logger.warning("MetaTrader5 not installed. Run: pip install MetaTrader5")
    mt5 = None


class MT5DataAcquisition:
    """
    MetaTrader 5 data acquisition for FX prices
    Free and available worldwide
    """

    def __init__(self):
        """Initialize MT5 connection"""
        self.is_connected = False

        if mt5 is None:
            logger.error("MetaTrader5 module not available")
            return

        # Initialize MT5
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return

        self.is_connected = True
        terminal_info = mt5.terminal_info()
        logger.info(
            f"MT5 connected: {terminal_info.company}, Build {terminal_info.build}")

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime = None,
        end_date: datetime = None,
        timeframe: str = "M5",
        count: int = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from MT5

        Args:
            symbol: Currency pair (e.g., 'EURUSD', 'GBPUSD')
            start_date: Start date (optional if count is provided)
            end_date: End date (optional if count is provided)
            timeframe: Timeframe ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1')
            count: Number of bars to fetch from current time (alternative to date range)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected:
            logger.error("MT5 not connected")
            return pd.DataFrame()

        # Convert timeframe string to MT5 constant
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        if timeframe not in timeframe_map:
            logger.error(f"Invalid timeframe: {timeframe}")
            return pd.DataFrame()

        mt5_timeframe = timeframe_map[timeframe]

        try:
            # Enable symbol if not visible
            if not self.validate_symbol(symbol):
                logger.error(f"Symbol {symbol} not available")
                return pd.DataFrame()

            # Fetch rates using count method (more reliable for getting recent data)
            if count is not None:
                logger.info(
                    f"Fetching last {count} bars for {symbol} ({timeframe})")
                rates = mt5.copy_rates_from_pos(
                    symbol,
                    mt5_timeframe,
                    0,  # Start from current bar
                    count
                )
            else:
                # Fetch rates using date range with yearly chunking to avoid "Invalid params" / timeouts
                logger.info(
                    f"Fetching {symbol} data from {start_date} to {end_date} ({timeframe})")
                
                start_clean = start_date.replace(microsecond=0)
                end_clean = end_date.replace(microsecond=0)
                
                all_rates = []
                current_start = start_clean
                
                # Chunk logic: Monthly for minute data, Yearly for others to avoid limits
                while current_start < end_clean:
                    # Determine next chunk end
                    if timeframe in ["M1", "M5", "M15", "M30"]:
                        # Monthly chunks for high frequency data (~6-8k bars)
                        if current_start.month == 12:
                            chunk_end = datetime(current_start.year + 1, 1, 1)
                        else:
                            chunk_end = datetime(current_start.year, current_start.month + 1, 1)
                    else:
                        # Yearly chunks for hourly/daily data (~6k bars for H1)
                        chunk_end = datetime(current_start.year + 1, 1, 1)

                    # Cap at actual end date
                    if chunk_end > end_clean:
                        chunk_end = end_clean
                        
                    logger.info(f"  Fetching chunk: {current_start.date()} to {chunk_end.date()}")
                    
                    chunk_rates = mt5.copy_rates_range(
                        symbol,
                        mt5_timeframe,
                        current_start,
                        chunk_end,
                    )
                    
                    if chunk_rates is not None and len(chunk_rates) > 0:
                        all_rates.append(chunk_rates)
                    else:
                        # Only warn if it's not a generic "no data" (error check)
                        err = mt5.last_error()
                        if err[0] != 1: # 1 = Success
                             # For debug: log error code
                             logger.debug(f"  MT5 Code: {err}")
                        
                        # Don't spam warnings for every empty month (common in old history)
                        # logger.warning(f"  No data for chunk {current_start.date()} - {chunk_end.date()}")

                    current_start = chunk_end
                
                if not all_rates:
                    logger.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()
                
                # Concatenate all chunks
                rates = np.concatenate(all_rates)

            if len(rates) == 0:
                 return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(rates)

            # Convert timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Drop duplicates just in case
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)

            # Rename columns to match OANDA format
            df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
            }, inplace=True)

            # Select relevant columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            logger.success(f"Fetched {len(df)} candles for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol} data: {e}")
            return pd.DataFrame()

    def get_available_symbols(self) -> list:
        """Get list of available currency pairs"""
        if not self.is_connected:
            logger.error("MT5 not connected")
            return []

        symbols = mt5.symbols_get()
        if symbols is None:
            logger.error("Failed to get symbols")
            return []

        # Filter for forex pairs
        forex_symbols = [
            s.name for s in symbols
            if s.name.endswith('USD') or s.name.startswith('EUR') or
            s.name.startswith('GBP') or s.name.startswith('USD')
        ]

        logger.info(f"Found {len(forex_symbols)} forex symbols")
        return forex_symbols

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is available"""
        if not self.is_connected:
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"Symbol {symbol} not found")
            return False

        if not symbol_info.visible:
            # Try to enable symbol
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Failed to enable symbol {symbol}")
                return False

        logger.info(f"Symbol {symbol} is available")
        return True

    def validate_data_quality(self, df: pd.DataFrame) -> tuple:
        """
        Validate data quality

        Returns:
            (is_valid, accuracy_score)
        """
        if df.empty:
            return False, 0.0

        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))

        # Check for gaps in time series
        time_diffs = df.index.to_series().diff()
        expected_diff = time_diffs.mode()[0]
        gaps = (time_diffs > expected_diff * 2).sum()
        gap_pct = gaps / len(df)

        accuracy = 1.0 - (missing_pct + gap_pct)
        is_valid = accuracy >= 0.98

        return is_valid, accuracy

    def close(self):
        """Close MT5 connection"""
        if self.is_connected:
            mt5.shutdown()
            logger.info("MT5 connection closed")
            self.is_connected = False

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


if __name__ == "__main__":
    # Example usage
    mt5_client = MT5DataAcquisition()

    if mt5_client.is_connected:
        # List available symbols
        symbols = mt5_client.get_available_symbols()
        print(f"\nAvailable symbols: {symbols[:10]}...")

        # Fetch EUR/USD data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        df = mt5_client.fetch_historical_data(
            symbol="EURUSD",
            start_date=start_date,
            end_date=end_date,
            timeframe="M5",
        )

        if not df.empty:
            print(f"\nFetched {len(df)} candles")
            print(f"\nFirst 5 candles:")
            print(df.head())
            print(f"\nLast 5 candles:")
            print(df.tail())

            # Validate quality
            is_valid, accuracy = mt5_client.validate_data_quality(df)
            print(f"\nData quality: {'Valid' if is_valid else 'Invalid'}")
            print(f"Accuracy: {accuracy:.2%}")

        mt5_client.close()

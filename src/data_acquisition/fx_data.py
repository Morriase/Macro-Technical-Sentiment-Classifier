"""
Data acquisition module for FX price data
Supports OANDA and Dukascopy data sources
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from loguru import logger
import time

from src.config import (
    OANDA_API_KEY,
    OANDA_ACCOUNT_ID,
    FX_DATA_GRANULARITY,
    DATA_DIR,
)


class FXDataAcquisition:
    """
    High-fidelity FX price data acquisition with support for OANDA API
    Provides 1-minute to daily granularity OHLCV data
    """

    def __init__(self, api_key: str = OANDA_API_KEY, environment: str = "practice"):
        """
        Initialize FX data acquisition client

        Args:
            api_key: OANDA API key
            environment: 'practice' or 'live'
        """
        self.api_key = api_key
        self.environment = environment
        self.client = None

        if self.api_key:
            self.client = oandapyV20.API(
                access_token=api_key, environment=environment)
        else:
            logger.warning("No OANDA API key provided. Using demo mode.")

    def fetch_oanda_candles(
        self,
        instrument: str,
        granularity: str = "M5",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        count: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candle data from OANDA

        Args:
            instrument: Currency pair (e.g., 'EUR_USD')
            granularity: Candle timeframe ('M1', 'M5', 'H1', 'H4', 'D')
            start_date: Start datetime
            end_date: End datetime
            count: Number of candles to fetch (alternative to date range)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.client:
            raise ValueError("OANDA client not initialized. Provide API key.")

        logger.info(f"Fetching {instrument} data at {granularity} granularity")

        params = {
            "granularity": granularity,
            "price": "MBA",  # Mid, Bid, Ask
        }

        if count:
            params["count"] = min(count, 5000)  # OANDA max per request
        else:
            if start_date:
                params["from"] = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            if end_date:
                params["to"] = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        all_candles = []

        try:
            # Handle pagination for large requests
            while True:
                request = instruments.InstrumentsCandles(
                    instrument=instrument, params=params
                )
                response = self.client.request(request)

                candles = response.get("candles", [])
                if not candles:
                    break

                all_candles.extend(candles)

                # Check if we need to paginate
                if len(candles) < 5000:
                    break

                # Update start time for next batch
                last_time = candles[-1]["time"]
                params["from"] = last_time
                time.sleep(0.1)  # Rate limiting

        except Exception as e:
            logger.error(f"Error fetching OANDA data: {e}")
            raise

        # Parse candles into DataFrame
        df = self._parse_oanda_candles(all_candles)
        logger.info(f"Fetched {len(df)} candles for {instrument}")

        return df

    def _parse_oanda_candles(self, candles: List[Dict]) -> pd.DataFrame:
        """Parse OANDA candle response into DataFrame"""
        data = []

        for candle in candles:
            if not candle.get("complete"):
                continue  # Skip incomplete candles

            mid = candle["mid"]

            data.append({
                "timestamp": pd.to_datetime(candle["time"]),
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(candle.get("volume", 0)),
            })

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        return df

    def resample_to_timeframe(
        self, df: pd.DataFrame, target_timeframe: str = "4H"
    ) -> pd.DataFrame:
        """
        Resample high-frequency data to target timeframe

        Args:
            df: High-frequency OHLCV DataFrame
            target_timeframe: Target timeframe ('1H', '4H', 'D')

        Returns:
            Resampled DataFrame
        """
        logger.info(f"Resampling data to {target_timeframe}")

        resampled = df.resample(target_timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })

        # Remove rows with NaN values
        resampled.dropna(inplace=True)

        return resampled

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Validate data quality and calculate metrics

        Args:
            df: OHLCV DataFrame

        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_timestamps": df.index.duplicated().sum(),
            "negative_spreads": (df["high"] < df["low"]).sum(),
            "zero_volume": (df["volume"] == 0).sum(),
        }

        # Check for data gaps
        time_diffs = df.index.to_series().diff()
        expected_freq = pd.infer_freq(df.index)
        if expected_freq:
            expected_diff = pd.Timedelta(expected_freq)
            gaps = (time_diffs > expected_diff * 1.5).sum()
            metrics["time_gaps"] = gaps

        # Calculate data quality score
        total_issues = sum([
            metrics["missing_values"],
            metrics["duplicate_timestamps"],
            metrics["negative_spreads"],
        ])

        metrics["quality_score"] = 1.0 - (total_issues / max(len(df), 1))

        logger.info(f"Data quality metrics: {metrics}")

        return metrics

    def fill_gaps(self, df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """
        Fill gaps in time series data

        Args:
            df: OHLCV DataFrame with potential gaps
            method: Fill method ('ffill', 'bfill', 'interpolate')

        Returns:
            DataFrame with filled gaps
        """
        logger.info(f"Filling gaps using method: {method}")

        if method == "ffill":
            df_filled = df.fillna(method="ffill")
        elif method == "bfill":
            df_filled = df.fillna(method="bfill")
        elif method == "interpolate":
            df_filled = df.interpolate(method="time")
        else:
            raise ValueError(f"Unknown fill method: {method}")

        return df_filled

    def save_data(self, df: pd.DataFrame, instrument: str, timeframe: str):
        """Save processed data to disk"""
        filename = f"{instrument}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = DATA_DIR / "fx_prices" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(filepath, compression="gzip")
        logger.info(f"Saved data to {filepath}")

    def load_data(self, instrument: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load processed data from disk"""
        pattern = f"{instrument}_{timeframe}_*.parquet"
        files = sorted((DATA_DIR / "fx_prices").glob(pattern))

        if not files:
            logger.warning(f"No data found for {instrument} {timeframe}")
            return None

        # Load most recent file
        latest_file = files[-1]
        df = pd.read_parquet(latest_file)
        logger.info(f"Loaded data from {latest_file}")

        return df

    def get_multiple_pairs(
        self,
        instruments: List[str],
        granularity: str = "M5",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple currency pairs

        Args:
            instruments: List of currency pairs
            granularity: Candle timeframe
            start_date: Start datetime
            end_date: End datetime

        Returns:
            Dictionary mapping instruments to DataFrames
        """
        data = {}

        for instrument in instruments:
            try:
                df = self.fetch_oanda_candles(
                    instrument=instrument,
                    granularity=granularity,
                    start_date=start_date,
                    end_date=end_date,
                )

                # Validate quality
                quality = self.validate_data_quality(df)

                if quality["quality_score"] < 0.95:
                    logger.warning(
                        f"{instrument} data quality below threshold: {quality['quality_score']:.2%}"
                    )

                data[instrument] = df

            except Exception as e:
                logger.error(f"Failed to fetch {instrument}: {e}")
                continue

        return data


if __name__ == "__main__":
    # Example usage
    from loguru import logger

    logger.add("logs/fx_data_acquisition.log", rotation="1 day")

    # Initialize data acquisition
    fx_data = FXDataAcquisition()

    # Fetch EUR/USD data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    df = fx_data.fetch_oanda_candles(
        instrument="EUR_USD",
        granularity="M5",
        start_date=start_date,
        end_date=end_date,
    )

    # Resample to 4H
    df_4h = fx_data.resample_to_timeframe(df, "4H")

    # Save data
    fx_data.save_data(df_4h, "EUR_USD", "4H")

    print(f"Fetched {len(df_4h)} 4H candles")
    print(df_4h.head())

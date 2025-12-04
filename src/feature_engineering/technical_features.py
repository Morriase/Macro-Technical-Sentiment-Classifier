"""
Technical indicator feature engineering module - OPTIMIZED VERSION
Based on correlation analysis by senior ML engineer:
- RSI: highest correlation (0.40 direction, 0.22 magnitude)
- MACD difference: strong correlation with target

REMOVED (redundant/irrelevant per author's analysis):
- Stochastic: >0.70 correlation with RSI (redundant)
- Bollinger Bands: -0.76 correlation with RSI (redundant)
- CCI: >0.70 correlation with RSI (redundant)
- ATR: near-zero correlation with target (irrelevant)
- Lagged features: LSTM handles sequences natively
- Multiple volatility windows: keep only vol_10
"""
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional
from loguru import logger

from src.config import TECHNICAL_INDICATORS


class TechnicalFeatureEngineer:
    """
    OPTIMIZED Technical indicator calculation
    Reduced from 81 to ~27 features based on correlation analysis

    Core indicators (author's selection):
    - RSI: Highest predictive power
    - MACD: Strong directional signal
    - EMA 50/200: Trend context
    - ADX/DI: Trend strength (not redundant with oscillators)
    - Returns: Autoregressive signal
    - Single volatility measure
    """

    def __init__(self, config: Dict = TECHNICAL_INDICATORS):
        """
        Initialize technical feature engineer

        Args:
            config: Dictionary with indicator parameters
        """
        self.config = config

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate OPTIMIZED technical indicators (lean set)

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with ~15 base technical features
        """
        df = df.copy()

        # Ensure OHLCV columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"DataFrame must contain columns: {required_cols}")

        # Calculate LEAN indicator set
        df = self._calculate_moving_averages(df)
        df = self._calculate_momentum_indicators(df)
        df = self._calculate_trend_indicators(df)
        df = self._calculate_returns(df)

        # Drop NaN rows created by indicators
        df.dropna(inplace=True)

        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMAs - ONLY 50 and 200 (author's selection)
        EMA 100 removed as redundant
        """
        # Only calculate EMA 50 and 200 for trend context
        df["ema_50"] = talib.EMA(df["close"], timeperiod=50)
        df["ema_200"] = talib.EMA(df["close"], timeperiod=200)

        # Normalized distances (use close range instead of ATR - ATR is irrelevant per author)
        price_range = df["high"].rolling(
            20).max() - df["low"].rolling(20).min()
        price_range = price_range.replace(0, np.nan).ffill()

        df["ema_50_dist"] = (df["close"] - df["ema_50"]) / price_range
        df["ema_200_dist"] = (df["close"] - df["ema_200"]) / price_range

        # Golden/Death cross signal (50 vs 200)
        df["ema_cross"] = np.where(df["ema_50"] > df["ema_200"], 1, -1)

        return df

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators - RSI and MACD ONLY

        REMOVED per author's correlation analysis:
        - Stochastic: >0.70 correlation with RSI (redundant)
        """
        # RSI - Author's #1 indicator (0.40 direction correlation)
        rsi_period = self.config.get("RSI_PERIOD", 14)
        df["rsi"] = talib.RSI(df["close"], timeperiod=rsi_period)
        df["rsi_norm"] = df["rsi"] / 100.0  # Normalized 0-1

        # MACD - Author's #2 indicator
        macd_params = self.config.get("MACD_PERIODS", {
            "fastperiod": 12, "slowperiod": 26, "signalperiod": 9
        })
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
            df["close"],
            fastperiod=macd_params["fastperiod"],
            slowperiod=macd_params["slowperiod"],
            signalperiod=macd_params["signalperiod"],
        )

        # MACD difference (Main - Signal) - author's specific recommendation
        df["macd_diff"] = df["macd"] - df["macd_signal"]

        # MACD crossover signal
        df["macd_cross"] = np.where(df["macd"] > df["macd_signal"], 1, -1)

        return df

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend indicators - ADX and DI only

        REMOVED per author's analysis:
        - CCI: >0.70 correlation with RSI (redundant)
        """
        # ADX - Trend strength (not redundant with momentum oscillators)
        df["adx"] = talib.ADX(df["high"], df["low"],
                              df["close"], timeperiod=14)

        # Plus/Minus Directional Indicators
        df["plus_di"] = talib.PLUS_DI(
            df["high"], df["low"], df["close"], timeperiod=14)
        df["minus_di"] = talib.MINUS_DI(
            df["high"], df["low"], df["close"], timeperiod=14)

        # DI difference (directional signal)
        df["di_diff"] = df["plus_di"] - df["minus_di"]

        return df

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns and SINGLE volatility measure

        REMOVED:
        - return_22: too long horizon
        - log_return: redundant with simple return
        - Multiple volatility windows: keep only vol_10
        - Annualized volatility: unnecessary complexity
        """
        # Simple returns (autoregressive signals)
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)

        # Single volatility measure (10-period rolling std)
        df["realized_vol"] = df["return_1"].rolling(window=10).std()

        return df

    def add_multi_timeframe_features(
        self,
        df_primary: pd.DataFrame,
        higher_timeframes: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Adds SIMPLIFIED multi-timeframe features
        Only RSI, ADX, and regime classification (reduced from full indicator set)

        Args:
            df_primary: The main dataframe (e.g., M5) to add features to.
            higher_timeframes: Dict mapping timeframe names (e.g., "H1", "H4")
                               to their OHLCV dataframes.

        Returns:
            Primary dataframe with MTF features merged in.
        """
        logger.info("Adding simplified multi-timeframe features...")
        df_merged = df_primary.copy()

        for tf, df_ht in higher_timeframes.items():
            logger.info(f"Calculating lean features for {tf} timeframe...")

            df_ht_feats = pd.DataFrame(index=df_ht.index)

            # Only RSI and ADX from higher timeframes (per author's selection)
            df_ht_feats[f'rsi_{tf}'] = talib.RSI(df_ht['close'], timeperiod=14)
            df_ht_feats[f'adx_{tf}'] = talib.ADX(
                df_ht['high'], df_ht['low'], df_ht['close'], timeperiod=14
            )

            # Simple regime classification
            ema_50 = talib.EMA(df_ht['close'], timeperiod=50)
            ema_200 = talib.EMA(df_ht['close'], timeperiod=200)

            df_ht_feats[f'regime_trending_{tf}'] = (
                df_ht_feats[f'adx_{tf}'] > 25).astype(int)
            df_ht_feats[f'regime_bullish_{tf}'] = (
                ema_50 > ema_200).astype(int)

            # Ensure datetime index for merging
            if not isinstance(df_merged.index, pd.DatetimeIndex):
                df_merged.index = pd.to_datetime(df_merged.index)
            if not isinstance(df_ht_feats.index, pd.DatetimeIndex):
                df_ht_feats.index = pd.to_datetime(df_ht_feats.index)

            # Merge using asof (forward fill from higher timeframe)
            df_merged = pd.merge_asof(
                left=df_merged.sort_index(),
                right=df_ht_feats.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )
            logger.success(f"✓ Merged {tf} features (4 columns).")

        return df_merged

    def normalize_features_for_lstm(
        self, df: pd.DataFrame, feature_cols: List[str], scaler=None
    ) -> tuple:
        """
        Normalize features for LSTM input

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            scaler: Pre-fitted scaler (if None, will fit new one)

        Returns:
            Tuple of (normalized_df, scaler)
        """
        from sklearn.preprocessing import MinMaxScaler

        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = scaler.transform(df[feature_cols])

        return df, scaler

    def get_feature_list(self) -> List[str]:
        """
        Returns the expected feature column names for this optimized version.
        Useful for validation and documentation.

        Returns:
            List of feature column names (excluding OHLCV and target columns)
        """
        base_features = [
            # EMAs (3)
            "ema_50_dist", "ema_200_dist", "ema_cross",
            # RSI (2) - Author's #1
            "rsi", "rsi_norm",
            # MACD (5) - Author's #2
            "macd", "macd_signal", "macd_hist", "macd_diff", "macd_cross",
            # Trend (4)
            "adx", "plus_di", "minus_di", "di_diff",
            # Returns (3)
            "return_1", "return_5", "return_10",
            # Volatility (1)
            "realized_vol",
        ]

        mtf_features = [
            # H1 (4)
            "rsi_H1", "adx_H1", "regime_trending_H1", "regime_bullish_H1",
            # H4 (4)
            "rsi_H4", "adx_H4", "regime_trending_H4", "regime_bullish_H4",
        ]

        # Macro features are added separately by main.py
        # Can be either old (3): tau_pre, tau_post, weighted_surprise
        # Or new FRED (5+): rate_differential, vix, gdp_growth, unemployment_rate, inflation_rate

        return base_features + mtf_features


if __name__ == "__main__":
    # Example usage
    from src.data_acquisition.fx_data import FXDataAcquisition
    from datetime import datetime, timedelta

    # Fetch sample data
    fx_data = FXDataAcquisition()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    df = fx_data.fetch_oanda_candles(
        instrument="EUR_USD",
        granularity="H4",
        start_date=start_date,
        end_date=end_date,
    )

    # Calculate technical features
    tech_engineer = TechnicalFeatureEngineer()
    df_features = tech_engineer.calculate_all_features(df)

    print(f"\n=== OPTIMIZED Feature Set ===")
    print(f"Total features: {len(df_features.columns)} columns")
    print(f"\nExpected features: {tech_engineer.get_feature_list()}")
    print(f"\nActual columns: {df_features.columns.tolist()}")
    print(f"\n{df_features.tail()}")

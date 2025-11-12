"""
Technical indicator feature engineering module
Uses TA-Lib for calculating standardized technical indicators
"""
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional
from loguru import logger

from src.config import TECHNICAL_INDICATORS


class TechnicalFeatureEngineer:
    """
    Technical indicator calculation and feature engineering
    Implements EMA, RSI, ATR, MACD, Bollinger Bands, Stochastic
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
        Calculate all technical indicators

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with all technical features
        """
        df = df.copy()

        # Ensure OHLCV columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"DataFrame must contain columns: {required_cols}")

        # Calculate each indicator group
        df = self._calculate_moving_averages(df)
        df = self._calculate_momentum_indicators(df)
        df = self._calculate_volatility_indicators(df)
        df = self._calculate_trend_indicators(df)
        df = self._calculate_lagged_features(df)
        df = self._calculate_returns(df)

        # Drop NaN rows created by indicators
        df.dropna(inplace=True)

        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Exponential Moving Averages (EMA)"""
        ema_periods = self.config["EMA_PERIODS"]

        for period in ema_periods:
            col_name = f"ema_{period}"
            df[col_name] = talib.EMA(df["close"], timeperiod=period)

            # Calculate normalized distance from EMA
            # Normalized by ATR to make it scale-invariant
            atr_col = f"atr_{self.config['ATR_PERIOD']}"
            # Ensure ATR is calculated before using it for normalization
            if atr_col not in df.columns:
                df[atr_col] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=self.config['ATR_PERIOD'])
            
            df[f"ema_{period}_dist_norm"] = (
                df["close"] - df[col_name]) / df[atr_col]

        # EMA crossover signals
        if len(ema_periods) >= 2:
            # 50 x 100 crossover
            df["ema_50_100_cross"] = np.where(
                df["ema_50"] > df["ema_100"], 1, -1
            )

            # 50 x 200 crossover
            if "ema_200" in df.columns:
                df["ema_50_200_cross"] = np.where(
                    df["ema_50"] > df["ema_200"], 1, -1
                )
                # New feature: Price vs. EMA(200) - long-term trend filter
                df["price_vs_ema_200_dist_norm"] = (df["close"] - df["ema_200"]) / df[atr_col]
                
                # New feature: EMA(200) slope
                df["ema_200_slope"] = df["ema_200"].diff()


        return df

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators (RSI, Stochastic, MACD)"""

        # RSI
        rsi_period = self.config["RSI_PERIOD"]
        df["rsi"] = talib.RSI(df["close"], timeperiod=rsi_period)

        # Normalized RSI (0 to 1)
        df["rsi_norm"] = df["rsi"] / 100.0

        # RSI categorical signals
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)

        # Stochastic Oscillator
        stoch_params = self.config["STOCHASTIC_PERIODS"]
        df["stoch_k"], df["stoch_d"] = talib.STOCH(
            df["high"],
            df["low"],
            df["close"],
            fastk_period=stoch_params["fastk_period"],
            slowk_period=stoch_params["slowk_period"],
            slowd_period=stoch_params["slowd_period"],
        )

        # Stochastic crossover
        df["stoch_cross"] = np.where(df["stoch_k"] > df["stoch_d"], 1, -1)

        # MACD
        macd_params = self.config["MACD_PERIODS"]
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
            df["close"],
            fastperiod=macd_params["fastperiod"],
            slowperiod=macd_params["slowperiod"],
            signalperiod=macd_params["signalperiod"],
        )

        # MACD crossover
        df["macd_cross"] = np.where(df["macd"] > df["macd_signal"], 1, -1)

        return df

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators (ATR, Bollinger Bands)"""

        # Average True Range
        atr_period = self.config["ATR_PERIOD"]
        df[f"atr_{atr_period}"] = talib.ATR(
            df["high"], df["low"], df["close"], timeperiod=atr_period
        )

        # Normalized ATR (Z-Score relative to its own 50-period MA)
        atr_ma_period = 50
        atr_col = f"atr_{atr_period}"
        df["atr_ma"] = talib.SMA(df[atr_col], timeperiod=atr_ma_period)
        df["atr_std"] = df[atr_col].rolling(window=atr_ma_period).std()

        df["atr_zscore"] = (df[atr_col] - df["atr_ma"]) / df["atr_std"]
        df["atr_zscore"].fillna(0, inplace=True)

        # Bollinger Bands
        bb_period = self.config["BOLLINGER_PERIOD"]
        bb_std = self.config["BOLLINGER_STD"]

        df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
            df["close"],
            timeperiod=bb_period,
            nbdevup=bb_std,
            nbdevdn=bb_std,
        )

        # Bollinger Band Width (normalized)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # Price position within bands
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )
        df["bb_position"].clip(0, 1, inplace=True)

        return df

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators (ADX, etc.)"""

        # Average Directional Index
        df["adx"] = talib.ADX(df["high"], df["low"],
                              df["close"], timeperiod=14)

        # Plus/Minus Directional Indicators
        df["plus_di"] = talib.PLUS_DI(
            df["high"], df["low"], df["close"], timeperiod=14)
        df["minus_di"] = talib.MINUS_DI(
            df["high"], df["low"], df["close"], timeperiod=14)

        # Commodity Channel Index
        df["cci"] = talib.CCI(df["high"], df["low"],
                              df["close"], timeperiod=14)
        
        # New Feature: Strong uptrend/downtrend based on ADX and DIs
        # ADX > 25 (configurable threshold) indicates a strong trend
        adx_threshold = 25 
        df["strong_uptrend_adx"] = ((df["adx"] > adx_threshold) & (df["plus_di"] > df["minus_di"])).astype(int)
        df["strong_downtrend_adx"] = ((df["adx"] > adx_threshold) & (df["minus_di"] > df["plus_di"])).astype(int)

        return df

    def _calculate_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate lagged price features for autoregression"""

        lag_periods = [1, 2, 3, 5, 10]

        for lag in lag_periods:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

        return df

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return measures and realized volatility"""

        # Simple returns
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)
        df["return_22"] = df["close"].pct_change(22)

        # Log returns
        df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

        # Realized volatility (rolling std of returns)
        volatility_windows = [5, 10, 22, 50]

        for window in volatility_windows:
            df[f"realized_vol_{window}"] = df["return_1"].rolling(
                window=window).std()

            # Annualized volatility (assuming 252 trading days for daily data)
            # For 4H data, adjust accordingly
            df[f"realized_vol_{window}_ann"] = df[f"realized_vol_{window}"] * \
                np.sqrt(252)

        return df

    def add_multi_timeframe_features(
        self, 
        df_primary: pd.DataFrame, 
        higher_timeframes: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Adds multi-timeframe (MTF) features and regime classification to the primary dataframe.

        Args:
            df_primary: The main dataframe (e.g., M5 or H4) to add features to.
            higher_timeframes: A dictionary mapping timeframe names (e.g., "H1", "H4")
                               to their corresponding OHLCV dataframes.

        Returns:
            The primary dataframe with MTF features merged in.
        """
        logger.info("Adding multi-timeframe features and regime classification...")
        df_merged = df_primary.copy()

        for tf, df_ht in higher_timeframes.items():
            logger.info(f"Calculating features for {tf} timeframe...")
            
            # Calculate indicators on the higher timeframe data
            df_ht_feats = pd.DataFrame(index=df_ht.index)
            df_ht_feats[f'ema_50_{tf}'] = talib.EMA(df_ht['close'], timeperiod=50)
            df_ht_feats[f'ema_200_{tf}'] = talib.EMA(df_ht['close'], timeperiod=200)
            df_ht_feats[f'rsi_{tf}'] = talib.RSI(df_ht['close'], timeperiod=14)
            df_ht_feats[f'adx_{tf}'] = talib.ADX(df_ht['high'], df_ht['low'], df_ht['close'], timeperiod=14)
            
            # Calculate regime features for the higher timeframe
            df_ht_feats = self._calculate_regime_features(df_ht_feats, tf)

            # Ensure the index is a datetime index for merging
            if not isinstance(df_merged.index, pd.DatetimeIndex):
                df_merged.index = pd.to_datetime(df_merged.index)
            if not isinstance(df_ht_feats.index, pd.DatetimeIndex):
                df_ht_feats.index = pd.to_datetime(df_ht_feats.index)

            # Merge the higher timeframe features into the primary dataframe
            df_merged = pd.merge_asof(
                left=df_merged.sort_index(),
                right=df_ht_feats.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )
            logger.success(f"✓ Merged {tf} features.")

        return df_merged

    def _calculate_regime_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calculates market regime features based on indicators.

        Args:
            df: Dataframe containing indicators for a specific timeframe.
            timeframe: The timeframe string (e.g., "H4") to be used in column names.

        Returns:
            Dataframe with added regime features.
        """
        # Regime: Trending or Ranging
        adx_col = f'adx_{timeframe}'
        df[f'regime_trending_{timeframe}'] = (df[adx_col] > 25).astype(int)
        df[f'regime_ranging_{timeframe}'] = (df[adx_col] <= 25).astype(int)

        # Regime: Bullish or Bearish Trend
        ema_50_col = f'ema_50_{timeframe}'
        ema_200_col = f'ema_200_{timeframe}'
        df[f'regime_bullish_{timeframe}'] = (df[ema_50_col] > df[ema_200_col]).astype(int)
        df[f'regime_bearish_{timeframe}'] = (df[ema_50_col] < df[ema_200_col]).astype(int)
        
        return df

    def calculate_feature_crosses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate interaction features (crosses between indicators)
        Captures complex market states
        """
        # RSI × ATR (momentum during high volatility)
        if "rsi_norm" in df.columns and "atr_zscore" in df.columns:
            df["rsi_atr_cross"] = df["rsi_norm"] * df["atr_zscore"]

        # MACD × Volatility
        if "macd_hist" in df.columns and "realized_vol_10" in df.columns:
            df["macd_vol_cross"] = df["macd_hist"] * df["realized_vol_10"]

        # Bollinger position × RSI (overbought/oversold confirmation)
        if "bb_position" in df.columns and "rsi_norm" in df.columns:
            df["bb_rsi_cross"] = df["bb_position"] * df["rsi_norm"]

        return df

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
    df_features = tech_engineer.calculate_feature_crosses(df_features)

    print(f"Features calculated: {len(df_features.columns)} columns")
    print(df_features.columns.tolist())
    print(df_features.tail())

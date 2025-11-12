"""
Macroeconomic event data acquisition module
Fetches high-impact economic calendar events and calculates surprise factors
Uses TradingView's free economic calendar API
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import requests
from loguru import logger

from src.config import (
    HIGH_IMPACT_EVENTS,
    DATA_DIR,
)


class MacroDataAcquisition:
    """
    Acquisition and processing of macroeconomic calendar events from TradingView
    Calculates surprise factors and temporal proximity features
    """

    def __init__(self):
        """Initialize macro data acquisition client"""
        self.base_url = 'https://economic-calendar.tradingview.com/events'
        self.headers = {'Origin': 'https://in.tradingview.com'}
        logger.info(
            "MacroDataAcquisition initialized with TradingView calendar API")

    def fetch_economic_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
        country: str = "US",
    ) -> pd.DataFrame:
        """
        Fetch economic calendar events from TradingView (FREE API)

        Args:
            start_date: Start date
            end_date: End date
            country: Country code ('US', 'GB', 'EU', 'JP', 'CH', 'AU', 'CA', 'NZ')

        Returns:
            DataFrame with economic events
        """
        logger.info(
            f"Fetching economic calendar for {country} from {start_date.date()} to {end_date.date()}")

        try:
            # TradingView API call
            payload = {
                'from': start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'to': end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'countries': country,
                # High importance only (1=low, 2=medium, 3=high)
                'minImportance': 3
            }

            response = requests.get(
                self.base_url, headers=self.headers, params=payload, timeout=30)

            if response.status_code != 200:
                logger.error(
                    f"API returned status code {response.status_code}")
                return pd.DataFrame()

            data = response.json()

            if 'result' not in data or not data['result']:
                logger.warning(f"No events found for {country}")
                return pd.DataFrame()

            # Parse events
            df = pd.DataFrame(data['result'])

            if df.empty:
                return df

            # Parse timestamp
            df['date'] = pd.to_datetime(df['date'], unit='s')

            # Standardize column names
            df = df.rename(columns={
                'title': 'event_name',
                'actual': 'actual_value',
                'forecast': 'consensus_forecast',
                'previous': 'previous_value',
                'importance': 'impact_level',
            })

            # Calculate surprise factor
            if 'actual_value' in df.columns and 'consensus_forecast' in df.columns:
                df['surprise_factor'] = (
                    (df['actual_value'] - df['consensus_forecast']) /
                    (df['consensus_forecast'].abs() + 0.01)
                ).fillna(0)
            else:
                df['surprise_factor'] = 0

            logger.success(
                f"Fetched {len(df)} high-impact events for {country}")

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching economic calendar: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return pd.DataFrame()

    def calculate_surprise_zscore(
        self,
        df: pd.DataFrame,
        lookback_window: int = 52,
    ) -> pd.DataFrame:
        """
        Calculate standardized surprise Z-score for each event

        Surprise Z-Score = (Actual - Consensus) / σ(Historical Forecast Errors)

        Args:
            df: DataFrame with economic events
            lookback_window: Number of historical releases to use for std calculation

        Returns:
            DataFrame with surprise_zscore column
        """
        logger.info("Calculating surprise Z-scores")

        if df.empty or "actual_value" not in df.columns:
            return df

        # Calculate forecast error
        df["forecast_error"] = df["actual_value"] - df["consensus_forecast"]

        # Group by event type and calculate rolling std
        df["surprise_zscore"] = 0.0

        for event_name in df["event_name"].unique():
            mask = df["event_name"] == event_name
            event_df = df[mask].copy()

            # Calculate rolling std of forecast errors
            rolling_std = event_df["forecast_error"].rolling(
                window=lookback_window,
                min_periods=5,
            ).std()

            # Calculate Z-score
            zscore = event_df["forecast_error"] / rolling_std

            # Fill NaN with simple normalization
            if zscore.isna().any():
                mean_error = event_df["forecast_error"].mean()
                std_error = event_df["forecast_error"].std()
                if std_error > 0:
                    zscore = zscore.fillna(
                        (event_df["forecast_error"] - mean_error) / std_error
                    )

            df.loc[mask, "surprise_zscore"] = zscore.fillna(0)

        # Cap extreme values
        df["surprise_zscore"] = df["surprise_zscore"].clip(-5, 5)

        return df

    def calculate_temporal_proximity(
        self,
        events_df: pd.DataFrame,
        price_df: pd.DataFrame,
        pre_event_hours: int = 24,
        post_event_hours: int = 48,
        decay_lambda: float = 0.1,
    ) -> pd.DataFrame:
        """
        Calculate temporal proximity features for each price bar

        τ_pre: Inverse proximity before event (0 to 1, peaks at event time)
        τ_post: Exponential decay after event (starts at 1, decays)

        Args:
            events_df: DataFrame with economic events
            price_df: DataFrame with OHLCV price data
            pre_event_hours: Hours before event to track anticipation
            post_event_hours: Hours after event to track influence
            decay_lambda: Decay rate for post-event influence

        Returns:
            DataFrame with proximity features aligned to price bars
        """
        logger.info("Calculating temporal proximity features")

        if events_df.empty:
            # Return empty features
            price_df["tau_pre"] = 0.0
            price_df["tau_post"] = 0.0
            price_df["weighted_surprise"] = 0.0
            return price_df

        # Initialize columns
        price_df["tau_pre"] = 0.0
        price_df["tau_post"] = 0.0
        price_df["weighted_surprise"] = 0.0

        # For each event, calculate proximity for all price bars
        for idx, event in events_df.iterrows():
            event_time = event["timestamp"]
            surprise = event.get("surprise_zscore", 0.0)

            # Skip if surprise is NaN
            if pd.isna(surprise):
                continue

            # Pre-event proximity (anticipation)
            pre_mask = (
                (price_df.index < event_time) &
                (price_df.index >= event_time - pd.Timedelta(hours=pre_event_hours))
            )

            if pre_mask.any():
                hours_to_event = (
                    event_time - price_df[pre_mask].index).total_seconds() / 3600
                # Inverse proximity (closer to event = higher value)
                tau_pre_values = 1.0 / (1.0 + hours_to_event / pre_event_hours)
                price_df.loc[pre_mask, "tau_pre"] = np.maximum(
                    price_df.loc[pre_mask, "tau_pre"],
                    tau_pre_values
                )

            # Post-event influence (exponential decay)
            post_mask = (
                (price_df.index >= event_time) &
                (price_df.index < event_time + pd.Timedelta(hours=post_event_hours))
            )

            if post_mask.any():
                hours_since_event = (
                    price_df[post_mask].index - event_time).total_seconds() / 3600
                # Exponential decay
                tau_post_values = np.exp(-decay_lambda * hours_since_event)

                # Weight surprise by decay
                weighted_surprise = surprise * tau_post_values

                price_df.loc[post_mask, "tau_post"] = np.maximum(
                    price_df.loc[post_mask, "tau_post"],
                    tau_post_values
                )

                price_df.loc[post_mask,
                             "weighted_surprise"] += weighted_surprise

        logger.info("Temporal proximity features calculated")

        return price_df

    def get_events_for_currency_pair(
        self,
        pair: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch events for both currencies in a pair

        Args:
            pair: Currency pair (e.g., 'EUR_USD')
            start_date: Start date
            end_date: End date

        Returns:
            Combined DataFrame with events for both currencies
        """
        # Map currency codes to countries
        currency_country_map = {
            "EUR": "EU",
            "USD": "US",
            "GBP": "GB",
            "JPY": "JP",
            "AUD": "AU",
            "CAD": "CA",
            "CHF": "CH",
            "NZD": "NZ",
        }

        # Parse pair
        base_currency = pair[:3]
        quote_currency = pair[-3:]

        all_events = []

        for currency, code in [(base_currency, "base"), (quote_currency, "quote")]:
            country = currency_country_map.get(currency)

            if not country:
                logger.warning(f"Unknown country for currency: {currency}")
                continue

            events = self.fetch_economic_calendar(
                start_date=start_date,
                end_date=end_date,
                country=country,
            )

            if not events.empty:
                events["currency"] = currency
                events["currency_role"] = code
                all_events.append(events)

        if not all_events:
            return pd.DataFrame()

        # Combine events
        combined_df = pd.concat(all_events, ignore_index=True)
        combined_df.sort_values("timestamp", inplace=True)

        # Calculate surprise Z-scores
        combined_df = self.calculate_surprise_zscore(combined_df)

        return combined_df

    def save_events(self, df: pd.DataFrame, pair: str):
        """Save processed events to disk"""
        filename = f"{pair}_events_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = DATA_DIR / "macro_events" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(filepath, compression="gzip")
        logger.info(f"Saved events to {filepath}")

    def load_events(self, pair: str) -> Optional[pd.DataFrame]:
        """Load processed events from disk"""
        pattern = f"{pair}_events_*.parquet"
        files = sorted((DATA_DIR / "macro_events").glob(pattern))

        if not files:
            logger.warning(f"No events found for {pair}")
            return None

        # Load most recent file
        latest_file = files[-1]
        df = pd.read_parquet(latest_file)
        logger.info(f"Loaded events from {latest_file}")

        return df


if __name__ == "__main__":
    # Example usage
    from loguru import logger

    logger.add("logs/macro_data_acquisition.log", rotation="1 day")

    # Initialize macro data acquisition
    macro_data = MacroDataAcquisition()

    # Fetch events for EUR/USD
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    events_df = macro_data.get_events_for_currency_pair(
        pair="EUR_USD",
        start_date=start_date,
        end_date=end_date,
    )

    print(f"Fetched {len(events_df)} events")
    print(events_df[["timestamp", "event_name",
          "currency", "surprise_zscore"]].head(10))

    # Save events
    macro_data.save_events(events_df, "EUR_USD")

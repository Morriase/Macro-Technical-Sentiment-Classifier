"""
FRED (Federal Reserve Economic Data) Macro Data Loader

Fetches REAL historical macroeconomic data from the Federal Reserve for training.
Replaces synthetic/AI-generated macro features with actual economic indicators.

FRED API: https://fred.stlouisfed.org/docs/api/fred/
- Free API with generous limits
- High-quality, official economic data
- Historical data back to 1950s for most series

Key Features:
- Currency-pair specific macro indicators
- Rate differentials (Fed vs ECB, BoE, BoJ, etc.)
- Economic health indicators (GDP, Inflation, Unemployment)
- Leading indicators (PMI, Consumer Confidence)
"""
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
from pathlib import Path
import json
import time
from functools import lru_cache

# Try to import from config
try:
    from src.config import DATA_DIR, CURRENCY_PAIRS, IS_KAGGLE
except ImportError:
    DATA_DIR = Path("./data")
    CURRENCY_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY",
                      "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD"]
    IS_KAGGLE = False

# Cache directory - use writable location on Kaggle
if IS_KAGGLE:
    FRED_CACHE_DIR = Path("/kaggle/working/fred_cache")
else:
    FRED_CACHE_DIR = DATA_DIR / "fred_cache"


# =============================================================================
# FRED SERIES CONFIGURATION
# Maps each currency's central bank rates and economic indicators
# =============================================================================

# Central Bank Policy Rates
CENTRAL_BANK_RATES = {
    "USD": "FEDFUNDS",           # Federal Funds Rate (Fed)
    # ECB Deposit Facility Rate (fallback: INTDSREZM193N)
    "EUR": "ECBDFR",
    # Bank of England Rate (fallback: INTDSRGBM193N)
    "GBP": "BOERUKM",
    "JPY": "IRSTCB01JPM156N",    # Japan Policy Rate (BoJ)
    # RBA Cash Rate Target (fallback: INTDSRNUM193N)
    "AUD": "RBATCTR",
    "CAD": "INTDSRCAM193N",      # Bank of Canada Rate
    "CHF": "INTDSRCHM193N",      # SNB Policy Rate
    # RBNZ Official Cash Rate (fallback: INTDSRNZM193N)
    "NZD": "RBANZOCR",
}

# Fallback series if primary not available
CENTRAL_BANK_RATES_FALLBACK = {
    "EUR": "INTDSREZM193N",
    "GBP": "INTDSRGBM193N",
    "AUD": "INTDSRAUM193N",
    "NZD": "INTDSRNZM193N",
}

# Economic Health Indicators by Country
ECONOMIC_INDICATORS = {
    # United States
    "US": {
        # Real GDP Growth Rate (Quarterly)
        "gdp_growth": "A191RL1Q225SBEA",
        "inflation_cpi": "CPIAUCSL",           # CPI All Urban Consumers
        "unemployment": "UNRATE",              # Unemployment Rate
        # Manufacturing Employment (proxy for PMI)
        "pmi_manufacturing": "MANEMP",
        "retail_sales": "RSXFS",               # Retail Sales
        "industrial_production": "INDPRO",     # Industrial Production Index
        "consumer_confidence": "UMCSENT",      # U of Michigan Consumer Sentiment
        "nfp": "PAYEMS",                       # Non-Farm Payrolls
        "trade_balance": "BOPGSTB",            # Trade Balance
        "housing_starts": "HOUST",             # Housing Starts
    },
    # Eurozone
    "EU": {
        # FIXED: Updated to currently active FRED series
        "gdp_growth": "CPMNACSCAB1GQEL",      # EA GDP Constant Prices (FIXED from CLVMNACSCAB1GQEA19)
        "inflation_cpi": "CP0000EZ19M086NEST", # EA HICP (FIXED from EA19CPALTT01GYM)
        "unemployment": "LRHUTTTTEZM156S",     # EA Unemployment Rate
        "industrial_production": "EA19PRINTO01IXOBM",  # EA Industrial Production
        "consumer_confidence": "CSCICP03EZM460S",      # Consumer Confidence
    },
    # United Kingdom
    "GB": {
        "gdp_growth": "UKNGDP",               # UK GDP
        "inflation_cpi": "GBRCPIALLMINMEI",    # UK CPI
        "unemployment": "LMUNRRTTGBM156S",     # UK Unemployment
        "industrial_production": "GBRPRINTO01IXOBM",   # UK Industrial Production
        "retail_sales": "UKRETASMMUM",         # UK Retail Sales
    },
    # Japan
    "JP": {
        "gdp_growth": "JPNRGDPEXP",           # Japan Real GDP
        "inflation_cpi": "JPNCPIALLMINMEI",    # Japan CPI
        "unemployment": "LRUNTTTTJPM156S",     # Japan Unemployment
        "industrial_production": "JPNPROINDMISMEI",    # Japan Industrial Production
    },
    # Australia
    "AU": {
        "gdp_growth": "AUSGDPRQPSMEI",        # Australia GDP
        "inflation_cpi": "AUSCPIALLQINMEI",    # Australia CPI (Quarterly)
        "unemployment": "LRUNTTTTAUM156S",     # Australia Unemployment
    },
    # Canada
    "CA": {
        "gdp_growth": "CANRGDPR",             # Canada Real GDP
        "inflation_cpi": "CPALTT01CAM657N",    # Canada CPI
        "unemployment": "LRUNTTTTCAM156S",     # Canada Unemployment
    },
    # Switzerland
    "CH": {
        "inflation_cpi": "CHECPIALLMINMEI",    # Switzerland CPI
        "unemployment": "LRUNTTTTCHM156S",     # Switzerland Unemployment
    },
    # New Zealand
    "NZ": {
        "gdp_growth": "NZLGDPRPCDSMEI",       # New Zealand GDP
        "inflation_cpi": "NZLCPIALLQINMEI",    # New Zealand CPI (Quarterly)
        "unemployment": "LRUNTTTNZM156S",      # New Zealand Unemployment
    },
}

# Currency to Country mapping
CURRENCY_TO_COUNTRY = {
    "USD": "US",
    "EUR": "EU",
    "GBP": "GB",
    "JPY": "JP",
    "AUD": "AU",
    "CAD": "CA",
    "CHF": "CH",
    "NZD": "NZ",
    "XAU": "US",  # Gold trades against USD primarily
}

# Global Risk Indicators (affect all pairs)
GLOBAL_RISK_INDICATORS = {
    "vix": "VIXCLS",                    # VIX Volatility Index
    "oil_price": "DCOILWTICO",          # WTI Oil Price
    "dxy_index": "DTWEXBGS",            # Trade Weighted Dollar Index
    # 10Y-2Y Yield Curve (recession indicator)
    "yield_curve": "T10Y2Y",
    "treasury_10y": "DGS10",            # 10-Year Treasury Rate
    "treasury_2y": "DGS2",              # 2-Year Treasury Rate
}


class FREDMacroLoader:
    """
    Fetches real macroeconomic data from FRED for forex model training.

    Provides currency-pair specific features:
    - Interest rate differentials (base vs quote currency central bank rates)
    - Economic health differentials (GDP, CPI, unemployment comparisons)
    - Global risk indicators

    Example:
        loader = FREDMacroLoader(api_key="your_key")
        df = loader.get_macro_features_for_pair("EUR_USD", start_date, end_date)
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        """
        Initialize FRED data loader.

        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env variable.
            cache_dir: Directory to cache downloaded data. Defaults to writable location.
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self.cache_dir = cache_dir or FRED_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self._last_request_time = None
        self._request_count = 0

        if self.api_key:
            logger.info(
                f"FRED Macro Loader initialized (API key: ...{self.api_key[-4:]})")
        else:
            logger.warning(
                "FRED API key not set. Set FRED_API_KEY environment variable.")

    def _rate_limit(self):
        """Enforce rate limiting (120 requests per minute for FRED)."""
        if self._last_request_time:
            elapsed = (datetime.now() -
                       self._last_request_time).total_seconds()
            if elapsed < 0.5:  # Max 2 requests per second
                time.sleep(0.5 - elapsed)
        self._last_request_time = datetime.now()
        self._request_count += 1

    def _get_cache_path(self, series_id: str) -> Path:
        """Get cache file path for a series."""
        return self.cache_dir / f"{series_id}.parquet"

    def _load_from_cache(self, series_id: str) -> Optional[pd.DataFrame]:
        """Load cached series data if available and recent."""
        cache_path = self._get_cache_path(series_id)
        if cache_path.exists():
            # Check if cache is less than 1 day old
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(days=1):
                try:
                    return pd.read_parquet(cache_path)
                except Exception as e:
                    logger.warning(
                        f"Failed to read cache for {series_id}: {e}")
        return None

    def _save_to_cache(self, series_id: str, df: pd.DataFrame):
        """Save series data to cache."""
        try:
            df.to_parquet(self._get_cache_path(series_id))
        except Exception as e:
            logger.warning(f"Failed to cache {series_id}: {e}")

    def fetch_series(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "m",  # monthly by default
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch a single FRED series.

        Args:
            series_id: FRED series ID (e.g., "FEDFUNDS", "CPIAUCSL")
            start_date: Start date for observations
            end_date: End date for observations
            frequency: Aggregation frequency ('d'=daily, 'w'=weekly, 'm'=monthly)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with 'date' and 'value' columns
        """
        if not self.api_key:
            logger.warning(
                f"No FRED API key - returning empty DataFrame for {series_id}")
            return pd.DataFrame(columns=["date", "value"])

        # Check cache first
        if use_cache:
            cached = self._load_from_cache(series_id)
            if cached is not None:
                # Filter to requested date range
                cached = cached[(cached["date"] >= start_date)
                                & (cached["date"] <= end_date)]
                if not cached.empty:
                    logger.debug(f"Using cached data for {series_id}")
                    return cached

        self._rate_limit()

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date.strftime("%Y-%m-%d"),
            "observation_end": end_date.strftime("%Y-%m-%d"),
            "frequency": frequency,
            "aggregation_method": "avg",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)

            if response.status_code != 200:
                logger.error(
                    f"FRED API error for {series_id}: {response.status_code}")
                return pd.DataFrame(columns=["date", "value"])

            data = response.json()

            if "observations" not in data or not data["observations"]:
                logger.warning(f"No observations found for {series_id}")
                return pd.DataFrame(columns=["date", "value"])

            # Parse observations
            df = pd.DataFrame(data["observations"])
            df = df[["date", "value"]]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])

            # Cache the full dataset
            if use_cache and not df.empty:
                self._save_to_cache(series_id, df)

            logger.debug(f"Fetched {len(df)} observations for {series_id}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching {series_id}: {e}")
            return pd.DataFrame(columns=["date", "value"])
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.DataFrame(columns=["date", "value"])

    def get_central_bank_rate(
        self,
        currency: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get central bank policy rate for a currency.

        Args:
            currency: Currency code (e.g., "USD", "EUR", "GBP")
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and rate columns
        """
        series_id = CENTRAL_BANK_RATES.get(currency)
        if not series_id:
            logger.warning(f"No central bank rate series for {currency}")
            return pd.DataFrame(columns=["date", f"{currency.lower()}_rate"])

        df = self.fetch_series(series_id, start_date, end_date)

        # If primary series failed, try fallback
        if df.empty and currency in CENTRAL_BANK_RATES_FALLBACK:
            fallback_id = CENTRAL_BANK_RATES_FALLBACK[currency]
            logger.info(f"Trying fallback series {fallback_id} for {currency}")
            df = self.fetch_series(fallback_id, start_date, end_date)

        if not df.empty:
            df = df.rename(columns={"value": f"{currency.lower()}_rate"})

        return df

    def get_economic_indicators(
        self,
        country: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get economic indicators for a country.

        Args:
            country: Country code (e.g., "US", "EU", "GB")
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and indicator columns
        """
        indicators = ECONOMIC_INDICATORS.get(country, {})
        if not indicators:
            logger.warning(f"No economic indicators configured for {country}")
            return pd.DataFrame(columns=["date"])

        result_df = None

        for name, series_id in indicators.items():
            df = self.fetch_series(series_id, start_date, end_date)
            if df.empty:
                continue

            df = df.rename(columns={"value": f"{country.lower()}_{name}"})

            if result_df is None:
                result_df = df
            else:
                result_df = pd.merge(result_df, df, on="date", how="outer")

        if result_df is None:
            return pd.DataFrame(columns=["date"])

        result_df = result_df.sort_values("date")
        return result_df

    def get_global_risk_indicators(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get global risk indicators.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and risk indicator columns
        """
        result_df = None

        for name, series_id in GLOBAL_RISK_INDICATORS.items():
            df = self.fetch_series(series_id, start_date,
                                   end_date, frequency="d")
            if df.empty:
                continue

            df = df.rename(columns={"value": name})

            if result_df is None:
                result_df = df
            else:
                result_df = pd.merge(result_df, df, on="date", how="outer")

        if result_df is None:
            return pd.DataFrame(columns=["date"])

        result_df = result_df.sort_values("date")
        return result_df

    def get_rate_differential(
        self,
        base_currency: str,
        quote_currency: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Calculate interest rate differential between two currencies.

        Rate Differential = Base Rate - Quote Rate

        Positive differential → Base currency yields more → Bullish for base
        Negative differential → Quote currency yields more → Bearish for base

        Args:
            base_currency: Base currency (e.g., "EUR")
            quote_currency: Quote currency (e.g., "USD")
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with rate differential features
        """
        # Fetch both rates
        base_df = self.get_central_bank_rate(
            base_currency, start_date, end_date)
        quote_df = self.get_central_bank_rate(
            quote_currency, start_date, end_date)

        if base_df.empty or quote_df.empty:
            logger.warning(
                f"Could not get rates for {base_currency}/{quote_currency}")
            return pd.DataFrame(columns=["date", "rate_differential"])

        # Merge on date
        merged = pd.merge(base_df, quote_df, on="date", how="outer")
        merged = merged.sort_values("date")

        # Forward fill missing values (rates don't change daily)
        merged = merged.ffill()

        base_col = f"{base_currency.lower()}_rate"
        quote_col = f"{quote_currency.lower()}_rate"

        # Calculate differential
        merged["rate_differential"] = merged[base_col] - merged[quote_col]

        # Calculate rate of change
        merged["rate_diff_change"] = merged["rate_differential"].diff()

        # Direction (widening/narrowing)
        merged["rate_diff_widening"] = (
            merged["rate_diff_change"] > 0).astype(int)

        return merged

    def get_macro_features_for_pair(
        self,
        currency_pair: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get all macro features for a specific currency pair.

        Includes:
        - Interest rate differential (base - quote)
        - Economic indicator differentials
        - Global risk indicators

        Args:
            currency_pair: Currency pair (e.g., "EUR_USD", "GBP_USD")
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with all macro features, indexed by date
        """
        # Parse currency pair
        normalized = currency_pair.upper().replace("/", "_").replace("-", "_")
        if "_" in normalized:
            base_ccy, quote_ccy = normalized.split("_")
        else:
            base_ccy, quote_ccy = normalized[:3], normalized[3:]

        logger.info(f"Fetching FRED macro features for {base_ccy}/{quote_ccy}")

        # 1. Rate Differential
        rate_diff_df = self.get_rate_differential(
            base_ccy, quote_ccy, start_date, end_date)

        # 2. Economic Indicators for both currencies
        base_country = CURRENCY_TO_COUNTRY.get(base_ccy, "")
        quote_country = CURRENCY_TO_COUNTRY.get(quote_ccy, "")

        base_econ_df = self.get_economic_indicators(
            base_country, start_date, end_date) if base_country else None
        quote_econ_df = self.get_economic_indicators(
            quote_country, start_date, end_date) if quote_country else None

        # 3. Global Risk Indicators
        risk_df = self.get_global_risk_indicators(start_date, end_date)

        # Merge all features
        result_df = rate_diff_df.copy() if not rate_diff_df.empty else pd.DataFrame(
            {"date": pd.date_range(start_date, end_date, freq="D")})

        if base_econ_df is not None and not base_econ_df.empty:
            result_df = pd.merge(result_df, base_econ_df,
                                 on="date", how="outer")

        if quote_econ_df is not None and not quote_econ_df.empty:
            result_df = pd.merge(result_df, quote_econ_df,
                                 on="date", how="outer")

        if not risk_df.empty:
            result_df = pd.merge(result_df, risk_df, on="date", how="outer")

        # Sort and forward-fill (economic data released monthly/quarterly)
        result_df = result_df.sort_values("date")
        result_df = result_df.ffill()

        # Calculate economic differentials
        result_df = self._calculate_differentials(
            result_df, base_country, quote_country)

        # Clean up column names
        result_df.columns = [col.replace(".", "_")
                             for col in result_df.columns]

        logger.success(
            f"✓ Created {len(result_df.columns)-1} FRED macro features for {currency_pair}")

        return result_df

    def _calculate_differentials(
        self,
        df: pd.DataFrame,
        base_country: str,
        quote_country: str
    ) -> pd.DataFrame:
        """Calculate differentials between base and quote country indicators."""
        base_prefix = f"{base_country.lower()}_"
        quote_prefix = f"{quote_country.lower()}_"

        # Find matching indicators
        for col in df.columns:
            if col.startswith(base_prefix):
                indicator = col.replace(base_prefix, "")
                quote_col = f"{quote_prefix}{indicator}"

                if quote_col in df.columns:
                    diff_col = f"{indicator}_diff"
                    df[diff_col] = df[col] - df[quote_col]

        return df

    def download_all_series_for_training(
        self,
        start_date: datetime,
        end_date: datetime,
        output_dir: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Download all FRED series needed for training and save to parquet files.
        Useful for Kaggle where API calls may be limited.

        Args:
            start_date: Start date
            end_date: End date
            output_dir: Output directory for parquet files

        Returns:
            Dictionary of series_id -> DataFrame
        """
        output_dir = output_dir or (DATA_DIR / "fred_data")
        output_dir.mkdir(parents=True, exist_ok=True)

        all_series = {}

        # Collect all unique series IDs
        series_ids = set()

        # Central bank rates
        series_ids.update(CENTRAL_BANK_RATES.values())
        series_ids.update(CENTRAL_BANK_RATES_FALLBACK.values())

        # Economic indicators
        for country_indicators in ECONOMIC_INDICATORS.values():
            series_ids.update(country_indicators.values())

        # Global risk
        series_ids.update(GLOBAL_RISK_INDICATORS.values())

        logger.info(f"Downloading {len(series_ids)} FRED series...")

        for i, series_id in enumerate(series_ids):
            logger.info(f"[{i+1}/{len(series_ids)}] Fetching {series_id}")
            df = self.fetch_series(series_id, start_date,
                                   end_date, frequency="d", use_cache=True)

            if not df.empty:
                all_series[series_id] = df
                # Save to parquet
                output_path = output_dir / f"{series_id}.parquet"
                df.to_parquet(output_path)

        logger.success(
            f"✓ Downloaded {len(all_series)} FRED series to {output_dir}")

        return all_series


def filter_macro_by_currency_pair(
    macro_df: pd.DataFrame,
    currency_pair: str
) -> pd.DataFrame:
    """
    Filter macro features to only those relevant to the currency pair.
    Analogous to filter_news_by_currency_pair for sentiment.

    Args:
        macro_df: DataFrame with all macro features
        currency_pair: Currency pair (e.g., "EUR_USD")

    Returns:
        Filtered DataFrame with pair-relevant features only
    """
    # Parse currency pair
    normalized = currency_pair.upper().replace("/", "_").replace("-", "_")
    if "_" in normalized:
        base_ccy, quote_ccy = normalized.split("_")
    else:
        base_ccy, quote_ccy = normalized[:3], normalized[3:]

    base_country = CURRENCY_TO_COUNTRY.get(base_ccy, "").lower()
    quote_country = CURRENCY_TO_COUNTRY.get(quote_ccy, "").lower()

    # Keep columns that are:
    # 1. Date column
    # 2. Rate differential columns
    # 3. Base country indicators
    # 4. Quote country indicators
    # 5. Global risk indicators
    # 6. Differential columns (_diff suffix)

    relevant_cols = ["date"]

    global_indicators = ["vix", "ted_spread", "gold_price",
                         "oil_price", "dxy_index", "yield_curve"]
    rate_cols = ["rate_differential", "rate_diff_change", "rate_diff_widening",
                 f"{base_ccy.lower()}_rate", f"{quote_ccy.lower()}_rate"]

    for col in macro_df.columns:
        col_lower = col.lower()

        # Keep date
        if col_lower == "date":
            continue

        # Keep rate differential columns
        if any(rc in col_lower for rc in rate_cols):
            relevant_cols.append(col)
            continue

        # Keep base/quote country indicators
        if base_country and col_lower.startswith(f"{base_country}_"):
            relevant_cols.append(col)
            continue

        if quote_country and col_lower.startswith(f"{quote_country}_"):
            relevant_cols.append(col)
            continue

        # Keep global risk indicators
        if any(gi in col_lower for gi in global_indicators):
            relevant_cols.append(col)
            continue

        # Keep differential columns
        if col_lower.endswith("_diff"):
            relevant_cols.append(col)
            continue

    # Filter to relevant columns that exist
    existing_cols = [c for c in relevant_cols if c in macro_df.columns]

    logger.info(
        f"📊 Filtered macro features for {currency_pair}: {len(existing_cols)}/{len(macro_df.columns)} columns")

    return macro_df[existing_cols]


# Test function
if __name__ == "__main__":
    # Set up logging
    from loguru import logger
    import sys
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    # Test FRED loader
    api_key = os.getenv("FRED_API_KEY", "8ef93cf694bee76342c15a8707ef3a28")
    loader = FREDMacroLoader(api_key=api_key)

    # Test date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print("\n" + "="*60)
    print("TESTING FRED MACRO LOADER")
    print("="*60)

    # Test fetching Fed Funds Rate
    print("\n1. Testing Fed Funds Rate fetch:")
    fed_df = loader.fetch_series("FEDFUNDS", start_date, end_date)
    print(f"   Got {len(fed_df)} observations")
    if not fed_df.empty:
        print(
            f"   Latest: {fed_df.iloc[-1]['date'].date()} = {fed_df.iloc[-1]['value']:.2f}%")

    # Test rate differential
    print("\n2. Testing EUR/USD rate differential:")
    rate_diff = loader.get_rate_differential(
        "EUR", "USD", start_date, end_date)
    if not rate_diff.empty:
        print(f"   Got {len(rate_diff)} observations")
        print(
            f"   Latest differential: {rate_diff.iloc[-1]['rate_differential']:.2f}%")

    # Test full pair features
    print("\n3. Testing full EUR_USD macro features:")
    macro_df = loader.get_macro_features_for_pair(
        "EUR_USD", start_date, end_date)
    print(f"   Got {len(macro_df.columns)} features, {len(macro_df)} rows")
    print(f"   Features: {list(macro_df.columns)[:10]}...")

    print("\n" + "="*60)
    print("FRED MACRO LOADER TEST COMPLETE")
    print("="*60)

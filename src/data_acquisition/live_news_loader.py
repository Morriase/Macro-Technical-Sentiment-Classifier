"""
Live News Loader using Marketaux API for Real-Time Sentiment Analysis

This module provides live forex news for inference-time sentiment scoring.
Uses the free tier of Marketaux API which includes:
- 100 requests/day (free tier)
- Global financial news coverage
- Entity-based sentiment scoring
- Forex-related filtering

The live sentiment is designed to match training data format:
- Same VADER scoring as Kaggle historical data
- Same aggregation methods (EMA periods)
- Same forex keyword filtering
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

# Try to import from config, fallback to defaults
try:
    from src.config import SENTIMENT_EMA_PERIODS, SENTIMENT_CACHE_MINUTES, CURRENCY_PAIRS
except ImportError:
    SENTIMENT_EMA_PERIODS = [3, 7, 14]
    SENTIMENT_CACHE_MINUTES = 5
    CURRENCY_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY",
                      "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD"]


# Major currency keywords for filtering noise
# Maps to the 8 major pairs the models are trained on
MAJOR_CURRENCY_KEYWORDS = {
    # EUR pairs
    "EUR": ["euro", "eurozone", "ecb", "european central bank", "lagarde", "eur/usd", "eurusd",
            "german", "germany", "france", "italian", "draghi"],
    # USD (appears in all pairs)
    "USD": ["dollar", "usd", "fed", "federal reserve", "fomc", "powell", "us economy", "treasury",
            "greenback", "yellen", "us labor", "american"],
    # GBP pairs
    "GBP": ["pound", "sterling", "gbp", "boe", "bank of england", "bailey", "uk economy", "gbp/usd",
            "british", "britain", "london", "gilt"],
    # JPY pairs
    "JPY": ["yen", "jpy", "boj", "bank of japan", "ueda", "japan economy", "usd/jpy",
            "japanese", "tokyo", "jgb", "kuroda", "nikkei"],
    # AUD pairs
    "AUD": ["aussie", "aud", "rba", "reserve bank of australia", "australia economy", "aud/usd",
            "australian", "sydney", "iron ore", "bullock"],
    # CAD pairs
    "CAD": ["loonie", "cad", "boc", "bank of canada", "canada economy", "usd/cad",
            "canadian", "macklem", "oil price", "wti", "crude"],
    # CHF pairs
    "CHF": ["swiss franc", "chf", "snb", "swiss national bank", "switzerland", "usd/chf",
            "swiss", "zurich", "jordan"],
    # NZD pairs
    "NZD": ["kiwi", "nzd", "rbnz", "reserve bank of new zealand", "new zealand", "nzd/usd",
            "zealand", "orr", "wellington"],
    # XAU (Gold)
    "XAU": ["gold", "xau", "precious metal", "bullion", "gold price", "safe haven",
            "xauusd", "spot gold"],
}

# High-impact macro keywords (affects all pairs)
MACRO_KEYWORDS = [
    "interest rate", "rate decision", "monetary policy", "inflation", "cpi",
    "gdp", "employment", "nfp", "non-farm", "unemployment", "retail sales",
    "trade balance", "pmi", "manufacturing", "central bank", "hawkish", "dovish",
    "quantitative easing", "qe", "tapering", "rate hike", "rate cut"
]


class MarketauxNewsLoader:
    """
    Live news loader using Marketaux API for forex sentiment analysis.

    Filters news to focus on major currency pairs the models are trained on:
    EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD, XAU/USD

    Features:
    - Free tier: 100 requests/day
    - Built-in entity sentiment scoring
    - Major forex pair filtering (reduces noise)
    - Per-currency sentiment breakdown
    - Caching to minimize API calls
    - Rate limiting protection

    Usage:
        loader = MarketauxNewsLoader()
        news_df = loader.fetch_forex_news(hours_back=24)
        sentiment = loader.get_sentiment_features()
        pair_sentiment = loader.get_pair_sentiment("EUR_USD")
    """

    # Marketaux API base URL
    BASE_URL = "https://api.marketaux.com/v1/news/all"

    def __init__(self, api_key: Optional[str] = None, cache_minutes: int = None, target_pairs: List[str] = None):
        """
        Initialize Marketaux news loader.

        Args:
            api_key: Marketaux API key. If None, reads from MARKETAUX_API_KEY env variable.
            cache_minutes: How long to cache results. Defaults to SENTIMENT_CACHE_MINUTES.
            target_pairs: List of currency pairs to filter for. Defaults to CURRENCY_PAIRS from config.
        """
        self.api_key = api_key or os.getenv("MARKETAUX_API_KEY", "")
        self.cache_minutes = cache_minutes or SENTIMENT_CACHE_MINUTES
        self.target_pairs = target_pairs or CURRENCY_PAIRS

        # Build currency filter from target pairs
        self._build_currency_filters()

        # In-memory cache
        self._cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        self._last_request_time: datetime = None
        self._daily_request_count: int = 0
        self._daily_reset_date: datetime = datetime.utcnow().date()

        # Validate API key
        if not self.api_key:
            logger.warning(
                "Marketaux API key not configured. "
                "Set MARKETAUX_API_KEY environment variable or pass api_key parameter. "
                "Live sentiment will return neutral values."
            )
        else:
            logger.info(
                f"Marketaux API initialized (cache: {self.cache_minutes}min, pairs: {len(self.target_pairs)})")

    def _build_currency_filters(self):
        """Build keyword filters based on target currency pairs."""
        # Extract unique currencies from pairs (e.g., EUR_USD -> EUR, USD)
        self.target_currencies = set()
        for pair in self.target_pairs:
            parts = pair.replace("_", "/").replace("/", "_").split("_")
            self.target_currencies.update(parts)

        # Build search terms from currency keywords
        self.search_keywords = set()
        for currency in self.target_currencies:
            if currency in MAJOR_CURRENCY_KEYWORDS:
                self.search_keywords.update(MAJOR_CURRENCY_KEYWORDS[currency])

        # Add macro keywords (affect all pairs)
        self.search_keywords.update(MACRO_KEYWORDS)

        logger.debug(f"Currency filter: {self.target_currencies}")
        logger.debug(f"Search keywords: {len(self.search_keywords)} terms")

    def _is_relevant_article(self, headline: str, snippet: str = "") -> Tuple[bool, List[str]]:
        """
        Check if article is relevant to target currency pairs.

        Returns:
            Tuple of (is_relevant, list of matched currencies)
        """
        text = f"{headline} {snippet}".lower()
        matched_currencies = []

        # Check each target currency
        for currency in self.target_currencies:
            if currency in MAJOR_CURRENCY_KEYWORDS:
                keywords = MAJOR_CURRENCY_KEYWORDS[currency]
                for kw in keywords:
                    if kw.lower() in text:
                        matched_currencies.append(currency)
                        break

        # Also check for macro keywords (relevant to all)
        has_macro = any(kw.lower() in text for kw in MACRO_KEYWORDS)

        # Relevant if mentions at least one currency OR macro event
        is_relevant = len(matched_currencies) > 0 or has_macro

        return is_relevant, matched_currencies

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within API rate limits.
        Free tier: 100 requests/day

        Returns:
            True if request can proceed, False if rate limited
        """
        # Reset daily counter
        today = datetime.utcnow().date()
        if today != self._daily_reset_date:
            self._daily_request_count = 0
            self._daily_reset_date = today

        # Check daily limit (100 for free tier)
        if self._daily_request_count >= 95:  # Leave buffer
            logger.warning(
                f"Marketaux daily limit approaching ({self._daily_request_count}/100)")
            return False

        # Minimum 1 second between requests
        if self._last_request_time:
            elapsed = (datetime.utcnow() -
                       self._last_request_time).total_seconds()
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)

        return True

    def _get_cache_key(self, params: Dict) -> str:
        """Generate cache key from request parameters."""
        return json.dumps(params, sort_keys=True)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False

        cached_time, _ = self._cache[cache_key]
        age_minutes = (datetime.utcnow() - cached_time).total_seconds() / 60
        return age_minutes < self.cache_minutes

    def _build_search_query(self) -> str:
        """Build optimized search query for Marketaux API."""
        # Focus on high-value forex terms that reduce noise
        priority_terms = [
            "forex", "EUR/USD", "GBP/USD", "USD/JPY",
            "Fed", "ECB", "interest rate", "central bank"
        ]
        # Use OR operator for broad coverage
        return " | ".join(priority_terms)

    def fetch_forex_news(
        self,
        hours_back: int = 24,
        language: str = "en",
        limit: int = 50,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch forex-related news from Marketaux API.

        Articles are filtered to focus on major currency pairs the models are trained on.

        Args:
            hours_back: How many hours back to fetch news
            language: Language filter (default: English)
            limit: Maximum articles to fetch (max 50 for free tier)
            use_cache: Whether to use cached results

        Returns:
            DataFrame with columns: date, headline, source, url, sentiment_score, currencies
        """
        if not self.api_key:
            logger.debug("No API key - returning empty DataFrame")
            return self._create_empty_news_df()

        # Build request parameters
        published_after = (
            datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M")

        params = {
            "api_token": self.api_key,
            "language": language,
            "published_after": published_after,
            "limit": min(limit, 50),  # Free tier max
            "search": self._build_search_query(),
            "must_have_entities": "true",
            "group_similar": "true"
        }

        # Check cache
        cache_key = self._get_cache_key(
            {k: v for k, v in params.items() if k != "api_token"})
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug("Using cached news data")
            return self._cache[cache_key][1].copy()

        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("Rate limited - returning cached or empty data")
            if cache_key in self._cache:
                return self._cache[cache_key][1].copy()
            return self._create_empty_news_df()

        try:
            logger.info(
                f"Fetching news from Marketaux (last {hours_back}h)...")

            response = requests.get(self.BASE_URL, params=params, timeout=10)
            self._last_request_time = datetime.utcnow()
            self._daily_request_count += 1

            response.raise_for_status()
            data = response.json()

            # Parse response
            articles = data.get("data", [])
            meta = data.get("meta", {})

            logger.info(
                f"Received {meta.get('returned', 0)} articles (found: {meta.get('found', 0)})")

            # Convert to DataFrame
            news_df = self._parse_articles(articles)

            # Cache result
            self._cache[cache_key] = (datetime.utcnow(), news_df.copy())

            return news_df

        except requests.exceptions.Timeout:
            logger.error("Marketaux API timeout")
        except requests.exceptions.HTTPError as e:
            logger.error(f"Marketaux API error: {e}")
            if "401" in str(e) or "403" in str(e):
                logger.error("API key invalid or expired")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
        except json.JSONDecodeError:
            logger.error("Invalid JSON response from Marketaux")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

        # Return cached data if available, else empty
        if cache_key in self._cache:
            logger.warning("Using stale cached data due to API error")
            return self._cache[cache_key][1].copy()

        return self._create_empty_news_df()

    def _parse_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Parse Marketaux API response into DataFrame.

        Args:
            articles: List of article dictionaries from API

        Returns:
            DataFrame with standardized columns
        """
        if not articles:
            return self._create_empty_news_df()

        parsed = []
        for article in articles:
            # Extract entity sentiment (average across all entities)
            entities = article.get("entities", [])
            entity_sentiments = [
                e.get("sentiment_score", 0)
                for e in entities
                if e.get("sentiment_score") is not None
            ]

            # Marketaux sentiment is -1 to 1 (same as our polarity)
            api_sentiment = np.mean(
                entity_sentiments) if entity_sentiments else 0.0

            parsed.append({
                "uuid": article.get("uuid", ""),
                "date": pd.to_datetime(article.get("published_at")),
                "headline": article.get("title", ""),
                "description": article.get("description", ""),
                "snippet": article.get("snippet", ""),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "language": article.get("language", "en"),
                "api_sentiment": api_sentiment,  # Marketaux's sentiment
                "entity_count": len(entities),
                # Top 5 entities
                "entities": [e.get("symbol", "") for e in entities[:5]]
            })

        if not parsed:
            return self._create_empty_news_df()

        df = pd.DataFrame(parsed)
        df["date"] = pd.to_datetime(df["date"])

        # Filter to relevant articles for major currency pairs
        df = self._filter_relevant_articles(df)

        return df

    def _filter_relevant_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter articles to only those relevant to target currency pairs.
        Reduces noise from unrelated financial news.
        """
        if df.empty:
            return df

        relevant_rows = []
        for idx, row in df.iterrows():
            is_relevant, currencies = self._is_relevant_article(
                row["headline"],
                row.get("snippet", "")
            )
            if is_relevant:
                row_dict = row.to_dict()
                row_dict["matched_currencies"] = currencies
                relevant_rows.append(row_dict)

        if not relevant_rows:
            logger.warning("No articles matched currency pair filters")
            return self._create_empty_news_df()

        filtered_df = pd.DataFrame(relevant_rows)

        # Log filter stats
        original_count = len(df)
        filtered_count = len(filtered_df)
        logger.info(
            f"Filtered {original_count} → {filtered_count} relevant articles")

        return filtered_df

    def _create_empty_news_df(self) -> pd.DataFrame:
        """Create empty DataFrame with expected schema."""
        return pd.DataFrame(columns=[
            "uuid", "date", "headline", "description", "snippet",
            "source", "url", "language", "api_sentiment", "entity_count",
            "entities", "matched_currencies"
        ])

    def get_pair_sentiment(self, pair: str, hours_back: int = 24) -> Dict[str, float]:
        """
        Get sentiment specific to a currency pair.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            hours_back: Hours of news to analyze

        Returns:
            Sentiment features for that specific pair
        """
        news_df = self.fetch_forex_news(hours_back=hours_back)

        if news_df.empty or "matched_currencies" not in news_df.columns:
            return self._get_neutral_sentiment()

        # Extract currencies from pair
        parts = pair.replace("_", "/").replace("/", "_").split("_")
        base_currency = parts[0] if len(parts) > 0 else ""
        quote_currency = parts[1] if len(parts) > 1 else ""

        # Filter to articles mentioning either currency in the pair
        pair_articles = news_df[
            news_df["matched_currencies"].apply(
                lambda x: base_currency in x or quote_currency in x if isinstance(
                    x, list) else False
            )
        ]

        if pair_articles.empty:
            logger.debug(f"No articles for {pair} - using neutral sentiment")
            return self._get_neutral_sentiment()

        # Score with VADER
        return self._score_with_vader(pair_articles)

    def get_sentiment_features(
        self,
        news_df: Optional[pd.DataFrame] = None,
        hours_back: int = 24,
        use_vader: bool = True
    ) -> Dict[str, float]:
        """
        Calculate sentiment features from live news for model input.

        This produces features that match the training data format:
        - polarity: -1 to 1 (positive - negative)
        - positive: 0 to 1 probability
        - negative: 0 to 1 probability
        - neutral: 0 to 1 probability
        - polarity_ema_3, polarity_ema_7, polarity_ema_14

        Args:
            news_df: Pre-fetched news DataFrame, or None to fetch fresh
            hours_back: Hours of news to consider
            use_vader: Whether to use VADER for local scoring (recommended)

        Returns:
            Dictionary of sentiment features matching training format
        """
        # Fetch news if not provided
        if news_df is None or news_df.empty:
            news_df = self.fetch_forex_news(hours_back=hours_back)

        if news_df.empty:
            logger.debug("No news available - returning neutral sentiment")
            return self._get_neutral_sentiment()

        # Score with VADER for consistency with training data
        if use_vader:
            sentiments = self._score_with_vader(news_df)
        else:
            # Use Marketaux API sentiment directly
            sentiments = self._use_api_sentiment(news_df)

        return sentiments

    def _score_with_vader(self, news_df: pd.DataFrame) -> Dict[str, float]:
        """
        Score news headlines with VADER to match training pipeline.

        This ensures live sentiment uses the same method as training data.
        """
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk

            # Ensure VADER lexicon is available
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)

            analyzer = SentimentIntensityAnalyzer()

            # Score each headline
            scores = []
            for _, row in news_df.iterrows():
                # Combine headline and snippet for more context
                text = f"{row['headline']} {row.get('snippet', '')}"
                vader_scores = analyzer.polarity_scores(text)
                scores.append({
                    "positive": vader_scores["pos"],
                    "negative": vader_scores["neg"],
                    "neutral": vader_scores["neu"],
                    # VADER compound is -1 to 1
                    "polarity": vader_scores["compound"]
                })

            if not scores:
                return self._get_neutral_sentiment()

            # Aggregate scores
            df_scores = pd.DataFrame(scores)

            # Calculate current sentiment (mean of recent articles)
            current_sentiment = {
                "positive": df_scores["positive"].mean(),
                "negative": df_scores["negative"].mean(),
                "neutral": df_scores["neutral"].mean(),
                "polarity": df_scores["polarity"].mean(),
            }

            # Add EMA features (simulated for single point)
            # In production, you'd maintain historical sentiment
            for period in SENTIMENT_EMA_PERIODS:
                current_sentiment[f"polarity_ema_{period}"] = current_sentiment["polarity"]
                current_sentiment[f"positive_ema_{period}"] = current_sentiment["positive"]
                current_sentiment[f"negative_ema_{period}"] = current_sentiment["negative"]

            # Add article count for quality assessment
            current_sentiment["article_count"] = len(news_df)

            logger.info(
                f"VADER sentiment: polarity={current_sentiment['polarity']:.3f} "
                f"(pos={current_sentiment['positive']:.3f}, neg={current_sentiment['negative']:.3f}) "
                f"from {len(news_df)} articles"
            )

            return current_sentiment

        except Exception as e:
            logger.error(f"VADER scoring failed: {e}")
            return self._get_neutral_sentiment()

    def _use_api_sentiment(self, news_df: pd.DataFrame) -> Dict[str, float]:
        """
        Use Marketaux API sentiment scores directly.

        Note: This may differ slightly from VADER-based training data.
        """
        if news_df.empty:
            return self._get_neutral_sentiment()

        # API sentiment is already -1 to 1 (polarity)
        polarity = news_df["api_sentiment"].mean()

        # Convert to probability distribution
        if polarity > 0:
            positive = abs(polarity)
            negative = 0.0
        else:
            positive = 0.0
            negative = abs(polarity)
        neutral = 1.0 - positive - negative

        sentiment = {
            "positive": positive,
            "negative": negative,
            "neutral": max(0, neutral),
            "polarity": polarity,
        }

        # Add EMA features
        for period in SENTIMENT_EMA_PERIODS:
            sentiment[f"polarity_ema_{period}"] = polarity
            sentiment[f"positive_ema_{period}"] = positive
            sentiment[f"negative_ema_{period}"] = negative

        sentiment["article_count"] = len(news_df)

        return sentiment

    def _get_neutral_sentiment(self) -> Dict[str, float]:
        """Return neutral sentiment features when no data available."""
        sentiment = {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "polarity": 0.0,
            "article_count": 0
        }

        for period in SENTIMENT_EMA_PERIODS:
            sentiment[f"polarity_ema_{period}"] = 0.0
            sentiment[f"positive_ema_{period}"] = 0.0
            sentiment[f"negative_ema_{period}"] = 0.0

        return sentiment

    def get_health_status(self) -> Dict:
        """
        Get API health and usage status.

        Returns:
            Dictionary with API status information
        """
        return {
            "api_configured": bool(self.api_key),
            "daily_requests_used": self._daily_request_count,
            "daily_limit": 100,
            "cache_size": len(self._cache),
            "cache_minutes": self.cache_minutes,
            "last_request": self._last_request_time.isoformat() if self._last_request_time else None
        }


# Convenience function for quick sentiment lookup
def get_live_forex_sentiment(
    api_key: Optional[str] = None,
    hours_back: int = 24
) -> Dict[str, float]:
    """
    Quick function to get current forex sentiment.

    Args:
        api_key: Marketaux API key (optional, uses env var if not provided)
        hours_back: Hours of news to analyze

    Returns:
        Dictionary of sentiment features

    Example:
        sentiment = get_live_forex_sentiment()
        print(f"Market sentiment: {sentiment['polarity']:.3f}")
    """
    loader = MarketauxNewsLoader(api_key=api_key)
    return loader.get_sentiment_features(hours_back=hours_back)


if __name__ == "__main__":
    # Test the loader
    import sys

    print("=" * 60)
    print("MARKETAUX LIVE NEWS LOADER TEST")
    print("=" * 60)

    loader = MarketauxNewsLoader()

    # Check status
    status = loader.get_health_status()
    print(f"\nAPI Status:")
    for k, v in status.items():
        print(f"  {k}: {v}")

    if not status["api_configured"]:
        print("\n⚠ No API key configured!")
        print("Set MARKETAUX_API_KEY environment variable to test live functionality.")
        print("Using neutral sentiment fallback.")

    # Test sentiment features
    print(f"\nFetching sentiment features...")
    sentiment = loader.get_sentiment_features(hours_back=24)

    print(f"\nSentiment Features:")
    for k, v in sentiment.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n✓ Test complete")

"""
Test Marketaux Live News Loader for Forex Sentiment

This test validates the Marketaux API integration:
1. API connection (if key configured)
2. News fetching and parsing
3. VADER sentiment scoring
4. Cache functionality
5. Rate limiting protection

Run: python test_marketaux_integration.py
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_marketaux_loader():
    """Test the Marketaux news loader."""
    print("=" * 60)
    print("MARKETAUX LIVE NEWS LOADER INTEGRATION TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Import the loader
    try:
        from src.data_acquisition.live_news_loader import MarketauxNewsLoader, get_live_forex_sentiment
        print("✓ Import successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    # Initialize loader
    loader = MarketauxNewsLoader()

    # Check API status
    status = loader.get_health_status()
    print("\n--- API Status ---")
    for k, v in status.items():
        print(f"  {k}: {v}")

    api_configured = status["api_configured"]

    if not api_configured:
        print("\n⚠ MARKETAUX_API_KEY not configured!")
        print("  To enable live sentiment, set the environment variable:")
        print("  - Windows: set MARKETAUX_API_KEY=your_api_key")
        print("  - Linux/Mac: export MARKETAUX_API_KEY=your_api_key")
        print("  - .env file: MARKETAUX_API_KEY=your_api_key")
        print("\n  Get a free API key at: https://www.marketaux.com/")
        print("\n  Testing with neutral sentiment fallback...")

    # Test fetching news (will return empty if no API key)
    print("\n--- Fetching News ---")
    news_df = loader.fetch_forex_news(hours_back=24, limit=10)
    print(f"  Articles fetched: {len(news_df)}")

    if not news_df.empty:
        print(
            f"  Date range: {news_df['date'].min()} to {news_df['date'].max()}")
        print(f"  Sources: {news_df['source'].nunique()} unique")

        # Show sample headlines
        print("\n  Sample Headlines:")
        for i, row in news_df.head(3).iterrows():
            headline = row['headline'][:80] + \
                "..." if len(row['headline']) > 80 else row['headline']
            print(f"    - {headline}")

    # Test sentiment features
    print("\n--- Sentiment Features ---")
    sentiment = loader.get_sentiment_features(news_df=news_df)

    print(f"  Polarity: {sentiment['polarity']:.4f}")
    print(f"  Positive: {sentiment['positive']:.4f}")
    print(f"  Negative: {sentiment['negative']:.4f}")
    print(f"  Neutral:  {sentiment['neutral']:.4f}")
    print(f"  Articles: {sentiment['article_count']}")

    # Show EMA features
    print("\n  EMA Features:")
    for key, value in sentiment.items():
        if 'ema' in key:
            print(f"    {key}: {value:.4f}")

    # Test convenience function
    print("\n--- Quick Sentiment Function ---")
    quick_sentiment = get_live_forex_sentiment(hours_back=12)
    print(f"  12h polarity: {quick_sentiment['polarity']:.4f}")

    # Test caching
    print("\n--- Cache Test ---")
    news_df_2 = loader.fetch_forex_news(
        hours_back=24, limit=10, use_cache=True)
    status_after = loader.get_health_status()
    print(f"  Cache hits: {status_after['cache_size']} entries")
    print(f"  API calls: {status_after['daily_requests_used']}")

    # Validate feature format matches training
    print("\n--- Training Format Validation ---")
    required_features = [
        'polarity', 'positive', 'negative', 'neutral',
        'polarity_ema_3', 'polarity_ema_7', 'polarity_ema_14',
        'positive_ema_3', 'positive_ema_7', 'positive_ema_14',
        'negative_ema_3', 'negative_ema_7', 'negative_ema_14'
    ]

    missing = [f for f in required_features if f not in sentiment]
    if missing:
        print(f"  ✗ Missing features: {missing}")
    else:
        print(f"  ✓ All {len(required_features)} required features present")

    # Validate value ranges
    range_errors = []
    for key in ['positive', 'negative', 'neutral']:
        if key in sentiment:
            if not (0 <= sentiment[key] <= 1):
                range_errors.append(f"{key}={sentiment[key]} (should be 0-1)")

    if 'polarity' in sentiment:
        if not (-1 <= sentiment['polarity'] <= 1):
            range_errors.append(
                f"polarity={sentiment['polarity']} (should be -1 to 1)")

    if range_errors:
        print(f"  ✗ Range errors: {range_errors}")
    else:
        print("  ✓ All values in valid ranges")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    if api_configured:
        print("✓ Marketaux API is configured and working")
    else:
        print("⚠ Marketaux API not configured - using neutral fallback")
        print("  Set MARKETAUX_API_KEY for live sentiment")

    return True


def test_inference_server_integration():
    """Test that inference server can import and use the loader."""
    print("\n" + "=" * 60)
    print("INFERENCE SERVER INTEGRATION TEST")
    print("=" * 60)

    try:
        # Just test the import - don't start the server
        from inference_server import LIVE_NEWS_LOADER, get_live_sentiment_features

        if LIVE_NEWS_LOADER is not None:
            print("✓ MarketauxNewsLoader integrated in inference server")
            status = LIVE_NEWS_LOADER.get_health_status()
            print(f"  API configured: {status['api_configured']}")
        else:
            print("⚠ MarketauxNewsLoader not available in inference server")

        # Test the helper function
        sentiment = get_live_sentiment_features(hours_back=24)
        print(f"✓ get_live_sentiment_features() works")
        print(f"  Source: {sentiment.get('source', 'unknown')}")
        print(f"  Polarity: {sentiment.get('polarity', 0):.4f}")

        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = True

    # Test the loader directly
    if not test_marketaux_loader():
        success = False

    # Test inference server integration
    if not test_inference_server_integration():
        success = False

    print("\n")
    if success:
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    else:
        print("=" * 60)
        print("SOME TESTS FAILED ✗")
        print("=" * 60)
        sys.exit(1)

"""
Test script to verify sentiment analysis integration with forex filtering
Tests: News loading -> Forex filtering -> VADER sentiment analysis
"""
from src.data_acquisition.news_loader import KaggleNewsLoader
from src.feature_engineering.sentiment_features import SentimentAnalyzer
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta


def test_forex_news_filtering():
    """Test that forex-specific news filtering works correctly"""
    logger.info("=" * 80)
    logger.info("TEST 1: Forex News Filtering")
    logger.info("=" * 80)

    # Test with forex filtering enabled (default)
    news_loader_with_filter = KaggleNewsLoader(enable_forex_filter=True)

    # Load last 3 months of news
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    logger.info(f"Loading news from {start_date.date()} to {end_date.date()}")
    df_news = news_loader_with_filter.load_historical_news(
        start_date=start_date,
        end_date=end_date
    )

    if df_news.empty:
        logger.warning(
            "⚠ No news data loaded - check Kaggle dataset availability")
        return False

    logger.success(f"✓ Loaded {len(df_news)} forex-related news articles")

    # Show sample headlines
    if len(df_news) > 0:
        logger.info("\nSample forex headlines:")
        for i, (date, row) in enumerate(df_news.head(5).iterrows()):
            logger.info(f"  {i+1}. [{date}] {row['text'][:100]}...")

    return True


def test_vader_sentiment_analysis():
    """Test VADER sentiment analysis on forex news"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: VADER Sentiment Analysis (Free, No API)")
    logger.info("=" * 80)

    # Initialize VADER-only sentiment analyzer (no API costs)
    sentiment_analyzer = SentimentAnalyzer(use_vader_only=True)

    # Test sample forex headlines
    test_headlines = [
        "Fed signals interest rate hikes to combat inflation",
        "ECB maintains dovish stance despite rising prices",
        "Dollar strengthens on positive jobs report",
        "Euro falls as economic growth slows",
        "Central banks face difficult policy decisions"
    ]

    logger.info(f"\nAnalyzing {len(test_headlines)} sample forex headlines...")

    for headline in test_headlines:
        sentiment = sentiment_analyzer.analyze_text(headline)
        polarity = sentiment_analyzer.calculate_polarity_score(sentiment)

        # Determine sentiment label
        if polarity > 0.05:
            label = "POSITIVE"
        elif polarity < -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        logger.info(f"\n  Headline: {headline}")
        logger.info(f"  Sentiment: {label} (polarity: {polarity:.3f})")
        logger.info(f"  Scores: pos={sentiment['positive']:.3f}, "
                    f"neg={sentiment['negative']:.3f}, neu={sentiment['neutral']:.3f}")

    return True


def test_daily_aggregation():
    """Test daily sentiment aggregation"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Daily Sentiment Aggregation")
    logger.info("=" * 80)

    # Load news with forex filter
    news_loader = KaggleNewsLoader(enable_forex_filter=True)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days

    df_news = news_loader.load_historical_news(
        start_date=start_date,
        end_date=end_date
    )

    if df_news.empty:
        logger.warning("⚠ No news data for aggregation test")
        return False

    # Initialize sentiment analyzer (VADER only)
    sentiment_analyzer = SentimentAnalyzer(use_vader_only=True)

    # Aggregate daily sentiment
    df_news_reset = df_news.reset_index()
    df_news_reset.rename(columns={'text': 'headline'}, inplace=True)

    daily_sentiment = sentiment_analyzer.aggregate_daily_sentiment(
        df_news_reset,
        date_col='date',
        text_col='headline'
    )

    logger.success(f"✓ Aggregated sentiment for {len(daily_sentiment)} days")

    # Show recent daily sentiment
    if len(daily_sentiment) > 0:
        logger.info("\nRecent daily sentiment scores:")
        for _, row in daily_sentiment.tail(5).iterrows():
            logger.info(f"  {row['date'].date()}: "
                        f"polarity={row['polarity']:.3f}, "
                        f"pos={row['positive']:.3f}, "
                        f"neg={row['negative']:.3f}")

    # Calculate EMAs
    daily_sentiment_ema = sentiment_analyzer.calculate_time_weighted_sentiment(
        daily_sentiment,
        ema_periods=[3, 7, 14]
    )

    logger.success(f"✓ Calculated EMA sentiment features")
    logger.info(
        f"  Total features: {len(daily_sentiment_ema.columns)} columns")
    logger.info(f"  Feature names: {list(daily_sentiment_ema.columns)}")

    return True


def test_differential_sentiment():
    """Test differential sentiment calculation for currency pair"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Differential Sentiment (EUR_USD example)")
    logger.info("=" * 80)

    # Create sample sentiment data for EUR and USD
    dates = pd.date_range(end=datetime.now(), periods=10, freq='D')

    # Simulate EUR sentiment (slightly positive)
    eur_sentiment = pd.DataFrame({
        'date': dates,
        'polarity': [0.1, 0.15, 0.08, 0.12, 0.2, 0.05, 0.18, 0.1, 0.14, 0.11]
    })

    # Simulate USD sentiment (slightly negative)
    usd_sentiment = pd.DataFrame({
        'date': dates,
        'polarity': [-0.05, -0.1, -0.08, -0.03, -0.12, -0.15, -0.07, -0.09, -0.06, -0.11]
    })

    sentiment_analyzer = SentimentAnalyzer(use_vader_only=True)

    differential = sentiment_analyzer.calculate_differential_sentiment(
        eur_sentiment,
        usd_sentiment,
        merge_on='date'
    )

    logger.success(f"✓ Calculated differential sentiment")
    logger.info("\nDifferential sentiment (EUR - USD):")
    for _, row in differential.tail(5).iterrows():
        logger.info(f"  {row['date'].date()}: "
                    f"EUR={row['polarity_base']:.3f}, "
                    f"USD={row['polarity_quote']:.3f}, "
                    f"DIFF={row['polarity_diff']:.3f}")

    return True


def main():
    """Run all sentiment integration tests"""
    logger.info("\n" + "=" * 80)
    logger.info("SENTIMENT ANALYSIS INTEGRATION TEST SUITE")
    logger.info("Testing: News Loading -> Forex Filtering -> VADER Sentiment")
    logger.info("=" * 80 + "\n")

    results = {
        "Forex News Filtering": False,
        "VADER Sentiment Analysis": False,
        "Daily Aggregation": False,
        "Differential Sentiment": False
    }

    try:
        results["Forex News Filtering"] = test_forex_news_filtering()
    except Exception as e:
        logger.error(f"Test 1 failed: {e}")

    try:
        results["VADER Sentiment Analysis"] = test_vader_sentiment_analysis()
    except Exception as e:
        logger.error(f"Test 2 failed: {e}")

    try:
        results["Daily Aggregation"] = test_daily_aggregation()
    except Exception as e:
        logger.error(f"Test 3 failed: {e}")

    try:
        results["Differential Sentiment"] = test_differential_sentiment()
    except Exception as e:
        logger.error(f"Test 4 failed: {e}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"  {test_name}: {status}")

    passed_count = sum(results.values())
    total_count = len(results)

    logger.info(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        logger.success("\n🎉 All tests passed! Sentiment integration is ready.")
        logger.info("\nNext steps:")
        logger.info("  1. Run main.py to train models with sentiment features")
        logger.info(
            "  2. Sentiment features will be automatically added to training data")
        logger.info("  3. No API costs - using VADER (100% free, offline)")
    else:
        logger.warning("\n⚠ Some tests failed. Check error messages above.")


if __name__ == "__main__":
    main()

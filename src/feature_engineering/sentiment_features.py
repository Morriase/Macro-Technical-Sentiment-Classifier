"""
NLP Sentiment Analysis Pipeline for Financial News
Hybrid approach: VADER (fast, free, offline) + Optional FinBERT (accurate, GPU-accelerated)

PAIR-SPECIFIC SENTIMENT:
Training now filters news to currency-relevant headlines only, matching inference behavior.
This eliminates the whole-noise vs pair-specific mismatch that caused low prediction accuracy.
"""
from src.config import SENTIMENT_MODEL, SENTIMENT_EMA_PERIODS, LDA_NUM_TOPICS
from loguru import logger
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
import os
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# =============================================================================
# CURRENCY PAIR KEYWORD FILTERING
# Maps to the 8 major pairs the models are trained on
# Must match live_news_loader.py for train-inference consistency
# =============================================================================
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

# High-impact macro keywords (affects all pairs - used as tiebreaker)
MACRO_KEYWORDS = [
    "interest rate", "rate decision", "monetary policy", "inflation", "cpi",
    "gdp", "employment", "nfp", "non-farm", "unemployment", "retail sales",
    "trade balance", "pmi", "manufacturing", "central bank", "hawkish", "dovish",
    "quantitative easing", "qe", "tapering", "rate hike", "rate cut"
]


def get_currencies_from_pair(currency_pair: str) -> Tuple[str, str]:
    """
    Extract base and quote currencies from a pair string.

    Args:
        currency_pair: Pair like "EUR_USD", "EURUSD", "EUR/USD"

    Returns:
        Tuple of (base_currency, quote_currency)
    """
    # Normalize separators
    normalized = currency_pair.upper().replace("/", "_").replace("-", "_")

    if "_" in normalized:
        parts = normalized.split("_")
        return parts[0], parts[1]
    elif len(normalized) == 6:
        return normalized[:3], normalized[3:]
    else:
        logger.warning(f"Cannot parse currency pair: {currency_pair}")
        return normalized, ""


def is_headline_relevant(headline: str, currency_pair: str) -> bool:
    """
    Check if a headline is relevant to the given currency pair.
    Uses MAJOR_CURRENCY_KEYWORDS for matching.

    Args:
        headline: News headline text
        currency_pair: Currency pair like "EUR_USD"

    Returns:
        True if headline mentions base or quote currency keywords
    """
    if not headline or not currency_pair:
        return False

    base_ccy, quote_ccy = get_currencies_from_pair(currency_pair)
    text_lower = headline.lower()

    # Get keywords for both currencies
    base_keywords = MAJOR_CURRENCY_KEYWORDS.get(base_ccy, [])
    quote_keywords = MAJOR_CURRENCY_KEYWORDS.get(quote_ccy, [])

    # Check for any base currency keyword
    base_match = any(kw in text_lower for kw in base_keywords)

    # Check for any quote currency keyword
    quote_match = any(kw in text_lower for kw in quote_keywords)

    # Also check for macro keywords (applies to all pairs)
    macro_match = any(kw in text_lower for kw in MACRO_KEYWORDS)

    # Relevant if: mentions EITHER currency OR is a macro event
    return base_match or quote_match or macro_match


def filter_news_by_currency_pair(
    news_df: pd.DataFrame,
    currency_pair: str,
    text_col: str = "text"
) -> pd.DataFrame:
    """
    Filter news DataFrame to only headlines relevant to the given currency pair.
    This ensures training sentiment matches inference sentiment.

    Args:
        news_df: DataFrame with news articles (requires 'text' or specified text_col)
        currency_pair: Currency pair like "EUR_USD", "GBP_USD", etc.
        text_col: Column name containing headline/text

    Returns:
        Filtered DataFrame with only relevant headlines
    """
    if news_df.empty or currency_pair is None:
        return news_df

    initial_count = len(news_df)

    # Apply relevance filter
    relevant_mask = news_df[text_col].apply(
        lambda x: is_headline_relevant(str(x), currency_pair)
    )

    filtered_df = news_df[relevant_mask].copy()

    base_ccy, quote_ccy = get_currencies_from_pair(currency_pair)
    logger.info(
        f"📰 Filtered news for {currency_pair}: {len(filtered_df)}/{initial_count} headlines "
        f"({len(filtered_df)/initial_count*100:.1f}% relevant to {base_ccy}/{quote_ccy})"
    )

    return filtered_df


# Optional imports - only needed for training, not inference
try:
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logger.debug("Gensim not available - LDA topic modeling disabled")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.debug(
        "Transformers not available - FinBERT disabled, using VADER only")

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logger.info("Downloading VADER lexicon for sentiment analysis...")
    nltk.download('vader_lexicon', quiet=True)
    logger.success("✓ VADER lexicon downloaded")


class SentimentAnalyzer:
    """
    Hybrid sentiment analysis using VADER (primary, fast, free) + FinBERT (optional)
    Implements differential sentiment scoring for currency pairs

    VADER: Rule-based sentiment analyzer optimized for social media/news
    - 100% free, no API costs
    - Fast (~1000 texts/second)
    - Works offline
    - Excellent for financial sentiment

    FinBERT: Deep learning model (optional enhancement)
    - More accurate for complex financial language
    - Requires GPU for practical use
    - Falls back to VADER if unavailable
    """

    def __init__(self, model_name: str = SENTIMENT_MODEL, device: str = None, use_vader_only: bool = False):
        """
        Initialize sentiment analyzer with VADER (primary) + optional FinBERT

        Args:
            model_name: HuggingFace model identifier for FinBERT (optional)
            device: 'cuda', 'cpu', or None (auto-detect)
            use_vader_only: If True, skip FinBERT and use only VADER (recommended for cost-free analysis)
        """
        self.model_name = model_name
        self.use_vader_only = use_vader_only or not TRANSFORMERS_AVAILABLE

        # Initialize VADER (always available, free)
        logger.info("Initializing VADER sentiment analyzer (free, offline)...")
        try:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            logger.success("✓ VADER sentiment analyzer loaded successfully")
        except Exception as e:
            logger.error(f"FATAL: VADER analyzer failed to initialize: {e}")
            raise RuntimeError(f"Failed to initialize VADER: {e}")

        # Initialize FinBERT (optional enhancement)
        self.sentiment_pipeline = None
        if not self.use_vader_only and TRANSFORMERS_AVAILABLE:
            import torch
            if device is None:
                self.device = 0 if torch.cuda.is_available() else -1
            else:
                self.device = device

            logger.info(
                f"Attempting to load FinBERT model (optional): {model_name}")
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    device=self.device,
                    max_length=512,
                    truncation=True,
                )
                logger.success(
                    "✓ FinBERT model loaded successfully (will use for complex analysis)")
            except Exception as e:
                logger.warning(f"FinBERT model not available: {e}")
                logger.warning(
                    "Falling back to VADER-only mode (this is fine for production)")
                self.use_vader_only = True
        else:
            logger.info("Using VADER-only mode (fast, free, no API costs)")

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of single text using VADER (primary) or FinBERT (optional)

        Args:
            text: Input text

        Returns:
            Dictionary with sentiment scores {positive, negative, neutral}
        """
        if not text or len(text.strip()) == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        try:
            # Use VADER (fast, free) if in VADER-only mode or FinBERT unavailable
            if self.use_vader_only or self.sentiment_pipeline is None:
                # VADER returns compound score (-1 to 1) and individual scores
                scores = self.vader_analyzer.polarity_scores(text)

                # VADER provides positive, negative, neutral, compound
                # Normalize to match our expected format
                return {
                    "positive": scores["pos"],
                    "negative": scores["neg"],
                    "neutral": scores["neu"],
                }

            # Use FinBERT (more accurate for complex financial text)
            else:
                result = self.sentiment_pipeline(
                    text[:512])[0]  # Limit text length
                label = result["label"].lower()
                score = result["score"]

                # Convert to probability distribution
                sentiment_scores = {
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 0.0,
                }

                sentiment_scores[label] = score

                # Distribute remaining probability
                remaining = 1.0 - score
                for key in sentiment_scores:
                    if key != label:
                        sentiment_scores[key] = remaining / 2

                return sentiment_scores

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            # Fall back to neutral sentiment
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """
        Analyze sentiment for batch of texts

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of sentiment score dictionaries
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            for text in batch:
                scores = self.analyze_text(text)
                results.append(scores)

        return results

    def calculate_polarity_score(self, sentiment_scores: Dict[str, float]) -> float:
        """
        Calculate polarity score from sentiment probabilities

        Polarity = P(positive) - P(negative)
        Range: [-1, 1]

        Args:
            sentiment_scores: Dictionary with sentiment probabilities

        Returns:
            Polarity score
        """
        return sentiment_scores.get("positive", 0.0) - sentiment_scores.get("negative", 0.0)

    def aggregate_daily_sentiment(
        self, news_df: pd.DataFrame, date_col: str = "date", text_col: str = "headline",
        currency_pair: str = None
    ) -> pd.DataFrame:
        """
        Aggregate news sentiment by day with optional currency pair filtering.

        PAIR-SPECIFIC FILTERING:
        When currency_pair is provided, only headlines mentioning the relevant 
        currencies are included. This ensures training sentiment matches inference.

        Without filtering (currency_pair=None):
          - Uses ALL 121K headlines → diluted, noisy signal
          - Model learns "average market mood" not pair-specific factors

        With filtering (currency_pair="EUR_USD"):
          - Uses ~15K EUR/USD relevant headlines → focused signal
          - Model learns pair-specific central bank, economy mentions
          - MATCHES live inference which also filters by pair

        Args:
            news_df: DataFrame with news articles
            date_col: Column name for date
            text_col: Column name for text content
            currency_pair: If provided, filter headlines to this pair (e.g., "EUR_USD")

        Returns:
            DataFrame with daily aggregated sentiment
        """
        # Make a copy to avoid modifying original
        df_working = news_df.copy()

        # CRITICAL: Apply pair-specific filtering if currency_pair provided
        if currency_pair is not None:
            df_working = filter_news_by_currency_pair(
                df_working, currency_pair, text_col
            )
            if df_working.empty:
                logger.warning(
                    f"⚠ No news found for {currency_pair} - returning empty sentiment"
                )
                return pd.DataFrame(columns=["date", "positive", "negative", "neutral", "polarity"])
        else:
            logger.warning(
                "⚠ No currency_pair provided - using ALL headlines (not recommended for training)"
            )

        logger.info(
            f"Aggregating daily sentiment for {len(df_working)} articles")

        # Ensure date column is datetime
        df_working[date_col] = pd.to_datetime(df_working[date_col])
        df_working["date_only"] = df_working[date_col].dt.date

        # Analyze sentiment for each article
        logger.info(f"Analyzing sentiment for {len(df_working)} articles")
        sentiments = self.analyze_batch(df_working[text_col].tolist())

        # Add sentiment scores to dataframe
        df_working["positive"] = [s["positive"] for s in sentiments]
        df_working["negative"] = [s["negative"] for s in sentiments]
        df_working["neutral"] = [s["neutral"] for s in sentiments]
        df_working["polarity"] = df_working.apply(
            lambda row: self.calculate_polarity_score({
                "positive": row["positive"],
                "negative": row["negative"]
            }), axis=1
        )

        # Aggregate by day
        daily_sentiment = df_working.groupby("date_only").agg({
            "positive": "mean",
            "negative": "mean",
            "neutral": "mean",
            "polarity": "mean",
        }).reset_index()

        daily_sentiment.rename(columns={"date_only": "date"}, inplace=True)
        daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])

        return daily_sentiment

    def calculate_time_weighted_sentiment(
        self, daily_sentiment: pd.DataFrame, ema_periods: List[int] = SENTIMENT_EMA_PERIODS
    ) -> pd.DataFrame:
        """
        Calculate time-weighted sentiment using EMAs
        Models decaying influence of past news

        Args:
            daily_sentiment: DataFrame with daily sentiment scores
            ema_periods: List of EMA periods

        Returns:
            DataFrame with EMA sentiment columns
        """
        logger.info("Calculating time-weighted sentiment")

        for period in ema_periods:
            daily_sentiment[f"polarity_ema_{period}"] = (
                daily_sentiment["polarity"].ewm(
                    span=period, adjust=False).mean()
            )

            daily_sentiment[f"positive_ema_{period}"] = (
                daily_sentiment["positive"].ewm(
                    span=period, adjust=False).mean()
            )

            daily_sentiment[f"negative_ema_{period}"] = (
                daily_sentiment["negative"].ewm(
                    span=period, adjust=False).mean()
            )

        return daily_sentiment

    def calculate_differential_sentiment(
        self,
        base_currency_sentiment: pd.DataFrame,
        quote_currency_sentiment: pd.DataFrame,
        merge_on: str = "date",
    ) -> pd.DataFrame:
        """
        Calculate differential sentiment for currency pair

        Differential Sentiment = Sentiment(Base) - Sentiment(Quote)
        Critical for FX prediction as rates are relative

        Args:
            base_currency_sentiment: Sentiment for base currency (e.g., EUR)
            quote_currency_sentiment: Sentiment for quote currency (e.g., USD)
            merge_on: Column to merge on

        Returns:
            DataFrame with differential sentiment features
        """
        logger.info("Calculating differential sentiment")

        # Merge sentiments
        merged = base_currency_sentiment.merge(
            quote_currency_sentiment,
            on=merge_on,
            suffixes=("_base", "_quote"),
            how="outer",
        )

        # Fill missing values
        merged.fillna(0, inplace=True)

        # Calculate differentials
        merged["polarity_diff"] = merged["polarity_base"] - \
            merged["polarity_quote"]
        merged["positive_diff"] = merged["positive_base"] - \
            merged["positive_quote"]
        merged["negative_diff"] = merged["negative_base"] - \
            merged["negative_quote"]

        # Calculate differentials for EMA features
        for period in SENTIMENT_EMA_PERIODS:
            base_col = f"polarity_ema_{period}_base"
            quote_col = f"polarity_ema_{period}_quote"

            if base_col in merged.columns and quote_col in merged.columns:
                merged[f"polarity_ema_{period}_diff"] = (
                    merged[base_col] - merged[quote_col]
                )

        return merged


class ThematicAnalyzer:
    """
    Thematic analysis using Latent Dirichlet Allocation (LDA)
    Identifies dominant market themes (e.g., monetary policy, geopolitics)
    """

    def __init__(self, num_topics: int = LDA_NUM_TOPICS):
        """
        Initialize thematic analyzer

        Args:
            num_topics: Number of LDA topics
        """
        self.num_topics = num_topics
        self.dictionary = None
        self.lda_model = None

    def preprocess_texts(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess texts for LDA

        Args:
            texts: List of text documents

        Returns:
            List of tokenized documents
        """
        import re
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        # Download stopwords if not already
        try:
            stop_words = set(stopwords.words('english'))
        except:
            import nltk
            nltk.download('stopwords')
            nltk.download('punkt')
            stop_words = set(stopwords.words('english'))

        processed_texts = []

        for text in texts:
            # Lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

            # Tokenize
            tokens = word_tokenize(text)

            # Remove stopwords and short tokens
            tokens = [t for t in tokens if t not in stop_words and len(t) > 3]

            processed_texts.append(tokens)

        return processed_texts

    def train_lda(self, texts: List[str]) -> None:
        """
        Train LDA model on corpus

        Args:
            texts: List of text documents
        """
        logger.info(f"Training LDA model with {self.num_topics} topics")

        # Preprocess
        processed_texts = self.preprocess_texts(texts)

        # Create dictionary and corpus
        self.dictionary = Dictionary(processed_texts)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)

        corpus = [self.dictionary.doc2bow(text) for text in processed_texts]

        # Train LDA
        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
        )

        logger.info("LDA model trained")

    def get_document_topics(self, text: str, top_n: int = 3) -> List[Tuple[int, float]]:
        """
        Get topic distribution for document

        Args:
            text: Input text
            top_n: Number of top topics to return

        Returns:
            List of (topic_id, probability) tuples
        """
        if not self.lda_model or not self.dictionary:
            return []

        # Preprocess
        processed = self.preprocess_texts([text])[0]
        bow = self.dictionary.doc2bow(processed)

        # Get topic distribution
        topics = self.lda_model.get_document_topics(bow)

        # Sort by probability
        topics = sorted(topics, key=lambda x: x[1], reverse=True)

        return topics[:top_n]


if __name__ == "__main__":
    # Example usage
    logger.add("logs/sentiment_analysis.log", rotation="1 day")

    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()

    # Test single text
    sample_text = "The Federal Reserve announced a rate hike, signaling strong economic growth."
    result = sentiment_analyzer.analyze_text(sample_text)

    print("Sentiment analysis result:")
    print(result)
    print(
        f"Polarity: {sentiment_analyzer.calculate_polarity_score(result):.3f}")

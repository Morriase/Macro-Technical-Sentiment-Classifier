"""
NLP Sentiment Analysis Pipeline for Financial News
Uses FinBERT for currency-pair differential sentiment analysis
"""
from src.config import SENTIMENT_MODEL, SENTIMENT_EMA_PERIODS, LDA_NUM_TOPICS
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class SentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT
    Implements differential sentiment scoring for currency pairs
    """

    def __init__(self, model_name: str = SENTIMENT_MODEL, device: str = None):
        """
        Initialize sentiment analyzer with FinBERT

        Args:
            model_name: HuggingFace model identifier
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_name = model_name

        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        logger.info(f"Loading sentiment model: {model_name}")

        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
                max_length=512,
                truncation=True,
            )
            logger.success("✓ Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"FATAL: FinBERT sentiment model failed to load: {e}")
            logger.error(
                "Sentiment features are REQUIRED for 67-feature training")
            raise RuntimeError(f"Failed to load sentiment model: {e}")

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of single text

        Args:
            text: Input text

        Returns:
            Dictionary with sentiment scores {positive, negative, neutral}
        """
        if not self.sentiment_pipeline:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        if not text or len(text.strip()) == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        try:
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
        self, news_df: pd.DataFrame, date_col: str = "date", text_col: str = "headline"
    ) -> pd.DataFrame:
        """
        Aggregate news sentiment by day

        Args:
            news_df: DataFrame with news articles
            date_col: Column name for date
            text_col: Column name for text content

        Returns:
            DataFrame with daily aggregated sentiment
        """
        logger.info("Aggregating daily sentiment")

        # Ensure date column is datetime
        news_df[date_col] = pd.to_datetime(news_df[date_col])
        news_df["date_only"] = news_df[date_col].dt.date

        # Analyze sentiment for each article
        logger.info(f"Analyzing sentiment for {len(news_df)} articles")
        sentiments = self.analyze_batch(news_df[text_col].tolist())

        # Add sentiment scores to dataframe
        news_df["positive"] = [s["positive"] for s in sentiments]
        news_df["negative"] = [s["negative"] for s in sentiments]
        news_df["neutral"] = [s["neutral"] for s in sentiments]
        news_df["polarity"] = news_df.apply(
            lambda row: self.calculate_polarity_score({
                "positive": row["positive"],
                "negative": row["negative"]
            }), axis=1
        )

        # Aggregate by day
        daily_sentiment = news_df.groupby("date_only").agg({
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

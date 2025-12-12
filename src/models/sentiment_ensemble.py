"""
Sentiment Ensemble - Option C Implementation
Combines Technical Model (trained) + FinBERT Sentiment (pretrained) at inference time.

No retraining required - both models vote independently and results are combined.

Architecture:
    Technical Model (7 features) → BUY/SELL probabilities
    FinBERT (news headlines)     → Bullish/Bearish sentiment
    ────────────────────────────────────────────────────────
    Ensemble Logic               → Combined signal + confidence
"""
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger
from dataclasses import dataclass


@dataclass
class EnsembleResult:
    """Result from sentiment ensemble voting."""
    signal: str                    # "BUY", "SELL", or "HOLD"
    confidence: float              # Combined confidence (0-1)
    tech_signal: str               # Technical model signal
    tech_confidence: float         # Technical model confidence
    sentiment_signal: str          # Sentiment signal (bullish/bearish)
    sentiment_score: float         # Sentiment polarity (-1 to 1)
    agreement: bool                # Whether tech and sentiment agree
    quality_boost: float           # Quality score adjustment


class SentimentEnsemble:
    """
    Ensemble voting between Technical Model and FinBERT Sentiment.
    
    Combination strategies:
    1. Weighted Average: Blend probabilities with configurable weights
    2. Confirmation Gate: Only trade if both agree on direction
    3. Quality Boost: Increase quality score when signals align
    
    Default: Weighted average with tech=0.7, sentiment=0.3
    """
    
    def __init__(
        self,
        tech_weight: float = 0.7,
        sentiment_weight: float = 0.3,
        require_agreement: bool = False,
        sentiment_threshold: float = 0.1,  # Min sentiment to count as directional
        boost_on_agreement: float = 10.0,  # Quality boost when signals agree
        penalty_on_disagreement: float = 15.0  # Quality penalty when signals disagree
    ):
        """
        Initialize sentiment ensemble.
        
        Args:
            tech_weight: Weight for technical model (0-1)
            sentiment_weight: Weight for sentiment (0-1), should sum to 1 with tech_weight
            require_agreement: If True, only trade when both agree (conservative)
            sentiment_threshold: Minimum |polarity| to consider sentiment directional
            boost_on_agreement: Quality score boost when tech and sentiment agree
            penalty_on_disagreement: Quality penalty when they disagree
        """
        # Normalize weights
        total = tech_weight + sentiment_weight
        self.tech_weight = tech_weight / total
        self.sentiment_weight = sentiment_weight / total
        
        self.require_agreement = require_agreement
        self.sentiment_threshold = sentiment_threshold
        self.boost_on_agreement = boost_on_agreement
        self.penalty_on_disagreement = penalty_on_disagreement
        
        logger.info(
            f"SentimentEnsemble initialized: tech={self.tech_weight:.0%}, "
            f"sentiment={self.sentiment_weight:.0%}, require_agreement={require_agreement}"
        )
    
    def combine(
        self,
        tech_proba: np.ndarray,
        sentiment: Dict[str, float],
        tech_predicted_class: int
    ) -> EnsembleResult:
        """
        Combine technical model prediction with sentiment.
        
        Args:
            tech_proba: [P(Sell), P(Buy)] from technical model
            sentiment: Dict with 'polarity' (-1 to 1), 'positive', 'negative', 'neutral'
            tech_predicted_class: 0=Sell, 1=Buy from technical model
            
        Returns:
            EnsembleResult with combined signal and metadata
        """
        # Extract technical signal
        tech_buy_prob = float(tech_proba[1])
        tech_sell_prob = float(tech_proba[0])
        tech_signal = "BUY" if tech_predicted_class == 1 else "SELL"
        tech_confidence = float(tech_proba[tech_predicted_class])
        
        # Extract sentiment signal
        polarity = sentiment.get('polarity', 0.0)
        positive = sentiment.get('positive', 0.0)
        negative = sentiment.get('negative', 0.0)
        
        # Determine sentiment direction
        if polarity > self.sentiment_threshold:
            sentiment_signal = "BUY"  # Bullish sentiment
            sentiment_buy_prob = 0.5 + (polarity / 2)  # Map [-1,1] to [0,1]
            sentiment_sell_prob = 1 - sentiment_buy_prob
        elif polarity < -self.sentiment_threshold:
            sentiment_signal = "SELL"  # Bearish sentiment
            sentiment_sell_prob = 0.5 + (abs(polarity) / 2)
            sentiment_buy_prob = 1 - sentiment_sell_prob
        else:
            sentiment_signal = "NEUTRAL"
            sentiment_buy_prob = 0.5
            sentiment_sell_prob = 0.5
        
        # Check agreement
        agreement = (
            (tech_signal == "BUY" and sentiment_signal == "BUY") or
            (tech_signal == "SELL" and sentiment_signal == "SELL") or
            sentiment_signal == "NEUTRAL"  # Neutral doesn't disagree
        )
        
        # Calculate quality adjustment
        if sentiment_signal == "NEUTRAL":
            quality_boost = 0.0  # No boost or penalty for neutral sentiment
        elif agreement:
            quality_boost = self.boost_on_agreement
        else:
            quality_boost = -self.penalty_on_disagreement
        
        # Combine probabilities (weighted average)
        combined_buy_prob = (
            self.tech_weight * tech_buy_prob + 
            self.sentiment_weight * sentiment_buy_prob
        )
        combined_sell_prob = (
            self.tech_weight * tech_sell_prob + 
            self.sentiment_weight * sentiment_sell_prob
        )
        
        # Determine final signal
        if self.require_agreement and not agreement and sentiment_signal != "NEUTRAL":
            # Disagreement with require_agreement → HOLD
            final_signal = "HOLD"
            final_confidence = 0.0
        elif combined_buy_prob > combined_sell_prob:
            final_signal = "BUY"
            final_confidence = combined_buy_prob
        else:
            final_signal = "SELL"
            final_confidence = combined_sell_prob
        
        return EnsembleResult(
            signal=final_signal,
            confidence=final_confidence,
            tech_signal=tech_signal,
            tech_confidence=tech_confidence,
            sentiment_signal=sentiment_signal,
            sentiment_score=polarity,
            agreement=agreement,
            quality_boost=quality_boost
        )
    
    def get_combined_probabilities(
        self,
        tech_proba: np.ndarray,
        sentiment: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Get combined BUY/SELL probabilities.
        
        Returns:
            (sell_prob, buy_prob) tuple
        """
        polarity = sentiment.get('polarity', 0.0)
        
        # Map sentiment polarity to probabilities
        sentiment_buy_prob = 0.5 + (polarity / 2)  # [-1,1] → [0,1]
        sentiment_sell_prob = 1 - sentiment_buy_prob
        
        # Weighted combination
        combined_buy = self.tech_weight * tech_proba[1] + self.sentiment_weight * sentiment_buy_prob
        combined_sell = self.tech_weight * tech_proba[0] + self.sentiment_weight * sentiment_sell_prob
        
        return (combined_sell, combined_buy)


class CachedSentimentProvider:
    """
    Provides cached sentiment from Marketaux + FinBERT.
    Caches aggressively to respect API rate limits (100/day free tier).
    """
    
    def __init__(self, cache_minutes: int = 60):
        """
        Initialize sentiment provider with caching.
        
        Args:
            cache_minutes: How long to cache sentiment (default 60 min)
        """
        self.cache_minutes = cache_minutes
        self._news_loader = None
        self._cache: Dict[str, Tuple[float, Dict]] = {}  # pair -> (timestamp, sentiment)
        
    def _get_loader(self):
        """Lazy load the news loader."""
        if self._news_loader is None:
            try:
                from src.data_acquisition.live_news_loader import MarketauxNewsLoader
                self._news_loader = MarketauxNewsLoader(cache_minutes=self.cache_minutes)
                logger.info("MarketauxNewsLoader initialized for sentiment")
            except Exception as e:
                logger.warning(f"Failed to initialize MarketauxNewsLoader: {e}")
                self._news_loader = False  # Mark as failed
        return self._news_loader if self._news_loader else None
    
    def get_sentiment(self, pair: str, hours_back: int = 24) -> Dict[str, float]:
        """
        Get sentiment for a currency pair with caching.
        
        Args:
            pair: Currency pair (e.g., "EUR_USD")
            hours_back: Hours of news to analyze
            
        Returns:
            Sentiment dict with polarity, positive, negative, neutral
        """
        import time
        
        # Check cache
        cache_key = f"{pair}_{hours_back}"
        if cache_key in self._cache:
            cached_time, cached_sentiment = self._cache[cache_key]
            age_minutes = (time.time() - cached_time) / 60
            if age_minutes < self.cache_minutes:
                logger.debug(f"Using cached sentiment for {pair} (age: {age_minutes:.1f}min)")
                return cached_sentiment
        
        # Fetch fresh sentiment
        loader = self._get_loader()
        if loader is None:
            return self._neutral_sentiment()
        
        try:
            sentiment = loader.get_pair_sentiment(pair, hours_back=hours_back)
            
            # Cache result
            self._cache[cache_key] = (time.time(), sentiment)
            
            logger.info(
                f"Sentiment for {pair}: polarity={sentiment.get('polarity', 0):.3f} "
                f"(pos={sentiment.get('positive', 0):.2f}, neg={sentiment.get('negative', 0):.2f})"
            )
            
            return sentiment
            
        except Exception as e:
            logger.warning(f"Failed to get sentiment for {pair}: {e}")
            return self._neutral_sentiment()
    
    def _neutral_sentiment(self) -> Dict[str, float]:
        """Return neutral sentiment when data unavailable."""
        return {
            'polarity': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0,
            'article_count': 0
        }
    
    def get_health_status(self) -> Dict:
        """Get sentiment provider health status."""
        loader = self._get_loader()
        if loader is None:
            return {
                'available': False,
                'reason': 'MarketauxNewsLoader not initialized'
            }
        
        try:
            status = loader.get_health_status()
            status['available'] = True
            status['cache_entries'] = len(self._cache)
            return status
        except Exception as e:
            return {
                'available': False,
                'reason': str(e)
            }


# Global instances for reuse
_sentiment_ensemble: Optional[SentimentEnsemble] = None
_sentiment_provider: Optional[CachedSentimentProvider] = None


def get_sentiment_ensemble() -> SentimentEnsemble:
    """Get or create global sentiment ensemble instance."""
    global _sentiment_ensemble
    if _sentiment_ensemble is None:
        _sentiment_ensemble = SentimentEnsemble()
    return _sentiment_ensemble


def get_sentiment_provider() -> CachedSentimentProvider:
    """Get or create global sentiment provider instance."""
    global _sentiment_provider
    if _sentiment_provider is None:
        _sentiment_provider = CachedSentimentProvider(cache_minutes=60)
    return _sentiment_provider


if __name__ == "__main__":
    # Test the ensemble
    print("=" * 60)
    print("SENTIMENT ENSEMBLE TEST")
    print("=" * 60)
    
    ensemble = SentimentEnsemble()
    
    # Test case 1: Tech says BUY, sentiment bullish (agreement)
    tech_proba = np.array([0.35, 0.65])  # 65% BUY
    sentiment = {'polarity': 0.3, 'positive': 0.5, 'negative': 0.2, 'neutral': 0.3}
    result = ensemble.combine(tech_proba, sentiment, tech_predicted_class=1)
    
    print(f"\nTest 1: Tech BUY + Bullish Sentiment")
    print(f"  Tech: {result.tech_signal} ({result.tech_confidence:.1%})")
    print(f"  Sentiment: {result.sentiment_signal} (polarity={result.sentiment_score:.2f})")
    print(f"  Agreement: {result.agreement}")
    print(f"  Combined: {result.signal} ({result.confidence:.1%})")
    print(f"  Quality boost: {result.quality_boost:+.0f}")
    
    # Test case 2: Tech says SELL, sentiment bullish (disagreement)
    tech_proba = np.array([0.70, 0.30])  # 70% SELL
    sentiment = {'polarity': 0.4, 'positive': 0.6, 'negative': 0.1, 'neutral': 0.3}
    result = ensemble.combine(tech_proba, sentiment, tech_predicted_class=0)
    
    print(f"\nTest 2: Tech SELL + Bullish Sentiment (DISAGREEMENT)")
    print(f"  Tech: {result.tech_signal} ({result.tech_confidence:.1%})")
    print(f"  Sentiment: {result.sentiment_signal} (polarity={result.sentiment_score:.2f})")
    print(f"  Agreement: {result.agreement}")
    print(f"  Combined: {result.signal} ({result.confidence:.1%})")
    print(f"  Quality boost: {result.quality_boost:+.0f}")
    
    # Test case 3: Tech says BUY, sentiment neutral
    tech_proba = np.array([0.40, 0.60])  # 60% BUY
    sentiment = {'polarity': 0.05, 'positive': 0.3, 'negative': 0.25, 'neutral': 0.45}
    result = ensemble.combine(tech_proba, sentiment, tech_predicted_class=1)
    
    print(f"\nTest 3: Tech BUY + Neutral Sentiment")
    print(f"  Tech: {result.tech_signal} ({result.tech_confidence:.1%})")
    print(f"  Sentiment: {result.sentiment_signal} (polarity={result.sentiment_score:.2f})")
    print(f"  Agreement: {result.agreement}")
    print(f"  Combined: {result.signal} ({result.confidence:.1%})")
    print(f"  Quality boost: {result.quality_boost:+.0f}")
    
    print("\n✓ Ensemble test complete")

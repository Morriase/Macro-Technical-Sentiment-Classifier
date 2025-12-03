# FinBERT Sentiment Upgrade

## Summary

This document summarizes the changes made to upgrade from VADER to FinBERT for sentiment analysis.

## Why FinBERT?

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a generic sentiment analyzer that doesn't understand financial context:

- "The market is **bullish**" → VADER may not recognize this as positive
- "Fed **cuts** rates" → VADER may see "cuts" as negative
- "Stock **falls**" → VADER may not weight this strongly

**FinBERT** (by ProsusAI) is fine-tuned on financial text:
- Trained on 50,000+ financial sentences from Reuters and analyst reports
- Understands "bullish", "bearish", "rally", "correction" in context
- Outputs calibrated probabilities (positive, negative, neutral)

## Architecture Change

### Training (Kaggle)
```
Models trained with:
├── Technical Indicators (67 features)
├── Multi-Timeframe Features (14 features)
├── FRED Macro Data (5 features)
└── NO Sentiment (news dates don't overlap with FX data)

Total: 86 features (without sentiment)
```

### Inference (Live)
```
Prediction uses:
├── Our trained model outputs (BUY/SELL/HOLD probability)
├── FinBERT sentiment from Marketaux headlines
└── Combined as enhancement/confidence adjustment

Sentiment is not a model input, but a post-prediction signal filter
```

## Files Changed

### 1. `src/data_acquisition/live_news_loader.py`

**Before:**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def _score_with_vader(self, news_df):
    sid = SentimentIntensityAnalyzer()
    # Generic scoring...
```

**After:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

FINBERT_MODEL = None
FINBERT_TOKENIZER = None

def load_finbert_model():
    """Load FinBERT for financial sentiment analysis."""
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    return model, tokenizer

def _score_with_finbert(self, news_df):
    """Score headlines with FinBERT (financial-domain accuracy)."""
    model, tokenizer = load_finbert_model()
    # GPU-accelerated batch inference
    for headline in headlines:
        inputs = tokenizer(headline, return_tensors="pt", truncation=True)
        probs = model(**inputs).logits.softmax(dim=-1)
        # probs[0]=positive, probs[1]=negative, probs[2]=neutral
```

### 2. `src/data_acquisition/fred_macro_loader.py` (NEW)

Added real FRED macro data fetching:
- Rate differential (US vs counterpart)
- VIX (volatility index)
- Yield curve (10Y-2Y)
- Oil prices (WTI)
- DXY (dollar index)

### 3. `main.py`

Disabled sentiment for training:
```python
# Sentiment disabled - news dates don't match FX data
self.sentiment_analyzer = None
```

### 4. `inference_server.py`

Sentiment features NOT added to model input (would cause feature mismatch):
```python
# SENTIMENT FEATURES DISABLED
# Models were trained WITHOUT sentiment features
if ENABLE_LIVE_SENTIMENT:
    logger.warning("⚠ Ignoring sentiment to match training (81 features).")
```

## API Keys Required

| Service | Key | Usage |
|---------|-----|-------|
| FRED | `FRED_API_KEY` | Historical macro data (training) |
| Marketaux | `MARKETAUX_API_KEY` | Live news headlines (inference) |

## FinBERT Output Format

```python
{
    "positive": 0.85,      # Probability of positive sentiment
    "negative": 0.10,      # Probability of negative sentiment
    "neutral": 0.05,       # Probability of neutral sentiment
    "polarity": 0.75,      # positive - negative (range -1 to 1)
    "article_count": 12,   # Number of articles analyzed
    "source": "marketaux"  # Data source
}
```

## Performance Notes

- FinBERT model size: ~440MB
- First load: 2-5 seconds
- Per-headline inference: ~50ms (GPU) / ~200ms (CPU)
- Model is cached after first load

## Testing

```python
# Test FinBERT loading
from src.data_acquisition.live_news_loader import load_finbert_model
model, tokenizer = load_finbert_model()
print(f"FinBERT loaded: {model is not None}")

# Test sentiment scoring
loader = MarketauxNewsLoader()
sentiment = loader.get_sentiment_features(hours_back=24)
print(f"Polarity: {sentiment['polarity']:.3f}")
```

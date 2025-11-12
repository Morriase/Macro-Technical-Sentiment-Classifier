# How the Model Learns Wins and Losses

## The Learning Process

### 1. **Target Creation (Labels)**

The model learns by being shown **historical examples** with labels:

```python
# For each 4H bar, we look 24 hours ahead:
forward_return = price_in_24h - current_price

# Convert to pips
forward_return_pips = forward_return * 10000  # For EUR/USD

# Label based on ATR threshold (now adaptive!)
if forward_return_pips > (0.5 * ATR):
    label = "Buy"   # Price went UP significantly
elif forward_return_pips < -(0.5 * ATR):
    label = "Sell"  # Price went DOWN significantly
else:
    label = "Hold"  # Price stayed in range (no clear direction)
```

**Example:**
- Current price: 1.0500
- ATR: 20 pips → Threshold = 10 pips (0.5 × 20)
- Price 24h later: 1.0515 (+15 pips)
- **Label: Buy** ✅ (because 15 > 10)

### 2. **What the Model Sees**

For each labeled example, the model gets:

**Input Features (X):**
- Technical indicators: RSI, MACD, EMA, ATR, etc.
- Macro events: Recent CPI, NFP, interest rate decisions
- Sentiment: FinBERT scores from news
- Multi-timeframe: H1 and H4 context

**Output Label (y):**
- Buy (0), Sell (1), or Hold (2)

### 3. **Training = Pattern Recognition**

The model learns patterns like:

**Pattern 1: Bullish Breakout**
```
IF:
  - RSI > 60 (momentum)
  - Price > EMA_50 (uptrend)
  - Recent positive CPI surprise
  - Bullish sentiment score
THEN:
  → Predict "Buy"
  → Check 24h later: Did price go up by >10 pips?
  → If YES: Reward the model ✅
  → If NO: Penalize the model ❌
```

**Pattern 2: Bearish Reversal**
```
IF:
  - RSI < 40 (weak momentum)
  - Price < EMA_200 (downtrend)
  - Negative NFP surprise
  - Bearish sentiment
THEN:
  → Predict "Sell"
  → Check 24h later: Did price drop by >10 pips?
  → If YES: Reward ✅
  → If NO: Penalize ❌
```

### 4. **Loss Function (How Wins/Losses Are Measured)**

The model uses **Cross-Entropy Loss** with label smoothing:

```python
# For each prediction:
predicted_probs = [0.7, 0.2, 0.1]  # [Buy, Sell, Hold]
actual_label = "Buy"  # (index 0)

# Calculate loss (error):
loss = -log(0.7) = 0.36  # Lower is better

# If prediction was wrong:
predicted_probs = [0.1, 0.7, 0.2]  # Predicted Sell
actual_label = "Buy"
loss = -log(0.1) = 2.30  # High loss = big penalty!
```

**The model adjusts its weights to minimize this loss.**

### 5. **Class Weights (Handling Imbalance)**

Since Hold is 62% of data, we use **class weights**:

```python
weights = {
    "Buy": 3.0,   # 3x penalty for getting Buy wrong
    "Sell": 3.0,  # 3x penalty for getting Sell wrong
    "Hold": 1.0   # 1x penalty for getting Hold wrong
}
```

This forces the model to **care more about Buy/Sell** predictions (the profitable ones).

### 6. **Walk-Forward Validation (Realistic Testing)**

The model is tested on **future data it has never seen**:

```
Train: Oct 2024 - Apr 2025
Test:  Apr 2025 - Jun 2025  ← Model has NEVER seen this data

Predictions on test set:
- Predicted Buy, Actual Buy → WIN ✅
- Predicted Sell, Actual Hold → LOSS ❌
- Predicted Hold, Actual Hold → WIN ✅

Accuracy = Wins / Total = 67%
```

## Why ATR-Based Threshold is Better

### Fixed Threshold (Old):
```
Threshold = 5 pips (always)

Problem:
- During low volatility (ATR=10): 5 pips is 50% of ATR (too strict!)
- During high volatility (ATR=40): 5 pips is 12.5% of ATR (too loose!)

Result: 62% Hold (imbalanced)
```

### ATR-Based Threshold (New):
```
Threshold = 0.5 × ATR (adaptive)

Benefits:
- Low volatility (ATR=10): Threshold = 5 pips (appropriate)
- High volatility (ATR=40): Threshold = 20 pips (appropriate)

Result: More balanced classes (~40-45% Hold expected)
```

## Summary: The Learning Loop

1. **Historical Data**: Model sees 50,000 examples with labels
2. **Pattern Learning**: Finds correlations between features and outcomes
3. **Prediction**: Given new features, predicts Buy/Sell/Hold
4. **Validation**: Check if prediction matches actual 24h outcome
5. **Feedback**: Adjust weights to improve accuracy
6. **Repeat**: Until validation accuracy stops improving

**The model doesn't know about "profit" directly** - it learns to predict **direction** (up/down/sideways). The trading strategy then uses these predictions to make profitable trades.

## Key Insight

The model learns that:
- **Buy signal** = "Price will move up by >0.5×ATR in next 24h"
- **Sell signal** = "Price will move down by >0.5×ATR in next 24h"
- **Hold signal** = "Price will stay within ±0.5×ATR range"

This is more realistic than fixed pips because it adapts to market volatility!

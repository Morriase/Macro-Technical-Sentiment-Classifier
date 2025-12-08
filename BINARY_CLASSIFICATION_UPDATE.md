# Binary Classification & Data Filtering Update

## Overview
The system has been redesigned from a 3-class (Buy/Sell/Hold) to a **Binary Classification (Buy/Sell)** model. The "Hold" class has been removed by filtering out small price moves from the training data entirely.

## Key Changes

### 1. Training Pipeline (`main.py`)
- **Binary Target**: Targets are now strictly 0 (Sell) or 1 (Buy).
- **Data Filtering**: 
  - Instead of labeling small moves as "Hold", these samples are **removed** from the training set.
  - Filtering uses `min_move_pips` (fixed) or `atr_multiplier` (dynamic) to define "small moves".
  - This forces the model to learn only from significant price movements.

### 2. Model Architecture
- **XGBoost**: `num_class` set to 2.
- **LSTM**: `num_classes` set to 2.
- **Hybrid Ensemble**: 
  - Meta-learner now trains on 2-class probabilities.
  - Class weights set to `'balanced'` to handle any remaining imbalance.

### 3. Inference Server (`inference_server.py`)
- **Output Mapping**: 
  - 0 = **SELL**
  - 1 = **BUY**
- **Probabilities**: Response now returns `BUY` and `SELL` probabilities. `HOLD` probability is hardcoded to 0.0.
- **Signal Quality**: Fuzzy logic scorer updated to interpret binary classes correctly.

### 4. Configuration (`src/config.py`)
- `num_classes` updated to 2.

## Next Steps
1. **Retrain Models**: You must retrain the models on Kaggle for these changes to take effect.
2. **Deploy**: After training, download the new models and deploy the updated `inference_server.py`.

## Verification
- `test_inference_server.py` has been updated to validate the new binary response structure.

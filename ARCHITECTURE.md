# Architecture Clarification: LSTM-XGBoost Hybrid

## Overview
This document clarifies the **correct architecture** as specified in `required.txt`.

## The Correct Architecture (Now Implemented)

### Two-Level Stacking Ensemble

**Level-0 Base Learners:**
1. **XGBoost Classifier**
   - Purpose: Handles tabular features (technical indicators, macro features, sentiment scores)
   - Strength: Excellent at capturing non-linear relationships and feature interactions
   - Output: 3-class probability distribution (Buy/Sell/Hold)

2. **LSTM Sequence Model**
   - Purpose: Captures temporal dependencies and sequential patterns
   - Strength: Remembers long-term patterns in financial time series
   - Architecture: 2-layer bidirectional LSTM with dropout
   - Sequence Length: 22 timesteps (~1 month of trading days)
   - Output: 3-class probability distribution (Buy/Sell/Hold)

**Level-1 Meta-Classifier:**
- **XGBoost Meta-Classifier**
   - Purpose: Learns optimal combination of base learner predictions
   - Input: Concatenated probability outputs from both base learners (6 features total)
   - Output: Final 3-class prediction

### Why This Architecture?

1. **Complementary Strengths:**
   - XGBoost: Excels at feature interactions, handles mixed data types, robust to noise
   - LSTM: Captures temporal dependencies, models sequential patterns, learns time-varying relationships

2. **Financial Time Series Requirements:**
   - Markets have both **cross-sectional patterns** (feature relationships at a point in time) ✓ XGBoost
   - Markets have **temporal patterns** (momentum, regime changes, cycles) ✓ LSTM

3. **Data Leakage Prevention:**
   - Uses Out-of-Fold (OOF) predictions to train meta-learner
   - Ensures meta-learner never sees training data base learners were trained on

## Files Modified

### 1. `src/models/lstm_model.py` (NEW)
- **LSTMSequenceClassifier**: PyTorch nn.Module with 2-layer LSTM
- **LSTMSequenceModel**: Training wrapper with:
  - Sequence preparation (2D → 3D transformation)
  - MinMax normalization
  - Early stopping
  - Model persistence

### 2. `src/models/hybrid_ensemble.py` (REWRITTEN)
- **HybridEnsemble**: Complete stacking implementation
  - `generate_out_of_fold_predictions()`: 5-fold CV for OOF predictions
  - `fit()`: Three-stage training (OOF → base learners → meta-learner)
  - `predict_proba()`: Inference pipeline
  - Model save/load functionality

### 3. `src/config.py` (UPDATED)
- Updated `ENSEMBLE_CONFIG` to specify LSTM parameters
- Removed Random Forest configuration
- Added LSTM hyperparameters (sequence_length, hidden_size, etc.)
- Changed meta_learner from MLP to XGBoost

## Training Flow

```
1. Input: Feature matrix X (n_samples, n_features)
   
2. Generate OOF Predictions (prevents leakage):
   ├─ 5-Fold Stratified CV
   │  ├─ Fold 1: Train XGBoost + LSTM on 80% → Predict on 20%
   │  ├─ Fold 2: Train XGBoost + LSTM on 80% → Predict on 20%
   │  └─ ... (5 folds total)
   └─ Result: OOF probability matrix (n_samples, 6)
   
3. Train Base Learners on Full Dataset:
   ├─ XGBoost: Fit on all samples
   └─ LSTM: Fit on all samples
   
4. Train Meta-Classifier:
   └─ XGBoost Meta: Fit on OOF predictions from step 2
```

## Inference Flow

```
New Data → Scale Features → XGBoost Base → Proba (3 classes)
                          ↘                           ↓
                           LSTM Base → Proba (3)  →  Concatenate (6 features)
                                                      ↓
                                                   XGBoost Meta
                                                      ↓
                                                Final Prediction
```

## Key Implementation Details

### LSTM Sequence Preparation
```python
# Input: X with shape (1000, 50)  # 1000 samples, 50 features
# Process: Create sliding windows of length 22
# Output: X_seq with shape (979, 22, 50)  # Lost sequence_length-1 samples
```

### Meta-Features
```python
# XGBoost output: (n_samples, 3) - [P(Buy), P(Sell), P(Hold)]
# LSTM output: (n_samples, 3) - [P(Buy), P(Sell), P(Hold)]
# Concatenated: (n_samples, 6) - All probabilities
```

## Differences from required2.txt

| Component | required.txt (CORRECT) | required2.txt (ALTERNATIVE) |
|-----------|----------------------|--------------------------|
| Base Learner 1 | XGBoost | XGBoost |
| Base Learner 2 | **LSTM** | Random Forest |
| Meta-Learner | **XGBoost** | MLP |
| Temporal Modeling | ✅ Yes (LSTM) | ❌ No |
| Complexity | Higher | Lower |

## Performance Expectations

### Individual Base Learners:
- **XGBoost alone**: ~55-58% accuracy (better than random)
- **LSTM alone**: ~53-56% accuracy (captures sequences)

### Ensemble:
- **Combined**: ~58-62% accuracy (synergistic improvement)
- **Confidence filtering**: Higher accuracy on high-confidence predictions (>0.60)

## Next Steps

1. ✅ LSTM model implemented
2. ✅ Hybrid ensemble rewritten
3. ✅ Configuration updated
4. ⏳ Complete dependency installation
5. ⏳ Test LSTM sequence transformation
6. ⏳ Validate OOF prediction generation
7. ⏳ Run end-to-end training pipeline

## Usage Example

```python
from src.models.hybrid_ensemble import HybridEnsemble

# Initialize
ensemble = HybridEnsemble(
    xgb_params={...},
    lstm_params={
        "sequence_length": 22,
        "hidden_size": 128,
        "epochs": 100,
    },
    n_folds=5,
)

# Train
ensemble.fit(X_train, y_train, X_val, y_val)

# Predict
predictions = ensemble.predict_proba(X_test)

# Analyze base learners
xgb_proba, lstm_proba = ensemble.get_base_learner_predictions(X_test)
```

## Why LSTM Was Originally Missing

The initial implementation referenced `required2.txt` which specifies a **Random Forest + MLP** architecture. However, the primary research document (`required.txt`) emphasizes **LSTM for temporal sequence modeling**, which is more appropriate for financial time series prediction.

**Key Quote from required.txt:**
> "LSTM networks capture temporal dependencies that gradient boosting misses"

This has now been corrected to match the primary specification.

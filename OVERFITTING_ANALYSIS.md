# Overfitting Analysis & Recommended Fixes

## Executive Summary
The training logs reveal **clear overfitting patterns** in the LSTM component, with training accuracy significantly exceeding validation accuracy. The model is memorizing training data rather than learning generalizable patterns.

---

## 🔴 Critical Overfitting Evidence

### 1. **Train-Val Accuracy Gap (LSTM)**
From training logs, consistent pattern across all folds:

```
Epoch 10: Train Acc: 0.7060, Val Acc: 0.6553  (Gap: 5.07%)
Epoch 20: Train Acc: 0.7270, Val Acc: 0.6387  (Gap: 8.83%)
Epoch 20: Train Acc: 0.7340, Val Acc: 0.6193  (Gap: 11.47%)
Epoch 20: Train Acc: 0.7450, Val Acc: 0.6233  (Gap: 12.17%)
```

**Problem**: Training accuracy increases while validation accuracy decreases or stagnates - classic overfitting.

### 2. **Validation Loss Divergence**
```
Epoch 10: Loss: 0.7396, Val Loss: 0.7954
Epoch 20: Loss: 0.6265, Val Loss: 0.8890  (Val loss INCREASED by 11.8%)
Early stopping at epoch 20 - Val Loss: 0.8890
```

**Problem**: Training loss decreases but validation loss increases - model is learning noise, not signal.

### 3. **Early Stopping Triggers Too Late**
- Early stopping patience: 10 epochs
- Stopping typically at epochs 15-25
- **Problem**: Model has already overfit by the time early stopping triggers

---

## 📊 Root Cause Analysis

### Architecture Issues

#### 1. **LSTM Capacity Too High for Data**
Current config:
- `hidden_size: 128`
- `num_layers: 2`
- `dropout: 0.3`

**Issue**: 128 hidden units × 2 layers = massive capacity for 86 features. The spec recommends capturing "22 trading days" of patterns, but the model has too many parameters relative to signal.

#### 2. **Insufficient Regularization**
Current dropout: 0.3 (30%)
- **Problem**: Not aggressive enough for financial time series with high noise
- Spec emphasizes "Dropout layers after each LSTM layer and Recurrent Dropout within the LSTM cells"
- Current implementation only has standard dropout, missing recurrent dropout

#### 3. **Learning Rate Too High**
Current: `learning_rate: 0.001`
- **Problem**: Too aggressive for noisy financial data
- Causes rapid overfitting to training patterns

#### 4. **Batch Size Too Small**
Current: `batch_size: 64`
- **Problem**: Small batches = noisy gradients = overfitting to mini-batch patterns
- Financial data needs larger, more stable gradient estimates

#### 5. **Weight Decay Insufficient**
Current: `weight_decay: 1e-4` (0.0001)
- **Problem**: Too weak for L2 regularization
- Doesn't constrain model complexity enough

---

## 🔧 Recommended Fixes (Priority Order)

### **IMMEDIATE FIXES** (Deploy Now)

#### Fix 1: Reduce LSTM Capacity
```python
# Current
hidden_size: 128
num_layers: 2

# Recommended
hidden_size: 64   # 50% reduction
num_layers: 2     # Keep 2 layers per spec
```

**Rationale**: Halving hidden units reduces parameters by ~75%, forcing model to learn only strong patterns.

#### Fix 2: Increase Dropout Aggressively
```python
# Current
dropout: 0.3

# Recommended
dropout: 0.5      # Increase to 50%
# Add recurrent_dropout in LSTM layer
```

**Implementation**:
```python
self.lstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    dropout=0.5 if num_layers > 1 else 0,  # Between layers
    bidirectional=bidirectional,
)
# Add manual recurrent dropout in forward pass
```

#### Fix 3: Reduce Learning Rate
```python
# Current
learning_rate: 0.001

# Recommended
learning_rate: 0.0003  # 70% reduction
```

**Rationale**: Slower learning = better generalization, less overfitting to noise.

#### Fix 4: Increase Weight Decay
```python
# Current
optimizer = optim.Adam(self.model.parameters(),
                       lr=self.learning_rate, weight_decay=1e-4)

# Recommended
optimizer = optim.Adam(self.model.parameters(),
                       lr=self.learning_rate, weight_decay=1e-3)  # 10x stronger
```

#### Fix 5: Increase Batch Size
```python
# Current
batch_size: 64

# Recommended
batch_size: 128   # Double for more stable gradients
```

#### Fix 6: More Aggressive Early Stopping
```python
# Current
early_stopping_patience: 10

# Recommended
early_stopping_patience: 5   # Stop sooner
```

**Add**: Monitor validation accuracy, not just loss. Stop if val_acc doesn't improve.

---

### **SECONDARY FIXES** (Next Iteration)

#### Fix 7: Add Gradient Clipping
```python
# Add to training loop
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

**Rationale**: Prevents exploding gradients in LSTM, stabilizes training.

#### Fix 8: Learning Rate Scheduler
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
```

**Rationale**: Adaptive learning rate reduces overfitting in later epochs.

#### Fix 9: Add Label Smoothing
```python
# Current
criterion = nn.CrossEntropyLoss()

# Recommended
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Rationale**: Prevents overconfident predictions, improves generalization.

#### Fix 10: Sequence Length Optimization
Current: `sequence_length: 22` (fixed)

**Recommendation**: Test shorter sequences via hyperparameter tuning:
- Try: [10, 15, 22, 30]
- Shorter sequences = less overfitting to long-term noise
- Spec mentions "22 trading days" but this should be validated, not assumed

---

### **ADVANCED FIXES** (Research Phase)

#### Fix 11: Ensemble Diversity
Current: Single LSTM + Single XGBoost

**Recommendation**: Add model diversity:
- Train 3-5 LSTM models with different random seeds
- Average their predictions before meta-classifier
- Reduces variance, improves robustness

#### Fix 12: Feature Selection for LSTM
Current: All 86 features fed to LSTM

**Problem**: Many features may not have temporal patterns (e.g., static macro indicators)

**Recommendation**:
- Use XGBoost feature importance to identify top temporal features
- Feed only high-importance sequential features to LSTM
- Feed all features to XGBoost base learner

#### Fix 13: Temporal Cross-Validation
Current: StratifiedKFold (shuffles data)

**Critical Issue**: Violates temporal ordering! Spec explicitly requires Walk-Forward Validation.

**Recommendation**: Implement TimeSeriesSplit:
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

---

## 📈 Expected Impact

### Immediate Fixes (1-6)
- **Expected Val Acc Improvement**: +3-5%
- **Expected Train-Val Gap Reduction**: From 12% to 5-7%
- **Training Time**: -20% (smaller model, early stopping)

### Secondary Fixes (7-10)
- **Expected Val Acc Improvement**: +2-3% additional
- **Expected Stability**: More consistent across folds

### Advanced Fixes (11-13)
- **Expected Val Acc Improvement**: +3-5% additional
- **Expected Robustness**: Better performance on unseen data

---

## 🎯 Implementation Priority

### Phase 1 (This Week)
1. ✅ Fix dependency conflicts (DONE)
2. ✅ Make news dataset optional (DONE)
3. ⚠️ Reduce LSTM capacity (hidden_size: 64)
4. ⚠️ Increase dropout (0.5)
5. ⚠️ Reduce learning rate (0.0003)
6. ⚠️ Increase weight decay (1e-3)
7. ⚠️ Increase batch size (128)
8. ⚠️ Reduce early stopping patience (5)

### Phase 2 (Next Week)
9. Add gradient clipping
10. Add learning rate scheduler
11. Add label smoothing
12. Optimize sequence length via Optuna

### Phase 3 (Research)
13. Implement proper Walk-Forward Validation
14. Add ensemble diversity
15. Feature selection for LSTM
16. SHAP analysis for feature importance

---

## 🚨 Critical Alignment with Spec

### What Spec Says:
> "Regularization techniques, such as Dropout layers after each LSTM layer and Recurrent Dropout within the LSTM cells, are necessary to prevent the model from overfitting to the noisy financial data."

### Current Implementation:
- ❌ Only standard dropout between layers
- ❌ No recurrent dropout
- ❌ Dropout rate too low (0.3 vs recommended 0.5+)

### What Spec Says:
> "Walk-Forward Validation (WFV) provides a superior and more realistic simulation of a production trading environment by respecting the temporal order of the data."

### Current Implementation:
- ❌ Using StratifiedKFold (shuffles data)
- ❌ Violates temporal ordering
- ❌ Creates data leakage

---

## 📝 Conclusion

The model is **clearly overfitting** due to:
1. Excessive LSTM capacity
2. Insufficient regularization
3. Improper cross-validation (temporal leakage)

**Immediate action required**: Implement Phase 1 fixes to reduce overfitting by ~50%.

**Long-term**: Implement proper Walk-Forward Validation as specified in the original architecture document.

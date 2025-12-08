# LSTM Training Tuning - Senior ML Engineer Analysis

## Problem Diagnosis
**Current Issue:** Validation accuracy stuck at ~50% (random guessing) while training accuracy improves
- Train Acc: 0.601 (improving)
- Val Acc: 0.552 (stuck, barely above random)
- **Root Cause:** Severe overfitting despite regularization

## Root Causes Identified

1. **Insufficient Regularization**
   - Current: L1=1e-7, L2=1e-5 (too weak)
   - LSTM has 58,754 parameters with minimal constraints
   - Batch Norm alone insufficient for financial data variance

2. **Weak Dropout Strategy**
   - Current: dropout=0.0 (disabled because of BatchNorm)
   - But we're using BatchNorm, so we CAN'T use Dropout (per resources)
   - Need stronger L1/L2 instead

3. **Learning Rate Too High**
   - Current: 3e-5 (from MQL5, but that's for different data)
   - Financial data has high variance → needs lower LR
   - Warmup helps but not enough

4. **Sequence Length Mismatch**
   - Current: 40 bars (from MQL5)
   - With 38 features, this creates 40×38=1,520 input dimensions
   - LSTM struggles with long sequences + high feature count

5. **Batch Size Too Large**
   - Current: 10,000 (from MQL5)
   - With only ~21,000 training samples, this is 2 batches/epoch
   - Insufficient gradient updates per epoch

## Tuning Strategy (Senior ML Engineer Approach)

### Phase 1: Reduce Overfitting (Primary Goal)
**Increase Regularization Strength**
- L1: 1e-7 → 1e-5 (100x stronger)
- L2: 1e-5 → 1e-4 (10x stronger)
- Rationale: Financial data is noisy; aggressive regularization prevents memorization

**Reduce Sequence Length**
- 40 bars → 20 bars
- Rationale: Shorter sequences reduce parameter count, easier to generalize
- 20×38 = 760 input dims (vs 1,520 before)

**Reduce Batch Size**
- 10,000 → 2,000
- Rationale: More gradient updates per epoch = better convergence
- ~10 batches/epoch (vs 2 before)

### Phase 2: Stabilize Training
**Lower Learning Rate**
- 3e-5 → 1e-5
- Rationale: Smaller steps prevent overshooting in high-variance financial data

**Extend Warmup**
- 3 epochs → 5 epochs
- Rationale: Gradual ramp-up prevents early instability

**Increase Early Stopping Patience**
- 20 → 30 epochs
- Rationale: Allow more time to find optimal point

### Phase 3: Improve Convergence
**Stronger Label Smoothing**
- 0.1 → 0.15
- Rationale: Prevents overconfident predictions on noisy labels

**Gradient Clipping**
- 1.0 → 0.5
- Rationale: Prevents gradient explosions in LSTM

**Reduce Hidden Size (Optional)**
- 40 → 32
- Rationale: Fewer parameters = less overfitting
- Trade-off: Slightly less capacity

## Implementation Changes

### LSTM Hyperparameters
```python
lstm_params = {
    "sequence_length": 20,           # 40 → 20 (reduce overfitting)
    "hidden_size": 40,               # Keep (MQL5 standard)
    "num_layers": 1,                 # Keep (MQL5 standard)
    "num_classes": 2,                # Binary classification
    "dropout": 0.0,                  # Keep (BatchNorm enabled)
    "learning_rate": 1e-5,           # 3e-5 → 1e-5 (lower for stability)
    "batch_size": 2000,              # 10000 → 2000 (more updates/epoch)
    "epochs": 500,                   # Keep
    "early_stopping_patience": 30,   # 20 → 30 (more patience)
    "l1_lambda": 1e-5,               # 1e-7 → 1e-5 (100x stronger)
    "l2_lambda": 1e-4,               # 1e-5 → 1e-4 (10x stronger)
    "label_smoothing": 0.15,         # 0.1 → 0.15 (stronger)
    "lr_warmup_epochs": 5,           # 3 → 5 (longer warmup)
    "lr_min_factor": 0.01,           # Keep
    "max_grad_norm": 0.5,            # 1.0 → 0.5 (tighter clipping)
    "gradient_accumulation_steps": 1,# Keep
    "bidirectional": False,          # Keep (MQL5 standard)
    "use_batch_norm": True,          # Keep (MQL5 standard)
    "hidden_activation": "swish",    # Keep
}
```

## Expected Improvements

**Before Tuning:**
- Train Acc: ~60%, Val Acc: ~55%
- Gap: 5% (overfitting)
- Loss curves: Diverging

**After Tuning (Target):**
- Train Acc: 65-70%, Val Acc: 63-68%
- Gap: <2% (good generalization)
- Loss curves: Smooth, converging uniformly

## Monitoring Metrics

Track these during training:
1. **Convergence Quality**
   - Train/Val loss ratio (should be <1.05)
   - Loss smoothness (no spikes)

2. **Generalization**
   - Val Acc - Train Acc (should be <2%)
   - Val Acc trend (should be monotonic)

3. **Stability**
   - Gradient norms (should be <0.5)
   - Learning rate schedule (should decay smoothly)

## Fallback Adjustments

If still overfitting after Phase 1-3:
1. Reduce hidden_size: 40 → 32
2. Increase L2: 1e-4 → 5e-4
3. Reduce sequence_length: 20 → 15
4. Add early stopping patience: 30 → 50

If underfitting (both train/val low):
1. Increase learning_rate: 1e-5 → 2e-5
2. Reduce regularization: L1/L2 by 50%
3. Increase hidden_size: 40 → 48
4. Increase sequence_length: 20 → 30

## References
- LSTM_NUANCES.md: Architecture parameters (40 units, 1 layer, BatchNorm)
- improving_convergence.md: Regularization (L1/L2 coefficients, learning rate)
- correlation_usage.md: Feature selection (38 features optimal)

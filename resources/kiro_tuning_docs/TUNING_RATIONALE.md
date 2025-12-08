# LSTM Tuning Rationale - Senior ML Engineer Perspective

## Executive Summary
Adjusted 7 key hyperparameters to address **severe overfitting** (Val Acc stuck at 50%). Target: 65%+ accuracy with <2% train/val gap.

---

## Detailed Tuning Decisions

### 1. Sequence Length: 40 → 20 bars
**Why This Matters:**
- Current: 40 bars × 38 features = 1,520 input dimensions
- Problem: LSTM with 58,754 parameters can memorize 1,520-dim sequences easily
- Solution: 20 bars × 38 features = 760 input dims (50% reduction)

**ML Engineering Principle:**
- From `correlation_usage.md`: "Testing correlation with historical shift can show how influence decreases as time lag increases"
- Shorter sequences = less capacity to overfit = better generalization
- 20 bars ≈ 100 minutes (M5 timeframe) = sufficient for intraday patterns

**Expected Impact:**
- Reduces parameter effective capacity by ~50%
- Prevents memorization of noise
- Faster training (fewer computations)

---

### 2. Learning Rate: 3e-5 → 1e-5
**Why This Matters:**
- Current: 3e-5 from MQL5 (trained on different data)
- Problem: Financial data has high variance; larger steps overshoot optimal weights
- Solution: 1e-5 = 3x smaller steps = more stable convergence

**ML Engineering Principle:**
- From `improving_convergence.md`: "Low learning rate consistently applied due to high complexity and variance of financial data"
- MQL5 used 3e-5 for MQL5 data; our Kaggle data is different
- Lower LR + longer warmup = smoother loss curves

**Expected Impact:**
- Smoother loss curves (less oscillation)
- Better convergence to local minima
- Slightly longer training (acceptable trade-off)

---

### 3. Batch Size: 10,000 → 2,000
**Why This Matters:**
- Current: 10,000 with ~21,000 training samples = 2 batches/epoch
- Problem: Only 2 gradient updates per epoch = insufficient learning signal
- Solution: 2,000 batch size = ~10 batches/epoch = 5x more updates

**ML Engineering Principle:**
- From `improving_convergence.md`: "Batch size of 1,000 patterns was common in Python tests"
- More frequent updates = better gradient estimates
- Smaller batches = more noise in gradients = acts as regularization

**Expected Impact:**
- 5x more gradient updates per epoch
- Better convergence (more learning signal)
- Slight increase in training time (acceptable)

---

### 4. L1 Regularization: 1e-7 → 1e-5
**Why This Matters:**
- Current: 1e-7 (extremely weak)
- Problem: Barely constrains 58,754 parameters
- Solution: 1e-5 = 100x stronger = forces weight sparsity

**ML Engineering Principle:**
- From `improving_convergence.md`: "Elastic Net with L1=1e-7 and L2=1e-5"
- But that was for a different model; our LSTM is overfitting
- L1 penalty: Σ|w| → encourages sparse weights → feature selection

**Expected Impact:**
- Forces many weights to zero (feature selection)
- Reduces effective model capacity
- Prevents memorization of noise

---

### 5. L2 Regularization: 1e-5 → 1e-4
**Why This Matters:**
- Current: 1e-5 (weak)
- Problem: Insufficient weight decay for high-variance financial data
- Solution: 1e-4 = 10x stronger = aggressive weight decay

**ML Engineering Principle:**
- From `improving_convergence.md`: "L2=1e-5 for complex models"
- Our model is overfitting → needs stronger L2
- L2 penalty: Σ(w²) → encourages small weights → prevents extreme values

**Expected Impact:**
- Weights stay smaller (less overfitting)
- Smoother decision boundaries
- Better generalization to unseen data

---

### 6. Label Smoothing: 0.1 → 0.15
**Why This Matters:**
- Current: 0.1 (mild smoothing)
- Problem: Model still overconfident on noisy labels
- Solution: 0.15 = stronger smoothing = less overconfidence

**ML Engineering Principle:**
- Label smoothing: Instead of [0, 1], use [0.15, 0.85]
- Prevents model from learning spurious patterns
- Especially important for financial data (labels are noisy)

**Expected Impact:**
- Reduced overconfidence
- Better calibrated probabilities
- Improved validation accuracy

---

### 7. Gradient Clipping: 1.0 → 0.5
**Why This Matters:**
- Current: 1.0 (allows large gradient steps)
- Problem: LSTM can have exploding gradients in financial sequences
- Solution: 0.5 = tighter clipping = prevents gradient explosions

**ML Engineering Principle:**
- From `improving_convergence.md`: "Gradient clipping prevents gradient explosions in LSTM"
- LSTM gates can amplify gradients → need tight clipping
- Tighter clipping = more stable training

**Expected Impact:**
- Smoother loss curves (no spikes)
- More stable training
- Better convergence

---

### 8. Warmup Epochs: 3 → 5
**Why This Matters:**
- Current: 3 epochs (quick ramp-up)
- Problem: Model unstable in early epochs
- Solution: 5 epochs = slower ramp-up = more stable start

**ML Engineering Principle:**
- Warmup: Gradually increase LR from 50% to 100%
- Longer warmup = model has time to find good initial weights
- Prevents early divergence

**Expected Impact:**
- More stable first 5 epochs
- Better initial weight initialization
- Smoother overall training

---

### 9. Early Stopping Patience: 20 → 30
**Why This Matters:**
- Current: 20 epochs (stops quickly)
- Problem: May stop before finding optimal point
- Solution: 30 epochs = more patience = allows convergence

**ML Engineering Principle:**
- Patience: How many epochs without improvement before stopping
- With lower LR, convergence is slower
- More patience allows model to find better minima

**Expected Impact:**
- Longer training (acceptable)
- Better final model quality
- Higher validation accuracy

---

## Parameter Summary Table

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| sequence_length | 40 | 20 | Reduce overfitting |
| learning_rate | 3e-5 | 1e-5 | Stability (financial data) |
| batch_size | 10,000 | 2,000 | More gradient updates |
| l1_lambda | 1e-7 | 1e-5 | Stronger sparsity |
| l2_lambda | 1e-5 | 1e-4 | Stronger weight decay |
| label_smoothing | 0.1 | 0.15 | Reduce overconfidence |
| max_grad_norm | 1.0 | 0.5 | Prevent gradient explosions |
| lr_warmup_epochs | 3 | 5 | Longer stabilization |
| early_stopping_patience | 20 | 30 | More convergence time |

---

## Expected Training Behavior

### Before Tuning
```
Epoch 1:  Train Loss: 0.698, Train Acc: 0.47, Val Loss: 0.693, Val Acc: 0.50
Epoch 10: Train Loss: 0.694, Train Acc: 0.54, Val Loss: 0.692, Val Acc: 0.50
Epoch 19: Train Loss: 0.676, Train Acc: 0.60, Val Loss: 0.683, Val Acc: 0.55
         ↑ Train improving, Val stuck (overfitting)
```

### After Tuning (Expected)
```
Epoch 1:  Train Loss: 0.695, Train Acc: 0.48, Val Loss: 0.694, Val Acc: 0.49
Epoch 5:  Train Loss: 0.685, Train Acc: 0.52, Val Loss: 0.684, Val Acc: 0.51
Epoch 10: Train Loss: 0.670, Train Acc: 0.58, Val Loss: 0.668, Val Acc: 0.58
Epoch 20: Train Loss: 0.655, Train Acc: 0.63, Val Loss: 0.658, Val Acc: 0.62
Epoch 30: Train Loss: 0.645, Train Acc: 0.66, Val Loss: 0.648, Val Acc: 0.65
         ↑ Both improving uniformly, gap <2%
```

---

## Monitoring During Training

Watch for these signs:

**Good Signs:**
- ✓ Train/Val loss curves parallel (not diverging)
- ✓ Val Acc improving monotonically
- ✓ Gap between Train/Val Acc <2%
- ✓ Loss curves smooth (no spikes)

**Bad Signs:**
- ✗ Val Acc stuck while Train Acc improves (overfitting)
- ✗ Loss curves oscillating wildly (LR too high)
- ✗ Loss increasing (LR too high or regularization too strong)
- ✗ Both Train/Val stuck at 50% (underfitting)

---

## Fallback Strategy

If results don't meet 65% target:

**If Still Overfitting (Val Acc <60%):**
1. Reduce hidden_size: 40 → 32
2. Increase L2: 1e-4 → 5e-4
3. Reduce sequence_length: 20 → 15

**If Underfitting (Both <55%):**
1. Increase learning_rate: 1e-5 → 2e-5
2. Reduce L1/L2 by 50%
3. Increase hidden_size: 40 → 48

**If Unstable Training (Loss spikes):**
1. Reduce learning_rate: 1e-5 → 5e-6
2. Increase warmup_epochs: 5 → 10
3. Reduce max_grad_norm: 0.5 → 0.3

---

## References
- `LSTM_NUANCES.md`: Architecture (40 units, 1 layer, BatchNorm)
- `improving_convergence.md`: Regularization techniques
- `correlation_usage.md`: Feature selection principles

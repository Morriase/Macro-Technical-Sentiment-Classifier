# LSTM Tuning - Quick Reference Card

## Changes Applied (7 Parameters)

```python
# BEFORE → AFTER
sequence_length:        40 → 20      # Reduce overfitting
learning_rate:          3e-5 → 1e-5  # Stability
batch_size:             10000 → 2000 # More updates/epoch
l1_lambda:              1e-7 → 1e-5  # Stronger sparsity
l2_lambda:              1e-5 → 1e-4  # Stronger decay
label_smoothing:        0.1 → 0.15   # Less overconfidence
max_grad_norm:          1.0 → 0.5    # Prevent explosions
lr_warmup_epochs:       3 → 5        # Longer stabilization
early_stopping_patience: 20 → 30     # More convergence time
```

## Target Metrics
- **Train Accuracy:** 65-70%
- **Val Accuracy:** 63-68%
- **Train/Val Gap:** <2%
- **Loss Curves:** Smooth, parallel, converging

## Key Principles Applied
1. ✓ **No Dropout + BatchNorm** (per resources)
2. ✓ **Aggressive Regularization** (L1/L2 100x-10x stronger)
3. ✓ **Lower Learning Rate** (financial data variance)
4. ✓ **Shorter Sequences** (reduce memorization)
5. ✓ **More Gradient Updates** (smaller batches)
6. ✓ **Tighter Gradient Clipping** (LSTM stability)
7. ✓ **Longer Warmup** (early stability)

## Monitoring Checklist
- [ ] Val Acc improving (not stuck at 50%)
- [ ] Train/Val loss curves parallel
- [ ] No loss spikes (gradient explosions)
- [ ] Smooth convergence
- [ ] Gap between Train/Val <2%

## If Not Working

**Still Overfitting?**
- Reduce hidden_size: 40 → 32
- Increase L2: 1e-4 → 5e-4
- Reduce sequence_length: 20 → 15

**Underfitting?**
- Increase learning_rate: 1e-5 → 2e-5
- Reduce L1/L2 by 50%
- Increase hidden_size: 40 → 48

**Unstable?**
- Reduce learning_rate: 1e-5 → 5e-6
- Increase warmup_epochs: 5 → 10
- Reduce max_grad_norm: 0.5 → 0.3

## Implementation Location
File: `src/models/hybrid_ensemble.py` (lines ~130-160)
Section: `lstm_params` dictionary in `HybridEnsemble.__init__()`

## Expected Training Timeline
- Epochs 1-5: Warmup phase (loss decreasing, acc ~50%)
- Epochs 5-15: Rapid improvement (acc 50% → 60%)
- Epochs 15-30: Convergence (acc 60% → 65%+)
- Epochs 30+: Plateau (early stopping triggers)

## Success Criteria
✓ Val Acc ≥ 65% by epoch 30
✓ Train/Val gap ≤ 2%
✓ Smooth loss curves
✓ No divergence between train/val

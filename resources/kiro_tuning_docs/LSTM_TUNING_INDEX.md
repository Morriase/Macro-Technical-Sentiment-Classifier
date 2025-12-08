# LSTM Training Tuning - Complete Documentation Index

## Quick Start
**New to this tuning?** Start here:
1. Read: `TUNING_QUICK_REFERENCE.md` (2 min read)
2. Read: `LSTM_TUNING_SUMMARY.md` (5 min read)
3. Run training with new parameters

---

## Documentation Structure

### 📋 Overview Documents
| Document | Purpose | Read Time |
|----------|---------|-----------|
| **TUNING_QUICK_REFERENCE.md** | Quick reference card with all changes | 2 min |
| **LSTM_TUNING_SUMMARY.md** | Complete summary of tuning | 5 min |
| **TUNING_CHECKLIST.md** | Pre-training and monitoring checklist | 3 min |

### 📚 Detailed Documentation
| Document | Purpose | Read Time |
|----------|---------|-----------|
| **LSTM_TRAINING_TUNING.md** | Detailed tuning strategy and rationale | 10 min |
| **TUNING_RATIONALE.md** | Deep dive into each parameter change | 15 min |
| **LSTM_TUNING_INDEX.md** | This file - navigation guide | 5 min |

### 🔍 Reference Documents
| Document | Purpose | Read Time |
|----------|---------|-----------|
| **LSTM_NUANCES.md** | Architecture principles from MQL5 | 10 min |
| **improving_convergence.md** | Regularization techniques | 10 min |
| **correlation_usage.md** | Feature selection principles | 10 min |

---

## Problem & Solution

### Problem
```
Current Training:
  Train Acc: ~60% (improving)
  Val Acc: ~55% (STUCK)
  Gap: 5% (severe overfitting)
```

### Root Causes
1. Weak regularization (L1=1e-7, L2=1e-5)
2. Long sequences (40 bars = 1,520 input dims)
3. Large batches (10,000 = only 2 batches/epoch)
4. High learning rate (3e-5 for financial data)
5. Loose gradient clipping (1.0)

### Solution
Adjusted 9 parameters following ML engineering principles:
- Aggressive regularization (L1/L2 100x-10x stronger)
- Shorter sequences (20 bars = 760 input dims)
- Smaller batches (2,000 = 10 batches/epoch)
- Lower learning rate (1e-5)
- Tighter gradient clipping (0.5)

---

## Parameter Changes (9 Total)

| # | Parameter | Before | After | Impact |
|---|-----------|--------|-------|--------|
| 1 | sequence_length | 40 | 20 | -50% input dims |
| 2 | learning_rate | 3e-5 | 1e-5 | Smoother convergence |
| 3 | batch_size | 10,000 | 2,000 | 5x more updates/epoch |
| 4 | l1_lambda | 1e-7 | 1e-5 | 100x stronger L1 |
| 5 | l2_lambda | 1e-5 | 1e-4 | 10x stronger L2 |
| 6 | label_smoothing | 0.1 | 0.15 | Reduce overconfidence |
| 7 | max_grad_norm | 1.0 | 0.5 | Prevent explosions |
| 8 | lr_warmup_epochs | 3 | 5 | Longer stabilization |
| 9 | early_stopping_patience | 20 | 30 | More convergence time |

---

## Target Metrics

**Success Criteria:**
- ✓ Val Accuracy ≥ 65% by epoch 30
- ✓ Train/Val gap ≤ 2%
- ✓ Smooth loss curves (no spikes)
- ✓ No divergence between train/val

**Expected Results:**
- Train Accuracy: 65-70%
- Val Accuracy: 63-68%
- Loss Curves: Parallel, converging uniformly

---

## Implementation

**File:** `src/models/hybrid_ensemble.py`
**Section:** `HybridEnsemble.__init__()` method
**Lines:** ~130-160 (lstm_params dictionary)
**Status:** ✓ APPLIED

---

## Training Phases

### Phase 1: Warmup (Epochs 1-5)
- Learning rate ramps from 50% to 100%
- Loss decreases gradually
- Accuracy ~50% (random baseline)
- **Goal:** Stabilize initial weights

### Phase 2: Rapid Improvement (Epochs 5-15)
- Learning rate stable
- Loss decreases rapidly
- Accuracy: 50% → 60%
- Train/Val curves parallel
- **Goal:** Learn main patterns

### Phase 3: Convergence (Epochs 15-30)
- Learning rate decays (cosine annealing)
- Loss plateaus
- Accuracy: 60% → 65%+
- Train/Val gap <2%
- **Goal:** Fine-tune weights

### Phase 4: Plateau (Epochs 30+)
- Loss stable
- Accuracy stable
- Early stopping triggers
- **Goal:** Stop before overfitting

---

## Monitoring During Training

### Good Signs ✓
- Val Acc improving (not stuck at 50%)
- Train/Val loss curves parallel
- No loss spikes
- Smooth convergence
- Gap between Train/Val <2%

### Warning Signs ✗
- Val Acc stuck while Train Acc improves
- Loss curves diverging
- Loss spikes (gradient explosions)
- Both Train/Val stuck at 50%

---

## Fallback Adjustments

### If Still Overfitting (Val Acc <60%)
```python
hidden_size: 40 → 32
l2_lambda: 1e-4 → 5e-4
sequence_length: 20 → 15
```

### If Underfitting (Both <55%)
```python
learning_rate: 1e-5 → 2e-5
l1_lambda: 1e-5 → 5e-6
l2_lambda: 1e-4 → 5e-5
hidden_size: 40 → 48
```

### If Unstable (Loss Spikes)
```python
learning_rate: 1e-5 → 5e-6
lr_warmup_epochs: 5 → 10
max_grad_norm: 0.5 → 0.3
```

---

## ML Engineering Principles

### From `improving_convergence.md`
✓ Elastic Net Regularization (L1/L2)
✓ Batch Normalization (no Dropout)
✓ Adam Optimizer (default betas)
✓ Low Learning Rate (financial data)
✓ Gradient Clipping (prevent explosions)
✓ Early Stopping (patience-based)

### From `LSTM_NUANCES.md`
✓ Architecture: 40 units, 1 layer
✓ BatchNorm before hidden layers
✓ No Dropout (BatchNorm replaces)
✓ No activation after LSTM

### From `correlation_usage.md`
✓ Feature Selection: 38 features
✓ Sequence Length: 20 bars
✓ Regularization: Prevent noise

---

## Document Navigation

**I want to...**

**Understand the problem quickly**
→ Read: `TUNING_QUICK_REFERENCE.md`

**Understand why each parameter changed**
→ Read: `TUNING_RATIONALE.md`

**Get a complete overview**
→ Read: `LSTM_TUNING_SUMMARY.md`

**Prepare for training**
→ Read: `TUNING_CHECKLIST.md`

**Understand the strategy**
→ Read: `LSTM_TRAINING_TUNING.md`

**Learn about LSTM architecture**
→ Read: `LSTM_NUANCES.md`

**Learn about regularization**
→ Read: `improving_convergence.md`

**Learn about feature selection**
→ Read: `correlation_usage.md`

---

## Key Contacts

**Questions about tuning?**
- See: `TUNING_RATIONALE.md` (detailed explanations)
- See: `TUNING_QUICK_REFERENCE.md` (quick answers)

**Questions about architecture?**
- See: `LSTM_NUANCES.md` (architecture principles)

**Questions about regularization?**
- See: `improving_convergence.md` (regularization techniques)

**Questions about features?**
- See: `correlation_usage.md` (feature selection principles)

---

## Status

✓ **READY FOR TRAINING**

All tuning parameters have been applied and documented.
Ready to run training with improved convergence and generalization.

---

## Timeline

- **2025-12-08:** Tuning analysis and implementation
- **2025-12-08:** Documentation created
- **Next:** Run training with tuned parameters
- **Expected:** Val Acc ≥ 65% by epoch 30

---

## Summary

**9 parameters tuned** following senior ML engineering principles from your resources.
**5 documentation files** created for reference and monitoring.
**Target:** 65%+ accuracy with <2% train/val gap.

Ready to nudge training toward your goals!

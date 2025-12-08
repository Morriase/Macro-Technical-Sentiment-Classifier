# LSTM Tuning - Pre-Training Checklist

## ✓ Tuning Applied
- [x] Sequence length: 40 → 20
- [x] Learning rate: 3e-5 → 1e-5
- [x] Batch size: 10,000 → 2,000
- [x] L1 regularization: 1e-7 → 1e-5
- [x] L2 regularization: 1e-5 → 1e-4
- [x] Label smoothing: 0.1 → 0.15
- [x] Gradient clipping: 1.0 → 0.5
- [x] Warmup epochs: 3 → 5
- [x] Early stopping patience: 20 → 30

## ✓ Documentation Created
- [x] LSTM_TRAINING_TUNING.md (detailed strategy)
- [x] TUNING_RATIONALE.md (rationale for each change)
- [x] TUNING_QUICK_REFERENCE.md (quick reference)
- [x] LSTM_TUNING_SUMMARY.md (complete summary)
- [x] TUNING_CHECKLIST.md (this file)

## ✓ Code Changes
- [x] Updated src/models/hybrid_ensemble.py
- [x] LSTM parameters modified in HybridEnsemble.__init__()
- [x] All 9 parameters applied correctly
- [x] Comments added explaining each change

## Before Running Training

### Verify Implementation
```bash
# Check that parameters are applied
grep -A 50 "lstm_params = lstm_params or" src/models/hybrid_ensemble.py
```

Expected output should show:
- sequence_length: 20
- learning_rate: 1e-5
- batch_size: 2000
- l1_lambda: 1e-5
- l2_lambda: 1e-4
- label_smoothing: 0.15
- max_grad_norm: 0.5
- lr_warmup_epochs: 5
- early_stopping_patience: 30

### Verify FRED Integration
```bash
# Check that FRED features are working
grep -n "rate_differential\|vix\|yield_curve\|dxy_index\|oil_price" main.py
```

Expected: 5 FRED macro features in feature schema

### Verify Feature Count
```bash
# Check that we have 38 features
grep "n_features" models/EUR_USD_feature_schema.json
```

Expected: `"n_features": 38`

## During Training - Monitoring

### Epoch 1-5 (Warmup Phase)
- [ ] Loss decreasing gradually
- [ ] Accuracy around 50% (random baseline)
- [ ] No loss spikes
- [ ] Learning rate ramping up

### Epoch 5-15 (Rapid Improvement)
- [ ] Loss decreasing rapidly
- [ ] Accuracy improving: 50% → 60%
- [ ] Train/Val curves parallel
- [ ] No divergence

### Epoch 15-30 (Convergence)
- [ ] Loss plateauing
- [ ] Accuracy: 60% → 65%+
- [ ] Train/Val gap <2%
- [ ] Smooth curves

### Epoch 30+ (Plateau)
- [ ] Loss stable
- [ ] Accuracy stable
- [ ] Early stopping triggers
- [ ] Final Val Acc ≥ 65%

## Success Criteria

### Must Have ✓
- [ ] Val Accuracy ≥ 65% by epoch 30
- [ ] Train/Val gap ≤ 2%
- [ ] Smooth loss curves (no spikes)
- [ ] No divergence between train/val

### Nice to Have
- [ ] Train Accuracy 65-70%
- [ ] Val Accuracy 63-68%
- [ ] Early stopping around epoch 30-50
- [ ] Consistent improvement across folds

## If Not Meeting Targets

### Val Acc Still <60% (Overfitting)
- [ ] Reduce hidden_size: 40 → 32
- [ ] Increase L2: 1e-4 → 5e-4
- [ ] Reduce sequence_length: 20 → 15
- [ ] Increase label_smoothing: 0.15 → 0.2

### Both Train/Val <55% (Underfitting)
- [ ] Increase learning_rate: 1e-5 → 2e-5
- [ ] Reduce L1: 1e-5 → 5e-6
- [ ] Reduce L2: 1e-4 → 5e-5
- [ ] Increase hidden_size: 40 → 48

### Loss Spikes (Instability)
- [ ] Reduce learning_rate: 1e-5 → 5e-6
- [ ] Increase warmup_epochs: 5 → 10
- [ ] Reduce max_grad_norm: 0.5 → 0.3
- [ ] Increase label_smoothing: 0.15 → 0.2

## Post-Training

### Validation
- [ ] Save best model
- [ ] Record final metrics
- [ ] Compare with baseline
- [ ] Check feature importance

### Deployment
- [ ] Update feature schema if needed
- [ ] Test inference server
- [ ] Validate predictions
- [ ] Deploy to production

## Documentation References

| Document | Purpose |
|----------|---------|
| LSTM_TRAINING_TUNING.md | Detailed tuning strategy |
| TUNING_RATIONALE.md | Rationale for each parameter |
| TUNING_QUICK_REFERENCE.md | Quick reference card |
| LSTM_TUNING_SUMMARY.md | Complete summary |
| LSTM_NUANCES.md | Architecture principles |
| improving_convergence.md | Regularization techniques |
| correlation_usage.md | Feature selection |

## Key Contacts

**Questions about tuning?**
- See: TUNING_RATIONALE.md (detailed explanations)
- See: TUNING_QUICK_REFERENCE.md (quick answers)

**Questions about architecture?**
- See: LSTM_NUANCES.md (architecture principles)

**Questions about regularization?**
- See: improving_convergence.md (regularization techniques)

---

## Status: ✓ READY FOR TRAINING

All tuning parameters have been applied and documented.
Ready to run training with improved convergence and generalization.

**Last Updated:** 2025-12-08
**Applied By:** Senior ML Engineer (Kiro)

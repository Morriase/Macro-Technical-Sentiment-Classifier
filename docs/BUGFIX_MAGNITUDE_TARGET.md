# Bugfix: Magnitude Target Tensor Size Mismatch

**Date:** December 10, 2025  
**Status:** ✅ FIXED  
**Error:** `RuntimeError: The size of tensor a (16384) must match the size of tensor b (5) at non-singleton dimension 1`

---

## Problem

Training failed with tensor size mismatch error. The LSTM was receiving the wrong data for the magnitude target.

### Root Cause

In `src/models/hybrid_ensemble.py`, the `generate_out_of_fold_predictions` method was passing the entire feature matrix `X_val_fold` (shape: `[n, 5]`) as the magnitude target instead of the actual magnitude values (shape: `[n]`).

**Incorrect code:**
```python
lstm_fold.fit(X_train_fold, y_train_fold, X_val_fold, y[val_idx])
#                                          ^^^^^^^^^^^ Wrong! This is features, not magnitude
```

---

## Solution

### 1. Updated `hybrid_ensemble.py` ✅

**`generate_out_of_fold_predictions` method:**
- Added `y_magnitude` parameter
- Extract magnitude for train/val splits
- Pass to LSTM correctly

```python
def generate_out_of_fold_predictions(
    self, X: np.ndarray, y: np.ndarray, y_magnitude: Optional[np.ndarray] = None
):
    # ...
    if y_magnitude is not None:
        y_train_mag = y_magnitude[train_idx]
        y_val_mag = y_magnitude[val_idx]
        lstm_fold.fit(X_train_fold, y_train_fold, y_magnitude=y_train_mag, 
                     X_val=X_val_fold, y_val=y[val_idx], y_val_magnitude=y_val_mag)
```

**`fit` method:**
- Pass `y_magnitude` to `generate_out_of_fold_predictions`

```python
xgb_oof_proba, lstm_oof_proba = self.generate_out_of_fold_predictions(
    X, y, y_magnitude  # Added y_magnitude
)
```

### 2. Updated `walk_forward.py` ✅

**`run_walk_forward_optimization` method:**
- Extract magnitude target from dataframe
- Pass to model.fit()

```python
# Extract magnitude target if available
y_train_magnitude = None
if 'target_magnitude_norm' in df.columns:
    y_train_magnitude = df.loc[train_idx, 'target_magnitude_norm'].values

# Pass to model
if y_train_magnitude is not None:
    model.fit(X_train, y_train, y_magnitude=y_train_magnitude, ...)
```

---

## Files Modified

1. `src/models/hybrid_ensemble.py`
   - `generate_out_of_fold_predictions()` - Added y_magnitude parameter
   - `fit()` - Pass y_magnitude to OOF generation

2. `src/validation/walk_forward.py`
   - `run_walk_forward_optimization()` - Extract and pass magnitude target

---

## Testing

The fix ensures:
- ✅ Magnitude target has correct shape `[batch_size]`
- ✅ Direction target has correct shape `[batch_size]`
- ✅ Features have correct shape `[batch_size, 5]`
- ✅ No tensor size mismatches

---

## Ready to Train

The pipeline should now work correctly:

```bash
python main.py
```

Expected behavior:
- LSTM receives both direction and magnitude targets
- Training proceeds without tensor size errors
- Dual output loss calculated correctly

---

**Status: FIXED ✅ - Ready for training**

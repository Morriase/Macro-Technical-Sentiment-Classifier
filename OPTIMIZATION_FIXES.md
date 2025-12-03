# Optimization Fixes for Memory Issues

## Problem
The training process was crashing with an "Out of Memory" error on Kaggle during the hyperparameter optimization phase. The logs indicated that the crash occurred immediately after starting the Optuna study.

## Root Cause Analysis
1. **Parallel Optuna Trials**: `OPTUNA_CONFIG["n_jobs"]` was set to `-1` (use all cores). On Kaggle, this caused multiple trials to run in parallel. Each trial loads data and creates models, leading to rapid memory exhaustion.
2. **DataLoader Workers**: `GPU_CONFIG["num_workers"]` was set to 4 (on Kaggle). Multiple worker processes for data loading consume additional RAM.
3. **Large Batch Size**: `ENSEMBLE_CONFIG` specified a batch size of 256 for the LSTM model, which contributes to memory pressure.

## Fixes Applied
Modified `src/config.py` to optimize for memory stability over speed:

1. **Disable Parallel Optimization**:
   - Set `OPTUNA_CONFIG["n_jobs"] = 1`.
   - This ensures trials run sequentially, significantly reducing peak memory usage.

2. **Disable DataLoader Workers**:
   - Set `GPU_CONFIG["num_workers"] = 0`.
   - This runs data loading in the main process, avoiding the overhead of worker processes.

3. **Reduce Batch Size**:
   - Reduced `ENSEMBLE_CONFIG["base_learners"]["lstm"]["batch_size"]` from 256 to 128.
   - This reduces the memory footprint during LSTM training.

## Verification
These changes ensure that the training process stays within the memory limits of the Kaggle environment (typically 13-16GB RAM). The trade-off is that optimization will take longer (linear time vs parallel), but it will complete without crashing.

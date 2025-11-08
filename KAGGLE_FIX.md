# Kaggle Environment Fix

## Problem
Kaggle's default transformers version has issues loading FinBERT.

## Solution
Run this at the start of your Kaggle notebook:

```python
!pip install --upgrade transformers huggingface-hub -q
```

Then restart the kernel and run your training:

```python
!python main.py
```

This upgrades transformers to a version that properly handles the FinBERT model files.

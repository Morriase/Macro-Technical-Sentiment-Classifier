# Kaggle Setup Guide

This guide helps you run the Macro-Technical Sentiment Classifier on Kaggle.

## Quick Start on Kaggle

### Step 1: Create New Notebook
1. Go to Kaggle.com and create a new notebook
2. Settings → Internet: **ON** (required for git clone)
3. Settings → GPU: **GPU T4 x2** (required for CUDA training)
4. Settings → Environment: Python

### Step 2: Install TA-Lib FIRST (Critical!)
```python
# TA-Lib must be installed before cloning repo
!pip install TA-Lib
```

### Step 3: Clone Repository
```python
!cd /kaggle/working && git clone https://github.com/Morriase/Macro-Technical-Sentiment-Classifier.git
%cd /kaggle/working/Macro-Technical-Sentiment-Classifier
```

### Step 4: Install Python Dependencies
```python
# Install required packages (most are pre-installed on Kaggle)
!pip install -q finnhub-python python-dotenv
```

### Step 5: Add Dataset as Input
1. In your Kaggle notebook, click "+ Add Data" in the right panel
2. Search for your dataset: `macros-and-ohlc` (or the name you uploaded)
3. Add it - it will be mounted at `/kaggle/input/macros-and-ohlc/`

### Step 6: Run Training
```python
!cd /kaggle/working/Macro-Technical-Sentiment-Classifier && python main.py
```

### Step 7: Monitor Progress
The training will output logs showing:
- Data loading (80k candles, 84 events)
- Feature engineering
- Walk-forward validation progress
- Model performance metrics

---

## One-Command Setup (Copy & Paste)

```python
# Run this in a single Kaggle cell
!pip install -q TA-Lib
!cd /kaggle/working && git clone https://github.com/Morriase/Macro-Technical-Sentiment-Classifier.git
!cd /kaggle/working/Macro-Technical-Sentiment-Classifier && pip install -q finnhub-python python-dotenv
!cd /kaggle/working/Macro-Technical-Sentiment-Classifier && python main.py
from src.models.lstm_model import LSTMSequenceModel
from src.models.hybrid_ensemble import HybridEnsemble

print("✓ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"TA-Lib version: {talib.__version__}")
```

### Step 7: Run the Pipeline
```python
from main import ForexClassifierPipeline

# Initialize pipeline
pipeline = ForexClassifierPipeline(
    currency_pair="EUR_USD",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

# Run full workflow
pipeline.run()
```

## Kaggle-Specific Notes

### Memory Management
Kaggle notebooks have ~13GB RAM. For large datasets:
```python
# In src/config.py, adjust:
BACKTEST_PERIOD = {
    "start_date": "2023-01-01",  # Shorter period
    "end_date": "2024-01-01",
}
```

### CPU-Only Training
The code automatically detects CPU and adjusts:
- LSTM uses CPU tensors
- XGBoost uses `n_jobs=-1` (all cores)
- Training time: ~15-30 minutes for 1 year of data

### Data Persistence
Save processed data to avoid re-fetching:
```python
# After data acquisition
import joblib
joblib.dump(df, '/kaggle/working/data_cache.pkl')
```

### API Rate Limits
- **OANDA**: 120 requests/second (more than enough)
- **Finnhub**: 60 calls/minute (free tier)
- Add delays if needed:
```python
import time
time.sleep(1)  # 1 second between API calls
```

## Common Issues

### Issue 1: TA-Lib Installation Fails
**Solution**: Kaggle has TA-Lib pre-installed in some environments
```python
try:
    import talib
except ImportError:
    !pip install TA-Lib
```

### Issue 2: Transformers Model Download
First run downloads ~500MB FinBERT model:
```python
# Cache model in Kaggle dataset for reuse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert",
    cache_dir="/kaggle/working/model_cache"
)
```

### Issue 3: Out of Memory
Reduce batch sizes in `src/config.py`:
```python
ENSEMBLE_CONFIG = {
    "base_learners": {
        "lstm": {
            "batch_size": 32,  # Reduce from 64
            "sequence_length": 15,  # Reduce from 22
        }
    }
}
```

### Issue 4: Slow Training
Enable early stopping:
```python
ENSEMBLE_CONFIG = {
    "base_learners": {
        "lstm": {
            "epochs": 50,  # Reduce from 100
            "early_stopping_patience": 5,  # Stop if no improvement
        }
    }
}
```

## Running Tests

### Test Individual Components
```python
# Test LSTM
!python src/models/lstm_model.py

# Test Hybrid Ensemble
!python src/models/hybrid_ensemble.py

# Test Data Acquisition
!python src/data_acquisition/fx_data.py
```

### Test Full Pipeline
```python
!python main.py
```

## Saving Results

```python
# Save trained model
ensemble.save_model('/kaggle/working/forex_model')

# Save predictions
predictions_df.to_csv('/kaggle/working/predictions.csv', index=False)

# Download from Kaggle Output tab
```

## Performance Tips

1. **Use Kaggle Datasets**: Upload pre-processed data as a dataset
2. **Version Control**: Save checkpoints every N epochs
3. **Logging**: Redirect logs to file for later review
4. **Parallel Processing**: Maximize CPU usage with `n_jobs=-1`

## Example Kaggle Notebook Structure

```python
# Cell 1: Setup
!git clone https://github.com/Morriase/Macro-Technical-Sentiment-Classifier.git
%cd Macro-Technical-Sentiment-Classifier
!pip install -q -r requirements.txt

# Cell 2: API Keys (keep private!)
import os
os.environ['OANDA_API_KEY'] = 'xxx'
os.environ['FINNHUB_API_KEY'] = 'xxx'

# Cell 3: NLTK Data
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Cell 4: Run Pipeline
from main import ForexClassifierPipeline
pipeline = ForexClassifierPipeline("EUR_USD", "2023-01-01", "2024-01-01")
results = pipeline.run()

# Cell 5: Analyze Results
import matplotlib.pyplot as plt
plt.plot(results['equity_curve'])
plt.title('Equity Curve')
plt.show()
```

## Getting API Keys

### OANDA (FX Data)
1. Sign up: https://www.oanda.com/
2. Create practice account (free)
3. Generate API token
4. Note your Account ID

### Finnhub (Economic Calendar)
1. Sign up: https://finnhub.io/
2. Free tier: 60 calls/minute
3. Copy API key from dashboard

## Resources

- **GitHub Repo**: https://github.com/Morriase/Macro-Technical-Sentiment-Classifier
- **OANDA API Docs**: https://developer.oanda.com/
- **Finnhub API Docs**: https://finnhub.io/docs/api
- **TA-Lib Indicators**: https://ta-lib.github.io/ta-lib-python/

## Support

For issues, create a GitHub issue or check the documentation files:
- `README.md` - Project overview
- `ARCHITECTURE.md` - Model architecture details
- `SETUP.md` - Local development setup
- `IMPLEMENTATION_SUMMARY.md` - Implementation details

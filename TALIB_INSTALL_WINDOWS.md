# TA-Lib Installation Instructions for Windows

## Problem
TA-Lib (Technical Analysis Library) requires C++ compilation on Windows, which pip cannot handle automatically.

## Solution: Install Pre-compiled Wheel

### Step 1: Download the Pre-compiled Wheel
Visit the unofficial Windows binaries repository:
**https://github.com/cgohlke/talib-build/releases**

Download the wheel file matching your Python version and architecture:
- For Python 3.13 64-bit: `TA_Lib-0.4.28-cp313-cp313-win_amd64.whl`
- For Python 3.12 64-bit: `TA_Lib-0.4.28-cp312-cp312-win_amd64.whl`
- For Python 3.11 64-bit: `TA_Lib-0.4.28-cp311-cp311-win_amd64.whl`

### Step 2: Install the Wheel
```powershell
# Activate your virtual environment
.\venv\Scripts\Activate.ps1

# Install the downloaded wheel (replace with your actual filename)
pip install "C:\Users\Morris\Downloads\TA_Lib-0.4.28-cp313-cp313-win_amd64.whl"
```

### Step 3: Verify Installation
```powershell
python -c "import talib; print(talib.__version__)"
```

Expected output: `0.4.0` or similar

## Alternative: Using conda
If you prefer conda (requires Anaconda/Miniconda):
```bash
conda install -c conda-forge ta-lib
```

## What is TA-Lib?
TA-Lib provides 150+ technical analysis indicators including:
- Moving Averages (EMA, SMA, WMA)
- Momentum Indicators (RSI, Stochastic, MACD)
- Volatility Indicators (ATR, Bollinger Bands)
- Volume Indicators (OBV, AD)
- Pattern Recognition (Candlestick patterns)

## Used in This Project
The `src/feature_engineering/technical_features.py` module relies on TA-Lib for:
- EMA (50, 100, 200 periods)
- RSI (14 period)
- ATR (14 period)
- MACD (12, 26, 9)
- Bollinger Bands (20 period)
- Stochastic Oscillator

## Troubleshooting

### Error: "Microsoft Visual C++ 14.0 or greater is required"
This means you're trying to compile from source. Use the pre-compiled wheel instead (Step 1 above).

### Error: "Could not find a version that satisfies the requirement"
Make sure you downloaded the wheel matching your exact Python version:
```powershell
python --version  # Check your Python version
```

### Error: Module 'talib' has no attribute 'EMA'
This means you installed the wrong package. Uninstall and reinstall:
```powershell
pip uninstall TA-Lib ta-lib
pip install "path\to\TA_Lib-0.4.28-cp313-cp313-win_amd64.whl"
```

## Quick Check Your Python Version
```powershell
python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor} ({sys.maxsize.bit_length() + 1}-bit)')"
```

Expected output examples:
- `Python 3.13 (64-bit)` → Use cp313-win_amd64
- `Python 3.12 (64-bit)` → Use cp312-win_amd64
- `Python 3.11 (64-bit)` → Use cp311-win_amd64

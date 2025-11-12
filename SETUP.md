# Setup Guide - Macro-Technical Sentiment Forex Classifier

Complete installation and setup instructions for Windows, Linux, and macOS.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Python Environment Setup](#python-environment-setup)
3. [TA-Lib Installation](#ta-lib-installation)
4. [Package Installation](#package-installation)
5. [API Keys Configuration](#api-keys-configuration)
6. [Verify Installation](#verify-installation)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.14+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB free space
- **Python**: 3.8 or higher
- **GPU**: Optional (CUDA-compatible for faster training)

### Recommended Setup
- **RAM**: 16GB+
- **CPU**: 8+ cores
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: SSD with 50GB+ free space

---

## Python Environment Setup

### 1. Install Python 3.8+

**Windows:**
Download from [python.org](https://www.python.org/downloads/)

**Linux:**
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev
```

**macOS:**
```bash
brew install python@3.9
```

### 2. Create Virtual Environment

```bash
# Navigate to project directory
cd "Macro-Technical Sentiment Classifier"

# Create virtual environment
python -m venv venv

# Activate environment
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (CMD)
.\venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

---

## TA-Lib Installation

TA-Lib requires system-level libraries before the Python wrapper can be installed.

### Windows

**Method 1: Pre-built Wheel (Easiest)**

1. Download the appropriate wheel for your Python version from:
   https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

   Example for Python 3.9, 64-bit:
   `TA_Lib‑0.4.28‑cp39‑cp39‑win_amd64.whl`

2. Install the wheel:
   ```powershell
   pip install TA_Lib‑0.4.28‑cp39‑cp39‑win_amd64.whl
   ```

**Method 2: Build from Source (Advanced)**

1. Download and install Visual Studio Build Tools
2. Download TA-Lib C library from http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip
3. Extract to `C:\ta-lib`
4. Set environment variables:
   ```powershell
   $env:INCLUDE="C:\ta-lib\c\include"
   $env:LIB="C:\ta-lib\c\lib"
   ```
5. Install Python wrapper:
   ```powershell
   pip install TA-Lib
   ```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential wget

# Download and compile TA-Lib
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Update library cache
sudo ldconfig

# Install Python wrapper
pip install TA-Lib
```

### macOS

```bash
# Install TA-Lib via Homebrew
brew install ta-lib

# Install Python wrapper
pip install TA-Lib
```

### Verify TA-Lib Installation

```python
python -c "import talib; print(talib.__version__)"
```

Expected output: `0.4.0-dev` or similar

---

## Package Installation

With TA-Lib installed, install all other dependencies:

```bash
# Ensure virtual environment is activated
# Install all requirements
pip install -r requirements.txt
```

This will install:
- PyTorch 2.0+
- XGBoost 2.0+
- Transformers (HuggingFace)
- Scikit-learn
- Pandas, NumPy
- Optuna
- And all other dependencies

### GPU Support (Optional)

If you have an NVIDIA GPU and want to accelerate training:

```bash
# Uninstall CPU-only PyTorch
pip uninstall torch

# Install PyTorch with CUDA support
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU support:
```python
python -c "import torch; print(torch.cuda.is_available())"
```

---

## API Keys Configuration

### 1. Create .env File

Copy the example file:
```bash
cp .env.example .env
```

### 2. Obtain API Keys

**OANDA (FX Price Data)**
1. Visit: https://www.oanda.com/
2. Sign up for a practice account (free)
3. Navigate to "Manage API Access"
4. Generate API token
5. Copy token and account ID

**Finnhub (Economic Calendar)**
1. Visit: https://finnhub.io/
2. Sign up for free account
3. Go to Dashboard → API Keys
4. Copy API key

**Trading Economics (Optional)**
1. Visit: https://tradingeconomics.com/
2. Sign up for API access
3. Choose free tier
4. Copy API key

### 3. Edit .env File

Open `.env` and add your keys:

```env
OANDA_API_KEY=abc123yourkey456
OANDA_ACCOUNT_ID=123-456-7890123-001
OANDA_ENVIRONMENT=practice

FINNHUB_API_KEY=xyz789yourkey
TRADING_ECONOMICS_API_KEY=te_key_here
```

**Security Note**: Never commit `.env` to version control!

---

## Verify Installation

Run the verification script:

```python
# verification_test.py
import sys
import torch
import xgboost as xgb
import talib
import transformers
import pandas as pd
import numpy as np
from sklearn import __version__ as sklearn_version

print("="*60)
print("Installation Verification")
print("="*60)

# Python version
print(f"Python: {sys.version}")

# Core packages
print(f"PyTorch: {torch.__version__}")
print(f"  - CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA Version: {torch.version.cuda}")
    print(f"  - GPU: {torch.cuda.get_device_name(0)}")

print(f"XGBoost: {xgb.__version__}")
print(f"TA-Lib: {talib.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Scikit-learn: {sklearn_version}")

# Test TA-Lib
try:
    close = np.random.random(100)
    sma = talib.SMA(close, timeperiod=10)
    print("TA-Lib: ✓ Working")
except Exception as e:
    print(f"TA-Lib: ✗ Error - {e}")

# Test environment variables
import os
from dotenv import load_dotenv
load_dotenv()

oanda_key = os.getenv("OANDA_API_KEY")
finnhub_key = os.getenv("FINNHUB_API_KEY")

print(f"\nEnvironment Variables:")
print(f"  OANDA_API_KEY: {'✓ Set' if oanda_key else '✗ Not Set'}")
print(f"  FINNHUB_API_KEY: {'✓ Set' if finnhub_key else '✗ Not Set'}")

print("="*60)
print("Verification Complete!")
print("="*60)
```

Run verification:
```bash
python verification_test.py
```

---

## Troubleshooting

### Common Issues

#### 1. TA-Lib Import Error

**Error**: `ImportError: DLL load failed while importing _ta_lib`

**Solution (Windows)**:
- Ensure you downloaded the correct wheel for your Python version
- Try reinstalling with the pre-built wheel
- Check that Visual C++ Redistributable is installed

**Solution (Linux)**:
```bash
sudo ldconfig
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

#### 2. OANDA API Connection Error

**Error**: `401 Unauthorized`

**Solution**:
- Verify API key is correct in `.env`
- Check that environment is set to `practice` not `live`
- Ensure account is active

#### 3. Out of Memory Error

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size in `src/config.py`
- Reduce MLP hidden layer sizes
- Use CPU instead of GPU for meta-learner

#### 4. PyTorch CUDA Not Available

**Error**: `torch.cuda.is_available()` returns `False`

**Solution**:
- Verify NVIDIA GPU is installed
- Update GPU drivers
- Reinstall PyTorch with correct CUDA version
- Check CUDA toolkit installation

#### 5. Module Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Add src to PYTHONPATH
# Windows
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

# Linux/Mac
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

Or run from project root:
```bash
python -m src.main
```

---

## Next Steps

After successful installation:

1. **Run Example Pipeline**:
   ```bash
   python main.py
   ```

2. **Explore Notebooks**:
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

3. **Configure Strategy**:
   - Edit `src/config.py` for your preferences
   - Adjust risk management parameters
   - Customize feature selection

4. **Read Documentation**:
   - Review `README.md` for usage
   - Check code comments for details
   - Review research documents in `resources/`

---

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Search existing GitHub issues
3. Create a new issue with:
   - Error message
   - Full traceback
   - Operating system
   - Python version
   - Output of `pip list`

---

## Performance Tips

### Speed Up Training

1. **Use GPU**: 3-5x faster for meta-learner
2. **Reduce Data**: Use fewer years for initial testing
3. **Parallel Processing**: Set `n_jobs=-1` in XGBoost/RF
4. **Reduce Optuna Trials**: Lower from 100 to 20 for testing

### Reduce Memory Usage

1. **Lower Batch Size**: Reduce from 64 to 32 or 16
2. **Downsample Data**: Use H1 instead of M5 for initial tests
3. **Feature Selection**: Remove low-importance features
4. **Use Sparse Matrices**: For high-dimensional features

---

**Setup complete! You're ready to start trading with AI. 🚀**

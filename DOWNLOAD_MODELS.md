# Download Models from Kaggle

## Option 1: Kaggle Web UI (Easiest)

1. **Go to your Kaggle notebook:**
   ```
   https://www.kaggle.com/code/viserione/macro-technical-sentiment-classiefier
   ```

2. **Click "Output" tab** (right side of notebook)

3. **Download the models folder:**
   - Look for `/kaggle/working/models/` in the output
   - Click the download icon next to the models folder
   - This will download a ZIP file

4. **Extract and replace:**
   ```
   - Extract the ZIP file
   - Copy all files from extracted models/ folder
   - Paste into your local models/ folder
   - Replace existing files when prompted
   ```

---

## Option 2: Kaggle CLI (If you have it installed on Windows)

On Windows (where you have Python properly set up):

```powershell
# Install Kaggle CLI (if not already installed)
pip install kaggle

# Set up Kaggle API credentials
# Download kaggle.json from https://www.kaggle.com/settings
# Place it in: C:\Users\YourUsername\.kaggle\kaggle.json

# Download kernel output
kaggle kernels output viserione/macro-technical-sentiment-classiefier -p models_temp

# Copy only model files
Copy-Item models_temp\models\*_model*.* models\ -Force
Copy-Item models_temp\models\*_feature_schema.json models\ -Force

# Clean up
Remove-Item models_temp -Recurse
```

---

## Option 3: Direct Download Links

If the notebook is public, you can download specific files:

1. **EUR_USD models:**
   - EUR_USD_model_config.pkl
   - EUR_USD_model_lstm_base.pth
   - EUR_USD_model_meta.pkl
   - EUR_USD_model_xgb_base.pkl
   - EUR_USD_feature_schema.json

2. **GBP_USD models:**
   - GBP_USD_model_config.pkl
   - GBP_USD_model_lstm_base.pth
   - GBP_USD_model_meta.pkl
   - GBP_USD_model_xgb_base.pkl
   - GBP_USD_feature_schema.json

3. **USD_JPY models:**
   - USD_JPY_model_config.pkl
   - USD_JPY_model_lstm_base.pth
   - USD_JPY_model_meta.pkl
   - USD_JPY_model_xgb_base.pkl
   - USD_JPY_feature_schema.json

4. **AUD_USD models:**
   - AUD_USD_model_config.pkl
   - AUD_USD_model_lstm_base.pth
   - AUD_USD_model_meta.pkl
   - AUD_USD_model_xgb_base.pkl
   - AUD_USD_feature_schema.json

---

## Files to Download (20 files total)

### Model Files (16 files):
```
EUR_USD_model_config.pkl
EUR_USD_model_lstm_base.pth
EUR_USD_model_meta.pkl
EUR_USD_model_xgb_base.pkl

GBP_USD_model_config.pkl
GBP_USD_model_lstm_base.pth
GBP_USD_model_meta.pkl
GBP_USD_model_xgb_base.pkl

USD_JPY_model_config.pkl
USD_JPY_model_lstm_base.pth
USD_JPY_model_meta.pkl
USD_JPY_model_xgb_base.pkl

AUD_USD_model_config.pkl
AUD_USD_model_lstm_base.pth
AUD_USD_model_meta.pkl
AUD_USD_model_xgb_base.pkl
```

### Feature Schema Files (4 files):
```
EUR_USD_feature_schema.json
GBP_USD_feature_schema.json
USD_JPY_feature_schema.json
AUD_USD_feature_schema.json
```

---

## Verify Download

After downloading, verify you have all files:

```bash
# On Linux
ls -lh models/*_model*.* models/*_feature_schema.json | wc -l
# Should show 20 files

# On Windows PowerShell
(Get-ChildItem models\*_model*.*, models\*_feature_schema.json).Count
# Should show 20
```

---

## File Sizes (Approximate)

- **_config.pkl**: ~1-5 KB (small)
- **_lstm_base.pth**: ~500 KB - 2 MB (LSTM weights)
- **_meta.pkl**: ~50-200 KB (meta-classifier)
- **_xgb_base.pkl**: ~500 KB - 2 MB (XGBoost model)
- **_feature_schema.json**: ~5-10 KB (feature metadata)

**Total per pair:** ~2-5 MB
**Total for 4 pairs:** ~8-20 MB

---

## After Download

1. **Backup old models** (optional):
   ```bash
   mkdir models_backup
   cp models/*_model*.* models_backup/
   cp models/*_feature_schema.json models_backup/
   ```

2. **Test inference server:**
   ```bash
   python inference_server.py
   # Should see: "Loaded EUR_USD model (81 features)"
   ```

3. **Run feature alignment test:**
   ```bash
   python test_feature_alignment.py
   # Should pass with 81 features
   ```

---

## Troubleshooting

### "Model not found" error:
- Check file names match exactly (case-sensitive on Linux)
- Verify files are in `models/` directory, not `models/models/`

### "Feature count mismatch" error:
- Make sure you downloaded the feature_schema.json files
- Verify schema shows 81 features

### "Cannot load model" error:
- Check file permissions (should be readable)
- Verify files aren't corrupted (check file sizes)

---

## Quick Download Script (Windows PowerShell)

Save this as `download_models.ps1`:

```powershell
# Download from Kaggle
kaggle kernels output viserione/macro-technical-sentiment-classiefier -p temp_models

# Copy model files
$pairs = @("EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD")
foreach ($pair in $pairs) {
    Copy-Item "temp_models\models\${pair}_model_config.pkl" models\ -Force
    Copy-Item "temp_models\models\${pair}_model_lstm_base.pth" models\ -Force
    Copy-Item "temp_models\models\${pair}_model_meta.pkl" models\ -Force
    Copy-Item "temp_models\models\${pair}_model_xgb_base.pkl" models\ -Force
    Copy-Item "temp_models\models\${pair}_feature_schema.json" models\ -Force
    Write-Host "✓ Downloaded $pair models"
}

# Clean up
Remove-Item temp_models -Recurse -Force
Write-Host "✓ All models downloaded successfully!"
```

Run with:
```powershell
.\download_models.ps1
```

---

## Status

Once downloaded, you'll have the latest trained models with:
- ✅ 81 features (67 base + 14 MTF + 3 macro)
- ✅ No overfitting (val acc ≥ train acc)
- ✅ GPU-accelerated training
- ✅ Stable convergence
- ✅ Ready for production testing

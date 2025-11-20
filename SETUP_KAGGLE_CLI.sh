#!/bin/bash
# Setup Kaggle CLI on Linux

echo "=========================================="
echo "Kaggle CLI Setup"
echo "=========================================="
echo ""

# Step 1: Create .kaggle directory
mkdir -p ~/.kaggle
echo "✓ Created ~/.kaggle directory"

# Step 2: Instructions for getting API token
echo ""
echo "To download your Kaggle API token:"
echo "1. Go to: https://www.kaggle.com/settings"
echo "2. Scroll down to 'API' section"
echo "3. Click 'Create New Token'"
echo "4. This will download 'kaggle.json'"
echo ""
echo "Then run:"
echo "  mv ~/Downloads/kaggle.json ~/.kaggle/"
echo "  chmod 600 ~/.kaggle/kaggle.json"
echo ""
echo "After that, run this script again to download models."
echo ""

# Check if kaggle.json exists
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✓ kaggle.json found!"
    chmod 600 ~/.kaggle/kaggle.json
    echo "✓ Set correct permissions"
    
    # Add kaggle to PATH
    export PATH="$HOME/.local/bin:$PATH"
    
    # Download models
    echo ""
    echo "Downloading models from Kaggle..."
    kaggle kernels output viserione/macro-technical-sentiment-classiefier -p /tmp/kaggle_models
    
    if [ $? -eq 0 ]; then
        echo "✓ Download successful!"
        
        # Copy model files to local models directory
        echo ""
        echo "Copying model files..."
        
        # Copy all model files
        cp /tmp/kaggle_models/models/*_model*.* models/ 2>/dev/null
        cp /tmp/kaggle_models/models/*_feature_schema.json models/ 2>/dev/null
        
        # Count files
        model_count=$(ls models/*_model*.* models/*_feature_schema.json 2>/dev/null | wc -l)
        echo "✓ Copied $model_count model files"
        
        # Clean up
        rm -rf /tmp/kaggle_models
        echo "✓ Cleaned up temporary files"
        
        echo ""
        echo "=========================================="
        echo "Models downloaded successfully!"
        echo "=========================================="
        echo ""
        echo "Verify with:"
        echo "  ls -lh models/*_model*.* models/*_feature_schema.json"
        
    else
        echo "✗ Download failed. Check your notebook name and permissions."
    fi
else
    echo "⚠ kaggle.json not found. Please follow the instructions above."
fi

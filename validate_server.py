#!/usr/bin/env python3
"""
Quick validation script to test if the inference server can start.
Run this BEFORE pushing to Render to catch errors fast.

Usage:
    python validate_server.py
"""

import sys
import os
from pathlib import Path

print("=" * 80)
print("VALIDATING INFERENCE SERVER SETUP")
print("=" * 80)

errors = []
warnings = []

# 1. Check Python version
print("\n1. Checking Python version...")
if sys.version_info < (3, 9):
    errors.append(
        f"Python 3.9+ required, found {sys.version_info.major}.{sys.version_info.minor}")
else:
    print(
        f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# 2. Check required files
print("\n2. Checking required files...")
required_files = [
    'inference_server.py',
    'Dockerfile',
    'start.sh',
    'render.yaml',
    'requirements_render.txt',
    'src/config.py',
    'src/feature_engineering/technical_features.py',
    'src/data_acquisition/macro_data.py',
]

for file in required_files:
    if not Path(file).exists():
        errors.append(f"Missing required file: {file}")
    else:
        print(f"   ✓ {file}")

# 3. Check models directory
print("\n3. Checking models directory...")
models_dir = Path('models')
if not models_dir.exists():
    errors.append("models/ directory not found")
else:
    model_files = list(models_dir.glob('*_model.pth_*'))
    schema_files = list(models_dir.glob('*_feature_schema.json'))

    if len(model_files) == 0:
        errors.append("No model files found in models/")
    else:
        print(f"   ✓ Found {len(model_files)} model files")

    if len(schema_files) == 0:
        errors.append("No feature schema files found in models/")
    else:
        print(f"   ✓ Found {len(schema_files)} schema files")

# 4. Test imports
print("\n4. Testing critical imports...")
try:
    import flask
    print("   ✓ flask")
except ImportError as e:
    errors.append(f"Failed to import flask: {e}")

try:
    import numpy
    print("   ✓ numpy")
except ImportError as e:
    errors.append(f"Failed to import numpy: {e}")

try:
    import pandas
    print("   ✓ pandas")
except ImportError as e:
    errors.append(f"Failed to import pandas: {e}")

try:
    import talib
    print("   ✓ talib")
except ImportError as e:
    errors.append(f"Failed to import talib: {e}")

try:
    import torch
    print("   ✓ torch")
except ImportError as e:
    errors.append(f"Failed to import torch: {e}")

try:
    import xgboost
    print("   ✓ xgboost")
except ImportError as e:
    errors.append(f"Failed to import xgboost: {e}")

# 5. Test app import
print("\n5. Testing inference_server.py import...")
try:
    sys.path.insert(0, str(Path.cwd()))
    from inference_server import app, TECH_ENGINEER, MACRO_ENGINEER
    print("   ✓ inference_server imported successfully")
    print(f"   ✓ Flask app created: {app}")
    print(f"   ✓ TECH_ENGINEER initialized: {TECH_ENGINEER}")
    print(f"   ✓ MACRO_ENGINEER initialized: {MACRO_ENGINEER}")
except Exception as e:
    errors.append(f"Failed to import inference_server: {e}")
    import traceback
    print(f"   ✗ Error: {e}")
    print(traceback.format_exc())

# 6. Test health endpoint
print("\n6. Testing Flask app routes...")
try:
    from inference_server import app
    with app.test_client() as client:
        response = client.get('/health')
        if response.status_code == 200:
            print(f"   ✓ /health endpoint works: {response.json}")
        else:
            warnings.append(f"/health returned {response.status_code}")
except Exception as e:
    warnings.append(f"Could not test /health endpoint: {e}")

# 7. Check Dockerfile
print("\n7. Checking Dockerfile...")
dockerfile = Path('Dockerfile')
if dockerfile.exists():
    content = dockerfile.read_text()
    if 'requirements_render.txt' in content:
        print("   ✓ Dockerfile references requirements_render.txt")
    else:
        warnings.append("Dockerfile doesn't reference requirements_render.txt")

    if 'start.sh' in content:
        print("   ✓ Dockerfile copies start.sh")
    else:
        errors.append("Dockerfile doesn't copy start.sh")

# 8. Check start.sh
print("\n8. Checking start.sh...")
startsh = Path('start.sh')
if startsh.exists():
    content = startsh.read_text()
    if 'gunicorn' in content:
        print("   ✓ start.sh contains gunicorn command")
    else:
        errors.append("start.sh doesn't contain gunicorn command")

    if '$PORT' in content or '${PORT}' in content:
        print("   ✓ start.sh uses PORT environment variable")
    else:
        errors.append("start.sh doesn't use PORT environment variable")

# Summary
print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

if errors:
    print(f"\n❌ ERRORS ({len(errors)}):")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")

if not errors:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nYour inference server is ready to deploy to Render.")
    print("\nNext steps:")
    print("  1. git add .")
    print("  2. git commit -m 'Your message'")
    print("  3. git push origin main")
    print("  4. Watch Render logs for deployment")
    sys.exit(0)
else:
    print("\n❌ VALIDATION FAILED!")
    print("\nFix the errors above before deploying to Render.")
    print("This will save you 15+ minutes of waiting for a failed build.")
    sys.exit(1)

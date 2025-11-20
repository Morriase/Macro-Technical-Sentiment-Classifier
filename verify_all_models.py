import joblib
import json
from pathlib import Path

pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD']

print("=" * 80)
print("VERIFYING ALL MODEL CONFIGS FOR RENDER DEPLOYMENT")
print("=" * 80)

all_good = True

for pair in pairs:
    print(f"\n{pair}:")

    # Check config exists
    config_path = f'models/{pair}_model_config.pkl'
    schema_path = f'models/{pair}_feature_schema.json'

    if not Path(config_path).exists():
        print(f"  ❌ Config not found: {config_path}")
        all_good = False
        continue

    if not Path(schema_path).exists():
        print(f"  ❌ Schema not found: {schema_path}")
        all_good = False
        continue

    # Load and verify
    config = joblib.load(config_path)
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    scaler_features = config['scaler'].n_features_in_
    metadata_features = config.get('n_features_', 'NOT FOUND')
    schema_features = schema['n_features']

    print(f"  Scaler: {scaler_features} features")
    print(f"  Metadata: {metadata_features} features")
    print(f"  Schema: {schema_features} features")
    print(f"  Trained: {schema['trained_date']}")

    if scaler_features == metadata_features == schema_features == 81:
        print(f"  ✅ PASS - All match 81 features")
    else:
        print(f"  ❌ FAIL - Mismatch detected!")
        all_good = False

print("\n" + "=" * 80)
if all_good:
    print("✅ ALL MODELS VERIFIED - READY FOR RENDER DEPLOYMENT")
    print("   No scaler errors will occur with 81-feature inputs")
else:
    print("❌ ISSUES DETECTED - DO NOT DEPLOY")
print("=" * 80)

import joblib
import json

# Check EUR_USD model config
config = joblib.load('models/EUR_USD_model_config.pkl')
print(f"Scaler expects: {config['scaler'].n_features_in_} features")
print(f"Model metadata n_features: {config.get('n_features_', 'NOT FOUND')}")

# Check feature schema
with open('models/EUR_USD_feature_schema.json', 'r') as f:
    schema = json.load(f)
print(f"Schema n_features: {schema['n_features']}")
print(f"Schema trained_date: {schema['trained_date']}")

# Verify they match
if config['scaler'].n_features_in_ == schema['n_features'] == 81:
    print("\n✅ ALL CHECKS PASSED - Scaler expects 81 features, schema has 81 features")
else:
    print("\n❌ MISMATCH DETECTED!")

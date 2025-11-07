"""
Quick test to verify LSTM shape mismatch fix in predict_proba
"""
import numpy as np
from src.models.hybrid_ensemble import HybridEnsemble

# Create dummy data
n_samples_train = 1000
n_samples_test = 500
n_features = 58

X_train = np.random.randn(n_samples_train, n_features)
y_train = np.random.randint(0, 3, n_samples_train)

X_test = np.random.randn(n_samples_test, n_features)

# Create model with default params
xgb_params = {
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 10,  # Small for quick test
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1
}

print("Creating HybridEnsemble...")
model = HybridEnsemble(xgb_params=xgb_params)

print(f"Training on {n_samples_train} samples...")
model.fit(X_train, y_train)

print(f"\nPredicting on {n_samples_test} test samples...")
try:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"✓ Predictions successful!")
    print(f"  - y_pred shape: {y_pred.shape} (expected: ({n_samples_test},))")
    print(
        f"  - y_proba shape: {y_proba.shape} (expected: ({n_samples_test}, 3))")

    if y_pred.shape[0] == n_samples_test and y_proba.shape[0] == n_samples_test:
        print("\n✅ TEST PASSED: Shape mismatch fix works!")
    else:
        print("\n❌ TEST FAILED: Shape mismatch still present!")

except ValueError as e:
    print(f"\n❌ TEST FAILED: {e}")
    print("Shape mismatch fix did not work!")

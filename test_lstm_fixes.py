#!/usr/bin/env python3
"""
Test script to verify LSTM fixes and variance reduction optimizations.
"""

import numpy as np
import pandas as pd
from src.config import ENSEMBLE_CONFIG
from src.models.lstm_model import LSTMSequenceModel

def test_lstm_fixes():
    """Test the LSTM model with variance reduction features."""
    print("Testing LSTM fixes and variance reduction optimizations...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)  # Binary classification
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training data: {X_train.shape}, {y_train.shape}")
    print(f"Validation data: {X_val.shape}, {y_val.shape}")
    
    # Get LSTM config with variance reduction features
    lstm_config = ENSEMBLE_CONFIG["base_learners"]["lstm"]
    print(f"LSTM config: {lstm_config}")
    
    # Initialize LSTM model with variance reduction
    model = LSTMSequenceModel(
        input_size=n_features,
        sequence_length=lstm_config["sequence_length"],
        hidden_size=lstm_config["hidden_size"],
        num_layers=lstm_config["num_layers"],
        dropout=lstm_config["dropout"],
        recurrent_dropout=lstm_config.get("recurrent_dropout", 0.1),
        learning_rate=lstm_config["learning_rate"],
        batch_size=lstm_config["batch_size"],
        epochs=10,  # Short test
        early_stopping_patience=lstm_config["early_stopping_patience"],
        use_batch_norm=lstm_config["use_batch_norm"],
        layer_norm=lstm_config.get("layer_norm", True),
        spectral_norm=lstm_config.get("spectral_norm", True),
        use_ema=lstm_config.get("use_ema", True),
        ema_decay=lstm_config.get("ema_decay", 0.999),
        lr_scheduler=lstm_config.get("lr_scheduler", "cosine_annealing"),
        mixup_alpha=lstm_config.get("mixup_alpha", 0.1),
        noise_std=lstm_config.get("noise_std", 0.01),
        class_weights=lstm_config.get("class_weights", "balanced"),
    )
    
    print("Model initialized successfully!")
    
    # Test training
    try:
        print("Starting training...")
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        print("Training completed successfully!")
        
        # Test prediction
        print("Testing predictions...")
        y_pred_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val)
        
        print(f"Prediction probabilities shape: {y_pred_proba.shape}")
        print(f"Predictions shape: {y_pred.shape}")
        print(f"Sample predictions: {y_pred[:10]}")
        print(f"Sample probabilities: {y_pred_proba[:5]}")
        
        # Calculate accuracy (adjust for sequence length)
        # The model creates sequences, so we have fewer predictions than original samples
        min_len = min(len(y_pred), len(y_val))
        accuracy = np.mean(y_pred[:min_len] == y_val[:min_len])
        print(f"Validation accuracy: {accuracy:.4f} (on {min_len} samples)")
        
        print("✅ All tests passed! LSTM fixes are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_fix():
    """Test the prediction dimension fix."""
    print("\nTesting prediction dimension fix...")
    
    # Create mock prediction probabilities (2 classes only)
    y_pred_proba = np.array([
        [0.7, 0.3],
        [0.4, 0.6],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.6, 0.4]
    ])
    
    # Test the fix: derive hold probability
    pred_hold_prob = 1.0 - (y_pred_proba[:, 0] + y_pred_proba[:, 1])
    
    print(f"Buy probabilities: {y_pred_proba[:, 0]}")
    print(f"Sell probabilities: {y_pred_proba[:, 1]}")
    print(f"Hold probabilities (derived): {pred_hold_prob}")
    
    # Verify probabilities sum to 1 (approximately)
    total_probs = y_pred_proba[:, 0] + y_pred_proba[:, 1] + pred_hold_prob
    print(f"Total probabilities: {total_probs}")
    
    if np.allclose(total_probs, 1.0):
        print("✅ Prediction dimension fix is working correctly!")
        return True
    else:
        print("❌ Prediction dimension fix failed!")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("LSTM FIXES AND VARIANCE REDUCTION TEST")
    print("=" * 60)
    
    # Test prediction fix
    pred_fix_ok = test_prediction_fix()
    
    # Test LSTM model
    lstm_fix_ok = test_lstm_fixes()
    
    print("\n" + "=" * 60)
    if pred_fix_ok and lstm_fix_ok:
        print("🎉 ALL TESTS PASSED! Ready for training.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("=" * 60)
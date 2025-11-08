"""
Test script to verify training plot generation without full retraining
Uses synthetic data for quick local testing
"""
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from src.models.hybrid_ensemble import HybridEnsemble
from src.config import ENSEMBLE_CONFIG

# Configure logger
logger.info("Starting plot generation test")

def generate_synthetic_data(n_samples=1000, n_features=58, sequence_length=22):
    """Generate synthetic financial time series data"""
    logger.info(f"Generating {n_samples} synthetic samples with {n_features} features")
    
    # Create synthetic features resembling financial data
    np.random.seed(42)
    
    # Generate base features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Add some structure to make it learnable
    # Feature 0-9: trend indicators
    X[:, 0:10] = np.cumsum(np.random.randn(n_samples, 10) * 0.1, axis=0)
    
    # Feature 10-19: momentum indicators
    X[:, 10:20] = np.sin(np.linspace(0, 10*np.pi, n_samples))[:, None] + np.random.randn(n_samples, 10) * 0.5
    
    # Feature 20-29: volatility indicators
    X[:, 20:30] = np.abs(np.random.randn(n_samples, 10))
    
    # Remaining features: random technical indicators
    X[:, 30:] = np.random.randn(n_samples, n_features - 30) * 2
    
    # Generate target with some correlation to features
    # Use simple rule: if sum of first 10 features > threshold -> BUY (2)
    #                   if sum < -threshold -> SELL (0)
    #                   else HOLD (1)
    feature_sum = X[:, 0:10].sum(axis=1)
    upper_threshold = np.percentile(feature_sum, 67)
    lower_threshold = np.percentile(feature_sum, 33)
    
    y = np.ones(n_samples, dtype=np.int32)  # Default HOLD
    y[feature_sum > upper_threshold] = 2     # BUY
    y[feature_sum < lower_threshold] = 0     # SELL
    
    logger.info(f"Target distribution: SELL={np.sum(y==0)}, HOLD={np.sum(y==1)}, BUY={np.sum(y==2)}")
    
    return X, y

def main():
    """Main test function"""
    logger.info("="*60)
    logger.info("Training Plot Generation Test")
    logger.info("="*60)
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=1000, n_features=58)
    
    # Use stratified split to ensure all classes in both sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    logger.info(f"Train distribution: SELL={np.sum(y_train==0)}, HOLD={np.sum(y_train==1)}, BUY={np.sum(y_train==2)}")
    logger.info(f"Val distribution: SELL={np.sum(y_val==0)}, HOLD={np.sum(y_val==1)}, BUY={np.sum(y_val==2)}")
    
    # Configure small model for fast testing
    test_xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "learning_rate": 0.05,
        "max_depth": 4,
        "n_estimators": 50,  # Reduced from 200
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    }
    
    test_lstm_params = {
        "hidden_size": 32,     # Reduced from 64
        "num_layers": 1,       # Reduced from 2
        "num_classes": 3,
        "sequence_length": 22,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 15,          # Reduced from 50
        "early_stopping_patience": 5,
    }
    
    # Create model
    logger.info("Initializing HybridEnsemble model")
    model = HybridEnsemble(
        xgb_params=test_xgb_params,
        lstm_params=test_lstm_params,
    )
    
    # Define plot path
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plot_path = results_dir / "test_plot"
    
    logger.info(f"Training model with plot generation: {plot_path}")
    logger.info("This will take ~2-3 minutes...")
    
    # Train model with plotting enabled
    model.fit(
        X_train, 
        y_train, 
        X_val, 
        y_val,
        save_plots_path=str(plot_path)
    )
    
    logger.info("Training completed!")
    
    # Check if plots were created
    loss_plot = Path(f"{plot_path}_loss.png")
    acc_plot = Path(f"{plot_path}_acc.png")
    
    if loss_plot.exists():
        logger.success(f"✓ Loss plot created: {loss_plot}")
    else:
        logger.error(f"✗ Loss plot NOT created: {loss_plot}")
    
    if acc_plot.exists():
        logger.success(f"✓ Accuracy plot created: {acc_plot}")
    else:
        logger.error(f"✗ Accuracy plot NOT created: {acc_plot}")
    
    # Test predictions
    logger.info("Testing predictions on validation set")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    
    logger.info(f"Predictions shape: {y_pred.shape}")
    logger.info(f"Probabilities shape: {y_pred_proba.shape}")
    logger.info(f"Predicted distribution: SELL={np.sum(y_pred==0)}, HOLD={np.sum(y_pred==1)}, BUY={np.sum(y_pred==2)}")
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    acc = accuracy_score(y_val, y_pred)
    ba_acc = balanced_accuracy_score(y_val, y_pred)
    
    logger.info(f"Validation Accuracy: {acc:.4f}")
    logger.info(f"Validation Balanced Accuracy: {ba_acc:.4f}")
    
    logger.info("="*60)
    logger.success("Test completed successfully!")
    logger.info("="*60)
    
    if loss_plot.exists() and acc_plot.exists():
        logger.info(f"View plots at:")
        logger.info(f"  - {loss_plot.absolute()}")
        logger.info(f"  - {acc_plot.absolute()}")
    else:
        logger.warning("Some plots were not generated. Check the logs above.")

if __name__ == "__main__":
    main()

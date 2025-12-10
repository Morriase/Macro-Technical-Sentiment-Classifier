#!/usr/bin/env python3
"""
Compare LSTM performance with and without dropout to demonstrate variance reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.models.lstm_model import LSTMSequenceModel

def test_dropout_comparison():
    """Compare LSTM with and without dropout."""
    print("Comparing LSTM with and without dropout...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    configs = [
        {
            "name": "With Dropout (0.2)",
            "dropout": 0.2,
            "recurrent_dropout": 0.1,
            "color": "red"
        },
        {
            "name": "No Dropout (BatchNorm only)",
            "dropout": 0.0,
            "recurrent_dropout": 0.0,
            "color": "blue"
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        model = LSTMSequenceModel(
            input_size=n_features,
            sequence_length=40,
            hidden_size=64,
            num_layers=2,
            dropout=config["dropout"],
            recurrent_dropout=config["recurrent_dropout"],
            learning_rate=5e-4,
            batch_size=256,
            epochs=15,
            early_stopping_patience=25,
            use_batch_norm=True,
            layer_norm=True,
            spectral_norm=True,
            use_ema=True,
        )
        
        # Train model
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        # Store results
        results[config["name"]] = {
            "train_losses": model.train_losses,
            "val_losses": model.val_losses,
            "train_accs": model.train_accs,
            "val_accs": model.val_accs,
            "color": config["color"]
        }
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for name, data in results.items():
        epochs = range(1, len(data["train_losses"]) + 1)
        color = data["color"]
        
        # Training loss
        axes[0, 0].plot(epochs, data["train_losses"], label=name, color=color, linewidth=2)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation loss
        axes[0, 1].plot(epochs, data["val_losses"], label=name, color=color, linewidth=2)
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training accuracy
        axes[1, 0].plot(epochs, data["train_accs"], label=name, color=color, linewidth=2)
        axes[1, 0].set_title("Training Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation accuracy
        axes[1, 1].plot(epochs, data["val_accs"], label=name, color=color, linewidth=2)
        axes[1, 1].set_title("Validation Accuracy")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("dropout_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved as 'dropout_comparison.png'")
    
    # Calculate variance metrics
    print("\n" + "="*60)
    print("VARIANCE ANALYSIS")
    print("="*60)
    
    for name, data in results.items():
        val_acc_var = np.var(data["val_accs"])
        val_loss_var = np.var(data["val_losses"])
        final_val_acc = data["val_accs"][-1] if data["val_accs"] else 0
        
        print(f"\n{name}:")
        print(f"  Validation Accuracy Variance: {val_acc_var:.6f}")
        print(f"  Validation Loss Variance: {val_loss_var:.6f}")
        print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
    
    return results

if __name__ == "__main__":
    results = test_dropout_comparison()
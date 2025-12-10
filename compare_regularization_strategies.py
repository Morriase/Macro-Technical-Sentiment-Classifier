#!/usr/bin/env python3
"""
Comprehensive comparison of regularization strategies for LSTM variance reduction.
Tests three configurations:
1. Dropout only (no BatchNorm)
2. BatchNorm only (no Dropout) 
3. Both Dropout + BatchNorm
"""

import numpy as np
import matplotlib.pyplot as plt
from src.models.lstm_model import LSTMSequenceModel

def test_regularization_strategies():
    """Compare different regularization strategies."""
    print("Comparing LSTM regularization strategies...")
    
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
            "name": "Dropout Only",
            "dropout": 0.2,
            "recurrent_dropout": 0.1,
            "use_batch_norm": False,
            "layer_norm": False,
            "color": "red",
            "description": "Traditional dropout regularization"
        },
        {
            "name": "BatchNorm Only", 
            "dropout": 0.0,
            "recurrent_dropout": 0.0,
            "use_batch_norm": True,
            "layer_norm": True,
            "color": "blue",
            "description": "Our optimized approach"
        },
        {
            "name": "Dropout + BatchNorm",
            "dropout": 0.2,
            "recurrent_dropout": 0.1,
            "use_batch_norm": True,
            "layer_norm": True,
            "color": "green",
            "description": "Combined regularization"
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting: {config['name']} - {config['description']}")
        
        model = LSTMSequenceModel(
            input_size=n_features,
            sequence_length=40,
            hidden_size=64,
            num_layers=2,
            dropout=config["dropout"],
            recurrent_dropout=config["recurrent_dropout"],
            use_batch_norm=config["use_batch_norm"],
            layer_norm=config["layer_norm"],
            learning_rate=5e-4,
            batch_size=256,
            epochs=15,
            early_stopping_patience=25,
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
            "color": config["color"],
            "description": config["description"]
        }
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot training curves
    for name, data in results.items():
        epochs = range(1, len(data["train_losses"]) + 1)
        color = data["color"]
        
        # Training loss
        axes[0, 0].plot(epochs, data["train_losses"], label=name, color=color, linewidth=2)
        axes[0, 0].set_title("Training Loss", fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation loss
        axes[0, 1].plot(epochs, data["val_losses"], label=name, color=color, linewidth=2)
        axes[0, 1].set_title("Validation Loss", fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training accuracy
        axes[1, 0].plot(epochs, data["train_accs"], label=name, color=color, linewidth=2)
        axes[1, 0].set_title("Training Accuracy", fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation accuracy
        axes[1, 1].plot(epochs, data["val_accs"], label=name, color=color, linewidth=2)
        axes[1, 1].set_title("Validation Accuracy", fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Variance comparison bar chart
    names = list(results.keys())
    val_loss_vars = [np.var(results[name]["val_losses"]) for name in names]
    val_acc_vars = [np.var(results[name]["val_accs"]) for name in names]
    colors = [results[name]["color"] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[0, 2].bar(x - width/2, val_loss_vars, width, label='Val Loss Variance', color=colors, alpha=0.7)
    axes[0, 2].set_title("Validation Loss Variance\n(Lower = More Stable)", fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel("Variance")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 2].bar(x, val_acc_vars, width, label='Val Acc Variance', color=colors, alpha=0.7)
    axes[1, 2].set_title("Validation Accuracy Variance\n(Lower = More Stable)", fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel("Variance")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("regularization_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved as 'regularization_comparison.png'")
    
    # Detailed analysis
    print("\n" + "="*80)
    print("COMPREHENSIVE REGULARIZATION ANALYSIS")
    print("="*80)
    
    for name, data in results.items():
        val_acc_var = np.var(data["val_accs"])
        val_loss_var = np.var(data["val_losses"])
        final_val_acc = data["val_accs"][-1] if data["val_accs"] else 0
        final_train_acc = data["train_accs"][-1] if data["train_accs"] else 0
        
        # Calculate stability metrics
        val_acc_std = np.std(data["val_accs"])
        val_loss_std = np.std(data["val_losses"])
        
        # Calculate convergence (improvement from first to last epoch)
        train_improvement = data["train_accs"][-1] - data["train_accs"][0] if len(data["train_accs"]) > 0 else 0
        val_improvement = data["val_accs"][-1] - data["val_accs"][0] if len(data["val_accs"]) > 0 else 0
        
        print(f"\n{name} ({results[name]['description']}):")
        print(f"  📊 Final Performance:")
        print(f"     Training Accuracy: {final_train_acc:.4f}")
        print(f"     Validation Accuracy: {final_val_acc:.4f}")
        print(f"  📈 Learning Progress:")
        print(f"     Training Improvement: {train_improvement:.4f}")
        print(f"     Validation Improvement: {val_improvement:.4f}")
        print(f"  🎯 Stability Metrics:")
        print(f"     Val Accuracy Variance: {val_acc_var:.6f}")
        print(f"     Val Accuracy Std Dev: {val_acc_std:.4f}")
        print(f"     Val Loss Variance: {val_loss_var:.6f}")
        print(f"     Val Loss Std Dev: {val_loss_std:.4f}")
    
    # Ranking analysis
    print(f"\n{'='*80}")
    print("RANKING ANALYSIS")
    print("="*80)
    
    # Rank by stability (lower variance = better)
    stability_ranking = sorted(results.items(), key=lambda x: np.var(x[1]["val_losses"]))
    print(f"\n🏆 STABILITY RANKING (by validation loss variance):")
    for i, (name, data) in enumerate(stability_ranking, 1):
        var = np.var(data["val_losses"])
        print(f"  {i}. {name}: {var:.6f}")
    
    # Rank by final performance
    performance_ranking = sorted(results.items(), key=lambda x: x[1]["val_accs"][-1] if x[1]["val_accs"] else 0, reverse=True)
    print(f"\n🎯 PERFORMANCE RANKING (by final validation accuracy):")
    for i, (name, data) in enumerate(performance_ranking, 1):
        acc = data["val_accs"][-1] if data["val_accs"] else 0
        print(f"  {i}. {name}: {acc:.4f}")
    
    # Best overall recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print("="*80)
    
    best_stability = stability_ranking[0][0]
    best_performance = performance_ranking[0][0]
    
    if best_stability == best_performance:
        print(f"🌟 CLEAR WINNER: {best_stability}")
        print(f"   Best in both stability AND performance!")
    else:
        print(f"🎯 Best Performance: {best_performance}")
        print(f"🛡️  Best Stability: {best_stability}")
        print(f"💡 Consider trade-offs based on your priority")
    
    return results

if __name__ == "__main__":
    results = test_regularization_strategies()
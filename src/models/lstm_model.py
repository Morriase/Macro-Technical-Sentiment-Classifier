"""
LSTM Sequence Model for Temporal Pattern Recognition
Designed for financial time series with strong temporal dependencies
Optimized for GPU/CUDA training on Kaggle
"""
from torch.amp import autocast, GradScaler
from src.config import ENSEMBLE_CONFIG, GPU_CONFIG, DEVICE, USE_CUDA
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
from typing import Tuple, Optional, List
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path FIRST (before src imports)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Enable Automatic Mixed Precision (AMP) for faster GPU training


class LSTMSequenceClassifier(nn.Module):
    """
    Two-layer LSTM network for sequence classification
    Captures temporal dependencies in financial time series
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM classifier

        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (Buy/Sell/Hold)
            dropout: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMSequenceClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)

        # Softmax for probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_hidden=False):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            return_hidden: Whether to return final hidden state

        Returns:
            Logits or (logits, hidden_state) if return_hidden=True
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_length, hidden_size * num_directions)
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        # c_n shape: (num_layers * num_directions, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Get the final hidden state from last layer
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]

        # Apply dropout
        hidden = self.dropout(hidden)

        # Fully connected layer
        logits = self.fc(hidden)

        if return_hidden:
            return logits, hidden

        return logits

    def predict_proba(self, x):
        """Predict class probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = self.softmax(logits)
        return probs


class LSTMSequenceModel:
    """
    Wrapper for LSTM model with training and prediction utilities
    """

    def __init__(
        self,
        input_size: int,
        sequence_length: int = 22,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: Optional[str] = None,
    ):
        """
        Initialize LSTM sequence model

        Args:
            input_size: Number of features per timestep
            sequence_length: Look-back window (number of timesteps)
            hidden_size: LSTM hidden units
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        # Device - use config or auto-detect
        if device is None:
            self.device = DEVICE
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = LSTMSequenceClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)

        # Mixed precision training (for faster GPU training)
        self.use_amp = GPU_CONFIG['mixed_precision']
        self.scaler_amp = GradScaler('cuda') if self.use_amp else None

        # Gradient accumulation for larger effective batch size
        self.gradient_accumulation_steps = GPU_CONFIG['gradient_accumulation_steps']

        # Scaler for feature normalization
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

        # Reduced logging for cleaner output
        # logger.info(f"LSTM model initialized on {self.device}")
        # logger.info(f"CUDA available: {USE_CUDA}, Mixed Precision: {self.use_amp}")
        # logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        # if USE_CUDA:
        #     logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        #     logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def prepare_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform 2D tabular data into 3D sequences for LSTM

        Args:
            X: Features array (n_samples, n_features)
            y: Labels array (n_samples,) - optional

        Returns:
            Tuple of (X_sequences, y_sequences) where X_sequences has shape
            (n_samples - sequence_length + 1, sequence_length, n_features)
        """
        n_samples, n_features = X.shape

        # Create sequences
        X_sequences = []
        y_sequences = [] if y is not None else None

        for i in range(self.sequence_length, n_samples + 1):
            X_sequences.append(X[i - self.sequence_length:i])
            if y is not None:
                y_sequences.append(y[i - 1])  # Use label of last timestep

        X_sequences = np.array(X_sequences)

        if y is not None:
            y_sequences = np.array(y_sequences)
            return X_sequences, y_sequences

        return X_sequences, None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_plots_path: Optional[str] = None,
    ):
        """
        Train LSTM model

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        # logger.info("Starting LSTM training")

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_scaled, y)
        # logger.info(f"Prepared {len(X_seq)} training sequences")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_train_tensor = torch.LongTensor(y_seq).to(self.device)

        # Validation data
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_seq).to(self.device)
        else:
            X_val_tensor = None
            y_val_tensor = None

        # Initialize histories for diagnostics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0

            # Mini-batch training with gradient accumulation
            indices = torch.randperm(len(X_train_tensor))
            optimizer.zero_grad()

            for batch_idx, i in enumerate(range(0, len(indices), self.batch_size)):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]

                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                    # Backward pass with scaled gradients
                    self.scaler_amp.scale(loss).backward()

                    # Update weights after accumulating gradients
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler_amp.step(optimizer)
                        self.scaler_amp.update()
                        optimizer.zero_grad()
                else:
                    # Standard training without AMP
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps

            avg_loss = total_loss / (len(indices) / self.batch_size)

            # Compute training accuracy on a sample (to avoid OOM on large datasets)
            # Use last batch for efficiency
            self.model.eval()
            with torch.no_grad():
                # Sample up to 1000 random indices to estimate train accuracy
                sample_size = min(1000, len(X_train_tensor))
                sample_indices = torch.randperm(
                    len(X_train_tensor))[:sample_size]
                X_sample = X_train_tensor[sample_indices]
                y_sample = y_train_tensor[sample_indices]

                train_logits = self.model(X_sample)
                train_preds = torch.argmax(train_logits, dim=1)
                train_acc = (train_preds == y_sample).float().mean().item()

            # Record training metrics
            self.train_losses.append(avg_loss)
            self.train_accs.append(train_acc)

            # Validation
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_preds = torch.argmax(val_outputs, dim=1)
                    val_acc = (val_preds == y_val_tensor).float().mean().item()

                # Record validation metrics
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

                # Only log every 10 epochs or at early stopping
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs} - "
                        f"Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
                    )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info(
                            f"Early stopping at epoch {epoch+1} - Val Loss: {val_loss:.4f}")
                        break
            else:
                if (epoch + 1) % 20 == 0:  # Log every 20 epochs instead of 10
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        self.is_fitted = True
        # logger.info("LSTM training completed")

        # Optionally save training curves (loss & accuracy)
        if save_plots_path is not None:
            try:
                self._save_training_plots(save_plots_path)
            except Exception as e:
                logger.warning(f"Failed to save training plots: {e}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Features array (n_samples, n_features)

        Returns:
            Probability array (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Normalize features
        X_scaled = self.scaler.transform(X)

        # Prepare sequences
        X_seq, _ = self.prepare_sequences(X_scaled)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            probs = self.model.predict_proba(X_tensor)

        return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Features array (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_final_hidden_states(self, X: np.ndarray) -> np.ndarray:
        """
        Get final hidden states for use in meta-learner

        Args:
            X: Features array (n_samples, n_features)

        Returns:
            Hidden states (n_samples, hidden_size)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Normalize features
        X_scaled = self.scaler.transform(X)

        # Prepare sequences
        X_seq, _ = self.prepare_sequences(X_scaled)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Get hidden states
        self.model.eval()
        with torch.no_grad():
            _, hidden = self.model(X_tensor, return_hidden=True)

        return hidden.cpu().numpy()

    def _save_training_plots(self, out_prefix: str):
        """Save loss and accuracy curves to files with given prefix.

        Args:
            out_prefix: Path prefix (without extension) where plots will be saved.
                        Two files will be created: <out_prefix>_loss.png and
                        <out_prefix>_acc.png
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            raise RuntimeError("matplotlib not available in environment")

        # Loss curve
        plt.figure(figsize=(8, 5))
        if hasattr(self, 'train_losses') and len(self.train_losses) > 0:
            plt.plot(self.train_losses, label='Train Loss')
        if hasattr(self, 'val_losses') and len(self.val_losses) > 0:
            plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Loss Curve')
        plt.legend()
        plt.grid(True)
        loss_path = f"{out_prefix}_loss.png"
        plt.savefig(loss_path, bbox_inches='tight')
        plt.close()

        # Accuracy curve
        plt.figure(figsize=(8, 5))
        if hasattr(self, 'train_accs') and len(self.train_accs) > 0:
            plt.plot(self.train_accs, label='Train Acc')
        if hasattr(self, 'val_accs') and len(self.val_accs) > 0:
            plt.plot(self.val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('LSTM Accuracy Curve')
        plt.legend()
        plt.grid(True)
        acc_path = f"{out_prefix}_acc.png"
        plt.savefig(acc_path, bbox_inches='tight')
        plt.close()

    def save_model(self, filepath: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': {
                'input_size': self.input_size,
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_classes': self.num_classes,
                'dropout': self.dropout,
            },
        }, filepath)
        logger.info(f"LSTM model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from disk"""
        # PyTorch 2.6+ changed weights_only default to True
        # Set to False to load models with sklearn objects (MinMaxScaler)
        checkpoint = torch.load(
            filepath, map_location=self.device, weights_only=False)

        # Restore config
        config = checkpoint['config']
        self.input_size = config['input_size']
        self.sequence_length = config['sequence_length']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.num_classes = config['num_classes']
        self.dropout = config['dropout']

        # Rebuild model
        self.model = LSTMSequenceClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
        ).to(self.device)

        # Load state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.is_fitted = True

        logger.info(f"LSTM model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate sample time series data
    X, y = make_classification(
        n_samples=5000,
        n_features=30,
        n_informative=20,
        n_classes=3,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # No shuffle for time series
    )

    # Initialize and train LSTM
    lstm_model = LSTMSequenceModel(
        input_size=X.shape[1],
        sequence_length=22,
        hidden_size=128,
        num_layers=2,
        epochs=50,
    )

    lstm_model.fit(X_train, y_train)

    # Predictions
    y_pred_proba = lstm_model.predict_proba(X_test)
    y_pred = lstm_model.predict(X_test)

    # Get hidden states for meta-learner
    hidden_states = lstm_model.get_final_hidden_states(X_test)

    print(f"Predictions shape: {y_pred.shape}")
    print(f"Probabilities shape: {y_pred_proba.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")

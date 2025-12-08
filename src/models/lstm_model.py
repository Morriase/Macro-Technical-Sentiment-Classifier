"""
LSTM Sequence Model for Temporal Pattern Recognition
MEMORY OPTIMIZED: True Lazy Loading (No 3D Array Creation)
Designed for financial time series with strong temporal dependencies
"""
from torch.amp import autocast, GradScaler
from src.config import ENSEMBLE_CONFIG, GPU_CONFIG, DEVICE, USE_CUDA
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger
from typing import Tuple, Optional, List
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import sys
import gc
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Add project root to path FIRST
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class LazySequenceDataset(Dataset):
    """
    Zero-copy dataset that slices sequences on-the-fly.
    Keeps data in 2D to save 40x RAM.

    RAM Math:
    - 2D Array: 2M rows × 33 features × 4 bytes = 0.26 GB
    - 3D Array: 2M rows × 40 seq × 33 features × 4 bytes = 10.5 GB
    - This class keeps data 2D and only creates 3D windows per batch
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int):
        # Store as float32 to save 50% RAM compared to float64
        self.X = torch.tensor(X.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Create 3D view only for this specific item
        # idx -> [idx, idx+seq_len]
        x_window = self.X[idx: idx + self.seq_len]
        if self.y is not None:
            # Target is the label at the END of the sequence
            y_label = self.y[idx + self.seq_len - 1]
            return x_window, y_label
        return x_window


class LSTMSequenceClassifier(nn.Module):
    """
    Deep LSTM network for sequence classification
    Captures temporal dependencies in financial time series
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_batch_norm: bool = True,
        hidden_activation: str = None,
    ):
        super(LSTMSequenceClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_batch_norm = use_batch_norm

        # 1. Input BatchNorm - stabilizes training
        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(input_size)

        # 2. LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 3. Post-LSTM BatchNorm
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        if use_batch_norm:
            self.lstm_bn = nn.BatchNorm1d(fc_input_size)

        # 4. Dropout & Output Head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_hidden=False):
        """
        Forward pass: Input → BatchNorm → LSTM → BatchNorm → Dropout → FC → Logits

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            return_hidden: Whether to return final hidden state
        """
        # Apply Input BatchNorm
        if self.use_batch_norm:
            # BN expects (batch, features, seq_len)
            batch_size, seq_len, features = x.shape
            x = x.transpose(1, 2)
            x = self.input_bn(x)
            x = x.transpose(1, 2)

        # LSTM - we only need the final hidden state
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]

        # Apply Post-LSTM BatchNorm
        if self.use_batch_norm:
            hidden = self.lstm_bn(hidden)

        # Dropout & FC
        hidden = self.dropout(hidden)
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
    Memory-Optimized LSTM Wrapper with True Lazy Loading

    Key Optimization: Never creates 3D arrays in RAM.
    Data stays 2D, sequences are sliced on-the-fly per batch.
    """

    def __init__(
        self,
        input_size: int,
        sequence_length: int = 40,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 500,
        early_stopping_patience: int = 25,
        device: Optional[str] = None,
        l1_lambda: float = 1e-7,
        l2_lambda: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        label_smoothing: float = 0.1,
        lr_warmup_epochs: int = 5,
        lr_min_factor: float = 0.01,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        bidirectional: bool = False,
        use_batch_norm: bool = True,
        hidden_activation: str = None,
        **kwargs,
    ):
        """
        Initialize LSTM sequence model with memory optimization.
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
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.beta1 = beta1
        self.beta2 = beta2
        self.label_smoothing = label_smoothing
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_factor = lr_min_factor
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.bidirectional = bidirectional
        self.use_batch_norm = use_batch_norm

        # Device
        self.device = torch.device(device if device else DEVICE)

        # Initialize model
        self.model = LSTMSequenceClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            use_batch_norm=use_batch_norm,
        ).to(self.device)

        # Multi-GPU Support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(self.model)

        # Mixed precision training
        self.use_amp = GPU_CONFIG.get('mixed_precision', False)
        self.scaler_amp = GradScaler(
            'cuda') if self.use_amp and torch.cuda.is_available() else None

        # Scaler for feature normalization (-1, 1) for Tanh/Swish
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.is_fitted = False

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_state = None

        # Log model info
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LSTM initialized: {hidden_size} units × {num_layers} layers, "
                    f"BatchNorm={use_batch_norm}, Dropout={dropout}, Params={param_count:,}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_plots_path: Optional[str] = None,
    ):
        """
        Train LSTM model with TRUE LAZY LOADING (no 3D array creation).
        """
        # 1. Scale Data (convert to float32 to save RAM)
        X = X.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X)

        # 2. Create Lazy Datasets (NO 3D Array Creation!)
        train_dataset = LazySequenceDataset(X_scaled, y, self.sequence_length)

        # Free original arrays
        del X_scaled
        gc.collect()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # CRITICAL: Never shuffle time series
            num_workers=0,  # Avoid memory duplication from multiprocessing
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = X_val.astype(np.float32)
            X_val_scaled = self.scaler.transform(X_val)
            val_dataset = LazySequenceDataset(
                X_val_scaled, y_val, self.sequence_length)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
            del X_val_scaled
            gc.collect()

        # 3. Class Weights (handle imbalance)
        y_np = np.array(y)
        unique_classes = np.unique(y_np)
        weights = compute_class_weight(
            'balanced', classes=unique_classes, y=y_np)
        full_weights = np.ones(self.num_classes, dtype=np.float32)
        for cls, weight in zip(unique_classes, weights):
            if int(cls) < self.num_classes:
                full_weights[int(cls)] = weight
        class_weights = torch.tensor(
            full_weights, device=self.device, dtype=torch.float32)
        logger.info(
            f"Using dynamic class weights: {class_weights.cpu().numpy()}")

        # 4. Loss & Optimizer
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=self.label_smoothing)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.l2_lambda,
        )

        # 5. Learning Rate Scheduler with Warmup
        def lr_lambda(epoch):
            if epoch < self.lr_warmup_epochs:
                return 0.5 + 0.5 * (epoch / self.lr_warmup_epochs)
            else:
                progress = (epoch - self.lr_warmup_epochs) / \
                    max(1, self.epochs - self.lr_warmup_epochs)
                return self.lr_min_factor + (1 - self.lr_min_factor) * (1 + np.cos(np.pi * progress)) / 2
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 6. Training Loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Forward pass (with optional AMP)
                if self.use_amp and self.scaler_amp:
                    with autocast('cuda'):
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        if self.l1_lambda > 0:
                            l1_penalty = sum(p.abs().sum()
                                             for p in self.model.parameters())
                            loss = loss + self.l1_lambda * l1_penalty

                    self.scaler_amp.scale(loss).backward()
                    self.scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm)
                    self.scaler_amp.step(optimizer)
                    self.scaler_amp.update()
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    if self.l1_lambda > 0:
                        l1_penalty = sum(p.abs().sum()
                                         for p in self.model.parameters())
                        loss = loss + self.l1_lambda * l1_penalty

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm)
                    optimizer.step()

                total_loss += loss.item()

                # Training accuracy
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    train_correct += (preds == batch_y).sum().item()
                    train_total += batch_y.size(0)

            avg_train_loss = total_loss / len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0.0
            self.train_losses.append(avg_train_loss)
            self.train_accs.append(train_acc)

            # Validation
            avg_val_loss = 0
            val_acc = 0
            if val_loader:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        preds = torch.argmax(outputs, dim=1)
                        val_correct += (preds == batch_y).sum().item()
                        val_total += batch_y.size(0)

                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                self.val_losses.append(avg_val_loss)
                self.val_accs.append(val_acc)

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best state
                    self.best_state = {k: v.cpu().clone()
                                       for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

            scheduler.step()

            # Logging
            if val_loader:
                logger.info(
                    f"    LSTM Epoch {epoch+1:3d}/{self.epochs} | "
                    f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.3f} | "
                    f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.3f}"
                )
            else:
                logger.info(
                    f"    LSTM Epoch {epoch+1:3d}/{self.epochs} | Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.3f}")

            # Early stopping
            if val_loader and patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} - Best Val Loss: {best_val_loss:.4f}")
                if self.best_state:
                    self.model.load_state_dict(self.best_state)
                break

        self.is_fitted = True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities with memory-optimized lazy loading.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Scale data
        X = X.astype(np.float32)
        X_scaled = self.scaler.transform(X)

        # Use lazy dataset for inference too
        dataset = LazySequenceDataset(X_scaled, None, self.sequence_length)
        loader = DataLoader(dataset, batch_size=1024,
                            shuffle=False, num_workers=0)

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.device)

                # Handle DataParallel wrapper
                if isinstance(self.model, nn.DataParallel):
                    logits = self.model(batch_X)
                    probs = torch.softmax(logits, dim=1)
                else:
                    probs = self.model.predict_proba(batch_X)

                all_probs.append(probs.cpu().numpy().astype(np.float32))

                del batch_X, probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del X_scaled
        gc.collect()

        return np.vstack(all_probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_final_hidden_states(self, X: np.ndarray) -> np.ndarray:
        """Get final hidden states for use in meta-learner."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = X.astype(np.float32)
        X_scaled = self.scaler.transform(X)

        dataset = LazySequenceDataset(X_scaled, None, self.sequence_length)
        loader = DataLoader(dataset, batch_size=1024,
                            shuffle=False, num_workers=0)

        self.model.eval()
        all_hidden = []

        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.device)

                if isinstance(self.model, nn.DataParallel):
                    # DataParallel doesn't support return_hidden
                    _, hidden = self.model.module.forward(
                        batch_X, return_hidden=True)
                else:
                    _, hidden = self.model.forward(batch_X, return_hidden=True)

                all_hidden.append(hidden.cpu().numpy().astype(np.float32))

        del X_scaled
        gc.collect()

        return np.vstack(all_hidden)

    def save_model(self, path: str):
        """Save model state and scaler."""
        import joblib

        # Save PyTorch model
        model_state = self.model.module.state_dict() if isinstance(
            self.model, nn.DataParallel) else self.model.state_dict()
        torch.save({
            'model_state_dict': model_state,
            'input_size': self.input_size,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'use_batch_norm': self.use_batch_norm,
        }, f"{path}_lstm.pt")

        # Save scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        logger.info(f"LSTM model saved to {path}_lstm.pt")

    def load_model(self, path: str):
        """Load model state and scaler."""
        import joblib

        # Load checkpoint
        checkpoint = torch.load(f"{path}_lstm.pt", map_location=self.device)

        # Reinitialize model with saved params
        self.model = LSTMSequenceClassifier(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            num_classes=checkpoint['num_classes'],
            dropout=checkpoint['dropout'],
            bidirectional=checkpoint['bidirectional'],
            use_batch_norm=checkpoint['use_batch_norm'],
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load scaler
        self.scaler = joblib.load(f"{path}_scaler.joblib")
        self.is_fitted = True
        logger.info(f"LSTM model loaded from {path}_lstm.pt")

    def _save_training_plots(self, save_path: str):
        """Save training curves."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Loss plot
            axes[0].plot(self.train_losses, label='Train Loss')
            if self.val_losses:
                axes[0].plot(self.val_losses, label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()

            # Accuracy plot
            axes[1].plot(self.train_accs, label='Train Acc')
            if self.val_accs:
                axes[1].plot(self.val_accs, label='Val Acc')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training Accuracy')
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(f"{save_path}_lstm_training.png", dpi=100)
            plt.close()
            logger.info(
                f"Training plots saved to {save_path}_lstm_training.png")
        except Exception as e:
            logger.warning(f"Could not save training plots: {e}")


# Backward compatibility alias
LSTMModel = LSTMSequenceModel

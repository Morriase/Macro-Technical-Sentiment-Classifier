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
    Deep LSTM network for sequence classification
    Captures temporal dependencies in financial time series
    OPTIMIZED: Swish activation + BatchNorm per senior ML engineer's findings
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 40,
        num_layers: int = 1,
        num_classes: int = 2,  # BINARY: Buy/Sell
        dropout: float = 0.0,
        bidirectional: bool = False,
        use_batch_norm: bool = True,
        hidden_activation: str = "swish",
    ):
        """
        Initialize LSTM classifier

        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of hidden units in LSTM (default 40 per MQL5)
            num_layers: Number of LSTM layers (default 1 per MQL5)
            num_classes: Number of output classes (Buy/Sell)
            dropout: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
            use_batch_norm: Whether to use BatchNormalization (stabilizes training)
            hidden_activation: Unused (LSTM has internal non-linearity via gates)
        """
        super(LSTMSequenceClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_batch_norm = use_batch_norm
        self.hidden_activation = hidden_activation

        # Input BatchNorm - stabilizes training (author's recommendation)
        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(input_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Post-LSTM BatchNorm
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        if use_batch_norm:
            self.lstm_bn = nn.BatchNorm1d(fc_input_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Note: No activation after LSTM output
        # LSTM gates provide internal non-linearity (sigmoid/tanh gates + cell state)
        # Additional activation function would be redundant and harmful to learning

        # Fully connected output layer
        self.fc = nn.Linear(fc_input_size, num_classes)

        # Softmax for probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_hidden=False):
        """
        Forward pass: Input → BatchNorm → LSTM → Dropout → FC → Logits
        No activation after LSTM (LSTM has internal non-linearity via gates)

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            return_hidden: Whether to return final hidden state

        Returns:
            Logits or (logits, hidden_state) if return_hidden=True
        """
        # Apply input BatchNorm (stabilizes training)
        if self.use_batch_norm:
            # BatchNorm1d expects (batch, features), so transpose
            batch_size, seq_len, features = x.shape
            x = x.transpose(1, 2)  # (batch, features, seq_len)
            x = self.input_bn(x)
            x = x.transpose(1, 2)  # (batch, seq_len, features)

        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_length, hidden_size * num_directions)
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Get the final hidden state from last layer
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]

        # Apply post-LSTM BatchNorm
        if self.use_batch_norm:
            hidden = self.lstm_bn(hidden)

        # No activation after LSTM - LSTM gates already provide non-linearity
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
    OPTIMIZED: Swish activation, BatchNorm, author's parameters
    """

    def __init__(
        self,
        input_size: int,
        sequence_length: int = 40,
        hidden_size: int = 40,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.0,
        learning_rate: float = 3e-5,
        batch_size: int = 10000,
        epochs: int = 500,
        early_stopping_patience: int = 20,
        device: Optional[str] = None,
        l1_lambda: float = 1e-7,
        l2_lambda: float = 1e-5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        label_smoothing: float = 0.1,
        lr_warmup_epochs: int = 3,
        lr_min_factor: float = 0.01,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        bidirectional: bool = False,
        use_batch_norm: bool = True,
        hidden_activation: str = "swish",
        **kwargs,  # Accept extra params gracefully
    ):
        """
        Initialize LSTM sequence model (MQL5_LSTM.mq5 exact parameters)

        Architecture: Input(160) → BatchNorm → LSTM(40) → Output(2)
        Matches MQL5 training: 1 LSTM layer, BatchNorm, no dropout, 3e-5 LR

        Args:
            input_size: Number of features per timestep
            sequence_length: Look-back window (40 bars = BarsToLine in MQL5)
            hidden_size: LSTM hidden units (40, matches HiddenLayer in MQL5)
            num_layers: Number of LSTM layers (1 = MQL5 default, no hidden layers)
            num_classes: Number of output classes (3 for Buy/Sell/Hold)
            dropout: Dropout rate (0.0 - BatchNorm replaces dropout per MQL5)
            learning_rate: Initial learning rate (3e-5 from MQL5, NOT 3e-4)
            batch_size: Training batch size (10000 from MQL5)
            epochs: Maximum training epochs (500 from MQL5)
            early_stopping_patience: Patience for early stopping (20 from MQL5)
            device: 'cuda', 'cpu', or None (auto-detect)
            l1_lambda: L1 regularization (1e-7, author's value)
            l2_lambda: L2 regularization (1e-5, author's value)
            label_smoothing: Label smoothing factor (0.1)
            lr_warmup_epochs: Epochs for learning rate warmup
            lr_min_factor: Minimum LR as fraction of initial
            max_grad_norm: Gradient clipping norm (1.0)
            gradient_accumulation_steps: Steps to accumulate gradients
            bidirectional: Use bidirectional LSTM
            use_batch_norm: Use BatchNormalization (author's recommendation)
            hidden_activation: Activation function ('swish' accelerates training)
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
        # Regularization parameters
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        # Optimizer momentum parameters
        self.beta1 = beta1
        self.beta2 = beta2
        # Training parameters
        self.label_smoothing = label_smoothing
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_factor = lr_min_factor
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.bidirectional = bidirectional
        # New: BatchNorm and Swish activation
        self.use_batch_norm = use_batch_norm
        self.hidden_activation = hidden_activation

        # Device - use config or auto-detect
        if device is None:
            self.device = DEVICE
        else:
            self.device = torch.device(device)

        # Initialize model with BatchNorm and Swish
        self.model = LSTMSequenceClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            use_batch_norm=use_batch_norm,
            hidden_activation=hidden_activation,
        ).to(self.device)

        # Mixed precision training (for faster GPU training)
        self.use_amp = GPU_CONFIG['mixed_precision']
        self.scaler_amp = GradScaler('cuda') if self.use_amp else None

        # Scaler for feature normalization
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

        # Log model info
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LSTM initialized: {hidden_size} units × {num_layers} layers, "
                    f"BatchNorm={use_batch_norm}, Activation={hidden_activation}, "
                    f"Params={param_count:,}")

    def prepare_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform 2D tabular data into 3D sequences for LSTM
        Optimized for memory using stride tricks

        Args:
            X: Features array (n_samples, n_features)
            y: Labels array (n_samples,) - optional

        Returns:
            Tuple of (X_sequences, y_sequences) where X_sequences has shape
            (n_samples - sequence_length + 1, sequence_length, n_features)
        """
        n_samples, n_features = X.shape

        # Use stride_tricks to avoid memory duplication during sequence creation
        from numpy.lib.stride_tricks import sliding_window_view

        # Create sliding window view
        # Shape: (n_windows, n_features, window_size)
        windows = sliding_window_view(
            X, window_shape=self.sequence_length, axis=0)

        # Transpose to get (n_windows, window_size, n_features)
        # Make a contiguous copy to avoid non-writable tensor warning
        X_sequences = np.ascontiguousarray(windows.transpose(0, 2, 1))

        # Handle targets - CRITICAL FIX:
        # If we have n_samples, sliding windows create (n_samples - seq_length + 1) windows
        # Each window i spans from [i, i+seq_length), and targets the END of the window
        # So target index for window i is (i + seq_length - 1)
        # This means y_sequences should be y[seq_length-1:] (length = n_samples - seq_length + 1)
        if y is not None:
            # y[seq_length-1:] = y[39:] for seq_length=40
            # This has length = n_samples - seq_length + 1 = same as X_sequences
            y_sequences = y[self.sequence_length - 1:]

            # Verify alignment
            assert len(X_sequences) == len(y_sequences), \
                f"X_sequences ({len(X_sequences)}) and y_sequences ({len(y_sequences)}) length mismatch!"

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

        # Convert to tensors (keep on CPU initially for DataLoader)
        X_train_tensor = torch.FloatTensor(X_seq)
        y_train_tensor = torch.LongTensor(y_seq)

        # Validation data (keep on CPU initially)
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val)
            X_val_tensor = torch.FloatTensor(X_val_seq)
            y_val_tensor = torch.LongTensor(y_val_seq)
        else:
            X_val_tensor = None
            y_val_tensor = None

        # Create DataLoader for efficient batching
        from torch.utils.data import Dataset, DataLoader
        import gc

        # Lazy Dataset to avoid memory duplication
        class LazyWindowDataset(Dataset):
            def __init__(self, X, y, sequence_length):
                self.X = torch.FloatTensor(X)  # Keep 2D tensor
                self.y = torch.LongTensor(y) if y is not None else None
                self.seq_len = sequence_length

            def __len__(self):
                return len(self.X) - self.seq_len

            def __getitem__(self, idx):
                # Slice window on the fly
                x_window = self.X[idx: idx + self.seq_len]
                if self.y is not None:
                    y_label = self.y[idx + self.seq_len - 1]
                    return x_window, y_label
                return x_window

        # Create dataset using LazyWindowDataset
        # Note: X_scaled is numpy, convert to tensor inside dataset
        train_dataset = LazyWindowDataset(X_scaled, y, self.sequence_length)

        # Validation data
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_dataset = LazyWindowDataset(
                X_val_scaled, y_val, self.sequence_length)
        else:
            val_dataset = None

        # Clean up intermediate arrays to free memory
        # X_scaled is now held by dataset (as tensor), so we can delete the numpy version
        del X_scaled
        gc.collect()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # CRITICAL: Never shuffle time series for LSTM
            num_workers=GPU_CONFIG['num_workers'],  # Prefetch data to GPU
            pin_memory=GPU_CONFIG['pin_memory'],  # Faster GPU transfer
            # Keep workers alive
            persistent_workers=GPU_CONFIG['num_workers'] > 0,
        )

        # Initialize histories for diagnostics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # CLASS WEIGHTS: Dynamic calculation based on training data
        # Handles both Binary (Buy/Sell) and Multi-class (Buy/Sell/Hold)
        from sklearn.utils.class_weight import compute_class_weight

        # Ensure y is numpy array
        y_np = np.array(y)
        unique_classes = np.unique(y_np)

        # Calculate balanced weights
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_np
        )

        # Map to full weight tensor (handle missing classes if any)
        full_weights = np.ones(self.num_classes, dtype=np.float32)
        for cls, weight in zip(unique_classes, weights):
            if int(cls) < self.num_classes:
                full_weights[int(cls)] = weight

        class_weights = torch.tensor(
            full_weights, device=self.device, dtype=torch.float32)
        logger.info(
            f"Using dynamic class weights: {class_weights.cpu().numpy()}")

        # Loss function with class weights and label smoothing
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.label_smoothing
        )

        # Use AdamW optimizer - better weight decay implementation
        # AdamW decouples weight decay from gradient update (fixes rising loss)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.l2_lambda,
            eps=1e-8,
        )

        # Learning rate scheduler with warmup to prevent early instability
        # Phase 1: Linear warmup for lr_warmup_epochs (1 epoch to warm up quickly)
        # Phase 2: Cosine annealing decay
        def lr_lambda(epoch):
            if epoch < self.lr_warmup_epochs:
                # Linear warmup from 50% to 100% of learning rate (faster ramp-up)
                # Without activation, model needs stronger initial gradients
                return 0.5 + 0.5 * (epoch / self.lr_warmup_epochs)
            else:
                # Cosine annealing after warmup
                progress = (epoch - self.lr_warmup_epochs) / \
                    max(1, self.epochs - self.lr_warmup_epochs)
                return self.lr_min_factor + (1 - self.lr_min_factor) * (1 + np.cos(np.pi * progress)) / 2

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training loop
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0

        # Validation loader (create once)
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0
            optimizer.zero_grad()

            # Mini-batch training with DataLoader (parallel data loading)
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                # Move batch to GPU
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)

                        # Add L1 regularization (Lasso) manually
                        if self.l1_lambda > 0:
                            l1_penalty = sum(p.abs().sum()
                                             for p in self.model.parameters())
                            loss = loss + self.l1_lambda * l1_penalty

                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                    # Backward pass with scaled gradients
                    self.scaler_amp.scale(loss).backward()

                    # Update weights after accumulating gradients
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # Unscale gradients and clip them
                        self.scaler_amp.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.max_grad_norm)

                        self.scaler_amp.step(optimizer)
                        self.scaler_amp.update()
                        optimizer.zero_grad()
                else:
                    # Standard training without AMP
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)

                    # Add L1 regularization (Lasso) manually
                    if self.l1_lambda > 0:
                        l1_penalty = sum(p.abs().sum()
                                         for p in self.model.parameters())
                        loss = loss + self.l1_lambda * l1_penalty

                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps

                # Calculate training accuracy for this batch
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    train_correct += (preds == y_batch).sum().item()
                    train_total += y_batch.size(0)

            # Calculate average loss (number of batches)
            num_batches = len(train_loader)
            avg_loss = total_loss / num_batches

            # Calculate average training accuracy
            train_acc = train_correct / train_total if train_total > 0 else 0.0

            # Record training metrics
            self.train_losses.append(avg_loss)
            self.train_accs.append(train_acc)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for X_v, y_v in val_loader:
                        X_v = X_v.to(self.device, non_blocking=True)
                        y_v = y_v.to(self.device, non_blocking=True)

                        outputs = self.model(X_v)
                        loss = criterion(outputs, y_v)
                        val_loss += loss.item()

                        preds = torch.argmax(outputs, dim=1)
                        val_correct += (preds == y_v).sum().item()
                        val_total += y_v.size(0)

                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total

                # Record validation metrics
                self.val_losses.append(avg_val_loss)
                self.val_accs.append(val_acc)

                # Check for improvement
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                # No validation set
                avg_val_loss = 0
                val_acc = 0
            # Step the learning rate scheduler (CosineAnnealing steps every epoch)
            scheduler.step()

            # Log every epoch for full visibility during training
            if val_loader is not None:
                logger.info(
                    f"    LSTM Epoch {epoch+1:3d}/{self.epochs} | "
                    f"Train Loss: {avg_loss:.4f}, Acc: {train_acc:.3f} | "
                    f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.3f}"
                )
            else:
                logger.info(
                    f"    LSTM Epoch {epoch+1:3d}/{self.epochs} | Train Loss: {avg_loss:.4f}, Acc: {train_acc:.3f}")

            # Early stopping
            if val_loader is not None:
                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")
                    break

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
        Predict class probabilities (memory optimized with batching)

        Args:
            X: Features array (n_samples, n_features)

        Returns:
            Probability array (n_samples, n_classes)
        """
        import gc

        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Normalize features
        X_scaled = self.scaler.transform(X).astype(np.float32)

        # Prepare sequences
        X_seq, _ = self.prepare_sequences(X_scaled)
        del X_scaled
        gc.collect()

        # Predict in batches to avoid OOM
        self.model.eval()
        batch_size = 512
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                batch = torch.FloatTensor(
                    X_seq[i:i+batch_size]).to(self.device)
                probs = self.model.predict_proba(batch)
                all_probs.append(probs.cpu().numpy().astype(np.float32))
                del batch, probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return np.vstack(all_probs)

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

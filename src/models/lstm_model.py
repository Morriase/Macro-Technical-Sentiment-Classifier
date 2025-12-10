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
    
    ZIGZAG UPDATE: Supports dual targets (direction + magnitude)

    RAM Math:
    - 2D Array: 2M rows × 5 features × 4 bytes = 0.04 GB (was 0.26 GB with 33 features)
    - 3D Array: 2M rows × 40 seq × 5 features × 4 bytes = 1.6 GB (was 10.5 GB)
    - This class keeps data 2D and only creates 3D windows per batch
    """

    def __init__(self, X: np.ndarray, y_direction: np.ndarray, y_magnitude: np.ndarray = None, sequence_length: int = 40):
        # Store as float32 to save 50% RAM compared to float64
        self.X = torch.tensor(X.astype(np.float32), dtype=torch.float32)
        self.y_direction = torch.tensor(y_direction, dtype=torch.long) if y_direction is not None else None
        self.y_magnitude = torch.tensor(y_magnitude.astype(np.float32), dtype=torch.float32) if y_magnitude is not None else None
        self.seq_len = sequence_length
        self.dual_output = y_magnitude is not None

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Create 3D view only for this specific item
        # idx -> [idx, idx+seq_len]
        x_window = self.X[idx: idx + self.seq_len]
        
        if self.y_direction is not None:
            # Targets are at the END of the sequence
            y_dir = self.y_direction[idx + self.seq_len - 1]
            
            if self.dual_output:
                y_mag = self.y_magnitude[idx + self.seq_len - 1]
                return x_window, y_dir, y_mag
            else:
                return x_window, y_dir
        
        return x_window


class LSTMSequenceClassifier(nn.Module):
    """
    Deep LSTM network for sequence classification with DUAL OUTPUT
    
    ZIGZAG APPROACH:
    - Output 1: Direction (classification) - Buy (1) or Sell (0)
    - Output 2: Magnitude (regression) - Distance to next extremum
    
    Captures temporal dependencies in financial time series
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.1,
        bidirectional: bool = False,
        use_batch_norm: bool = True,
        layer_norm: bool = True,
        spectral_norm: bool = True,
        hidden_activation: str = None,
        dual_output: bool = True,  # ZIGZAG: Enable dual output
    ):
        super(LSTMSequenceClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_batch_norm = use_batch_norm
        self.layer_norm = layer_norm
        self.dual_output = dual_output

        # 1. Input BatchNorm - stabilizes training
        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(input_size)

        # 2. LSTM layers with recurrent dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 3. Post-LSTM normalization
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        
        if use_batch_norm:
            self.lstm_bn = nn.BatchNorm1d(fc_input_size)
        
        if layer_norm:
            self.layer_norm_layer = nn.LayerNorm(fc_input_size)

        # 4. Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.recurrent_dropout = nn.Dropout(recurrent_dropout) if recurrent_dropout > 0 else None
        
        # 5. DUAL OUTPUT HEADS (ZIGZAG APPROACH) with spectral normalization
        if dual_output:
            # Head 1: Direction (classification)
            direction_head = nn.Linear(fc_input_size, num_classes)
            if spectral_norm:
                self.direction_head = nn.utils.spectral_norm(direction_head)
            else:
                self.direction_head = direction_head
                
            # Head 2: Magnitude (regression)
            magnitude_head = nn.Linear(fc_input_size, 1)
            if spectral_norm:
                self.magnitude_head = nn.utils.spectral_norm(magnitude_head)
            else:
                self.magnitude_head = magnitude_head
        else:
            # Single output (backward compatibility)
            fc_layer = nn.Linear(fc_input_size, num_classes)
            if spectral_norm:
                self.fc = nn.utils.spectral_norm(fc_layer)
            else:
                self.fc = fc_layer
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_hidden=False):
        """
        Forward pass: Input → BatchNorm → LSTM → BatchNorm → Dropout → Dual Heads
        
        ZIGZAG APPROACH:
        Returns (direction_logits, magnitude_pred) if dual_output=True
        Returns direction_logits only if dual_output=False

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

        # Apply Post-LSTM normalization
        if self.use_batch_norm:
            hidden = self.lstm_bn(hidden)
        
        if self.layer_norm:
            hidden = self.layer_norm_layer(hidden)

        # Apply recurrent dropout if enabled
        if self.recurrent_dropout is not None:
            hidden = self.recurrent_dropout(hidden)

        # Regular dropout
        hidden = self.dropout(hidden)
        
        # DUAL OUTPUT (ZIGZAG APPROACH)
        if self.dual_output:
            direction_logits = self.direction_head(hidden)
            magnitude_pred = self.magnitude_head(hidden).squeeze(-1)  # Remove last dim
            
            if return_hidden:
                return direction_logits, magnitude_pred, hidden
            return direction_logits, magnitude_pred
        else:
            # Single output (backward compatibility)
            logits = self.fc(hidden)
            if return_hidden:
                return logits, hidden
            return logits

    def predict_proba(self, x):
        """Predict class probabilities (direction only)"""
        with torch.no_grad():
            if self.dual_output:
                direction_logits, _ = self.forward(x)
                probs = self.softmax(direction_logits)
            else:
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
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.1,
        learning_rate: float = 5e-4,
        batch_size: int = 256,
        epochs: int = 300,
        early_stopping_patience: int = 25,
        device: Optional[str] = None,
        l1_lambda: float = 5e-6,
        l2_lambda: float = 5e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        lr_scheduler: str = "cosine_annealing",
        lr_warmup_epochs: int = 10,
        lr_min_factor: float = 0.001,
        lr_patience: int = 5,
        max_grad_norm: float = 0.5,
        gradient_accumulation_steps: int = 2,
        bidirectional: bool = False,
        use_batch_norm: bool = True,
        layer_norm: bool = True,
        spectral_norm: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        noise_std: float = 0.01,
        mixup_alpha: float = 0.1,
        class_weights: str = "balanced",
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
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.lr_scheduler = lr_scheduler
        self.lr_patience = lr_patience
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.class_weights = class_weights
        self.layer_norm = layer_norm
        self.spectral_norm = spectral_norm
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_factor = lr_min_factor
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.bidirectional = bidirectional
        self.use_batch_norm = use_batch_norm

        # Device
        self.device = torch.device(device if device else DEVICE)

        # ZIGZAG: Dual output DISABLED for now - magnitude loss was causing issues
        # The magnitude target wasn't being passed correctly through the OOF pipeline
        # TODO: Re-enable once the pipeline properly passes magnitude through all paths
        self.dual_output = False  # DISABLED: Was causing Val Loss explosion
        # self.dual_output = ENSEMBLE_CONFIG.get('base_learners', {}).get('lstm', {}).get('num_outputs', 1) == 2
        
        # Initialize model with variance reduction features
        self.model = LSTMSequenceClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            bidirectional=bidirectional,
            use_batch_norm=use_batch_norm,
            layer_norm=layer_norm,
            spectral_norm=spectral_norm,
            dual_output=self.dual_output,  # ZIGZAG: Enable dual output
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

        # Variance reduction components
        self.ema_model = None  # EMA model for stable predictions
        if self.use_ema:
            self.ema_model = self._create_ema_model()
        
        # Learning rate scheduler
        self.scheduler = None
        
        # Class weights for imbalanced data
        self.computed_class_weights = None

        # Log model info
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LSTM initialized: {hidden_size} units × {num_layers} layers, "
                    f"BatchNorm={use_batch_norm}, Dropout={dropout}, Params={param_count:,}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_magnitude: Optional[np.ndarray] = None,  # ZIGZAG: Magnitude target
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        y_val_magnitude: Optional[np.ndarray] = None,  # ZIGZAG: Val magnitude
        save_plots_path: Optional[str] = None,
    ):
        """
        Train LSTM model with TRUE LAZY LOADING (no 3D array creation).
        
        ZIGZAG UPDATE: Supports dual targets (direction + magnitude)
        
        Args:
            X: Features (2D array)
            y: Direction target (1D array) - Buy (1) or Sell (0)
            y_magnitude: Magnitude target (1D array) - Distance to extremum (optional)
            X_val: Validation features
            y_val: Validation direction target
            y_val_magnitude: Validation magnitude target (optional)
            save_plots_path: Path to save training plots
        """
        # 1. Scale Data (convert to float32 to save RAM)
        X = X.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X)

        # 2. Create Lazy Datasets (NO 3D Array Creation!)
        # ZIGZAG: Pass magnitude target if available
        if self.dual_output and y_magnitude is not None:
            train_dataset = LazySequenceDataset(X_scaled, y, y_magnitude, self.sequence_length)
        else:
            train_dataset = LazySequenceDataset(X_scaled, y, None, self.sequence_length)

        # Free original arrays
        del X_scaled
        gc.collect()

        # DataLoader - OPTIMIZED FOR 2x T4 GPUs
        # Use workers to keep GPUs fed, pin_memory for fast transfer
        num_workers = GPU_CONFIG.get('num_workers', 4)
        pin_memory = GPU_CONFIG.get(
            'pin_memory', True) and torch.cuda.is_available()
        prefetch_factor = GPU_CONFIG.get(
            'prefetch_factor', 2) if num_workers > 0 else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # CRITICAL: Never shuffle time series
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        )
        logger.info(
            f"DataLoader: batch_size={self.batch_size}, workers={num_workers}, pin_memory={pin_memory}")

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = X_val.astype(np.float32)
            X_val_scaled = self.scaler.transform(X_val)
            
            # ZIGZAG: Pass magnitude target if available
            if self.dual_output and y_val_magnitude is not None:
                val_dataset = LazySequenceDataset(X_val_scaled, y_val, y_val_magnitude, self.sequence_length)
            else:
                val_dataset = LazySequenceDataset(X_val_scaled, y_val, None, self.sequence_length)
            
            val_loader = DataLoader(
                val_dataset,
                # Larger batch for validation (no gradients)
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=num_workers > 0,
            )
            del X_val_scaled
            gc.collect()

        # 3. Class Weights (VARIANCE REDUCTION: Balanced class weights)
        self.computed_class_weights = self._compute_class_weights(np.array(y))
        logger.info(f"Using dynamic class weights: {self.computed_class_weights.cpu().numpy() if self.computed_class_weights is not None else 'None'}")

        # 4. Loss & Optimizer (VARIANCE REDUCTION: Improved optimizer settings)
        # ZIGZAG: Dual loss functions
        criterion_direction = nn.CrossEntropyLoss(
            weight=self.computed_class_weights, label_smoothing=self.label_smoothing)
        criterion_magnitude = nn.MSELoss() if self.dual_output else None
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        # 5. Learning Rate Scheduler (VARIANCE REDUCTION: Advanced scheduling)
        num_training_steps = len(train_loader) * self.epochs
        self.scheduler = self._create_lr_scheduler(optimizer, num_training_steps)
        
        # Warmup scheduler
        def lr_lambda(epoch):
            if epoch < self.lr_warmup_epochs:
                return 0.5 + 0.5 * (epoch / self.lr_warmup_epochs)
            return 1.0
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 6. Training Loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, batch_data in enumerate(train_loader):
                # ZIGZAG: Unpack dual targets if available
                if self.dual_output and len(batch_data) == 3:
                    batch_X, batch_y_dir, batch_y_mag = batch_data
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y_dir = batch_y_dir.to(self.device, non_blocking=True)
                    batch_y_mag = batch_y_mag.to(self.device, non_blocking=True)
                else:
                    batch_X, batch_y_dir = batch_data
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y_dir = batch_y_dir.to(self.device, non_blocking=True)
                    batch_y_mag = None

                # VARIANCE REDUCTION: Apply noise and mixup
                batch_X = self._add_noise_to_inputs(batch_X)
                if self.mixup_alpha > 0:
                    batch_X, y_a, y_b, lam = self._mixup_data(batch_X, batch_y_dir, self.mixup_alpha)
                else:
                    y_a, y_b, lam = batch_y_dir, None, 1.0

                optimizer.zero_grad()

                # Forward pass (with optional AMP)
                if self.use_amp and self.scaler_amp:
                    with autocast('cuda'):
                        if self.dual_output:
                            dir_logits, mag_pred = self.model(batch_X)
                            # Handle mixup for direction loss
                            if y_b is not None:
                                loss_dir = lam * criterion_direction(dir_logits, y_a) + (1 - lam) * criterion_direction(dir_logits, y_b)
                            else:
                                loss_dir = criterion_direction(dir_logits, y_a)
                            
                            if batch_y_mag is not None:
                                loss_mag = criterion_magnitude(mag_pred, batch_y_mag)
                                loss = loss_dir + 0.5 * loss_mag  # Weight magnitude less
                            else:
                                loss = loss_dir
                        else:
                            outputs = self.model(batch_X)
                            # Handle mixup for single output
                            if y_b is not None:
                                loss = lam * criterion_direction(outputs, y_a) + (1 - lam) * criterion_direction(outputs, y_b)
                            else:
                                loss = criterion_direction(outputs, y_a)
                        
                        if self.l1_lambda > 0:
                            l1_penalty = sum(p.abs().sum() for p in self.model.parameters())
                            loss = loss + self.l1_lambda * l1_penalty

                    self.scaler_amp.scale(loss).backward()
                    self.scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm)
                    self.scaler_amp.step(optimizer)
                    self.scaler_amp.update()
                else:
                    if self.dual_output:
                        dir_logits, mag_pred = self.model(batch_X)
                        # Handle mixup for direction loss
                        if y_b is not None:
                            loss_dir = lam * criterion_direction(dir_logits, y_a) + (1 - lam) * criterion_direction(dir_logits, y_b)
                        else:
                            loss_dir = criterion_direction(dir_logits, y_a)
                        
                        if batch_y_mag is not None:
                            loss_mag = criterion_magnitude(mag_pred, batch_y_mag)
                            loss = loss_dir + 0.5 * loss_mag  # Weight magnitude less
                        else:
                            loss = loss_dir
                    else:
                        outputs = self.model(batch_X)
                        # Handle mixup for single output
                        if y_b is not None:
                            loss = lam * criterion_direction(outputs, y_a) + (1 - lam) * criterion_direction(outputs, y_b)
                        else:
                            loss = criterion_direction(outputs, y_a)
                    
                    if self.l1_lambda > 0:
                        l1_penalty = sum(p.abs().sum() for p in self.model.parameters())
                        loss = loss + self.l1_lambda * l1_penalty

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm)
                    optimizer.step()

                total_loss += loss.item()

                # Training accuracy (direction only)
                with torch.no_grad():
                    if self.dual_output:
                        preds = torch.argmax(dir_logits, dim=1)
                    else:
                        preds = torch.argmax(outputs, dim=1)
                    train_correct += (preds == batch_y_dir).sum().item()
                    train_total += batch_y_dir.size(0)

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
                    for batch_data in val_loader:
                        # ZIGZAG: Unpack dual targets if available
                        if self.dual_output and len(batch_data) == 3:
                            batch_X, batch_y_dir, batch_y_mag = batch_data
                            batch_X = batch_X.to(self.device, non_blocking=True)
                            batch_y_dir = batch_y_dir.to(self.device, non_blocking=True)
                            batch_y_mag = batch_y_mag.to(self.device, non_blocking=True)
                        else:
                            batch_X, batch_y_dir = batch_data
                            batch_X = batch_X.to(self.device, non_blocking=True)
                            batch_y_dir = batch_y_dir.to(self.device, non_blocking=True)
                            batch_y_mag = None
                        
                        if self.dual_output:
                            dir_logits, mag_pred = self.model(batch_X)
                            loss_dir = criterion_direction(dir_logits, batch_y_dir)
                            if batch_y_mag is not None:
                                loss_mag = criterion_magnitude(mag_pred, batch_y_mag)
                                loss = loss_dir + 0.5 * loss_mag
                            else:
                                loss = loss_dir
                            preds = torch.argmax(dir_logits, dim=1)
                        else:
                            outputs = self.model(batch_X)
                            loss = criterion_direction(outputs, batch_y_dir)
                            preds = torch.argmax(outputs, dim=1)
                        
                        val_loss += loss.item()
                        val_correct += (preds == batch_y_dir).sum().item()
                        val_total += batch_y_dir.size(0)

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

            # VARIANCE REDUCTION: Update EMA model
            if self.use_ema:
                self._update_ema_model()

            # VARIANCE REDUCTION: Advanced scheduler handling
            if epoch < self.lr_warmup_epochs:
                warmup_scheduler.step()
            elif self.scheduler:
                if self.lr_scheduler == "reduce_on_plateau" and val_loader:
                    self.scheduler.step(avg_val_loss)
                elif self.lr_scheduler == "cosine_annealing":
                    self.scheduler.step()

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
        dataset = LazySequenceDataset(X_scaled, None, y_magnitude=None, sequence_length=self.sequence_length)
        loader = DataLoader(dataset, batch_size=1024,
                            shuffle=False, num_workers=0)

        # VARIANCE REDUCTION: Use EMA model for more stable predictions
        model_to_use = self.ema_model if (self.use_ema and self.ema_model is not None) else self.model
        model_to_use.eval()
        all_probs = []

        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.device)

                # Handle DataParallel wrapper
                if isinstance(model_to_use, nn.DataParallel):
                    logits = model_to_use(batch_X)
                    probs = torch.softmax(logits, dim=1)
                else:
                    probs = model_to_use.predict_proba(batch_X)

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

        dataset = LazySequenceDataset(X_scaled, None, y_magnitude=None, sequence_length=self.sequence_length)
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

    def _create_ema_model(self):
        """Create EMA model for stable predictions."""
        import copy
        ema_model = copy.deepcopy(self.model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model

    def _update_ema_model(self):
        """Update EMA model weights."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def _compute_class_weights(self, y: np.ndarray):
        """Compute balanced class weights."""
        if self.class_weights == "balanced":
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            return torch.FloatTensor(weights).to(self.device)
        return None

    def _create_lr_scheduler(self, optimizer, num_training_steps):
        """Create learning rate scheduler."""
        if self.lr_scheduler == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_training_steps, eta_min=self.learning_rate * self.lr_min_factor
            )
        elif self.lr_scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=self.lr_patience, verbose=True
            )
        return None

    def _add_noise_to_inputs(self, x):
        """Add small noise for regularization."""
        if self.noise_std > 0 and self.model.training:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x

    def _mixup_data(self, x, y, alpha=0.1):
        """Apply mixup augmentation."""
        if alpha > 0 and self.model.training:
            lam = np.random.beta(alpha, alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)
            
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam
        return x, y, None, 1.0

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
            'recurrent_dropout': self.recurrent_dropout,
            'bidirectional': self.bidirectional,
            'use_batch_norm': self.use_batch_norm,
            'layer_norm': self.layer_norm,
            'spectral_norm': self.spectral_norm,
        }, f"{path}_lstm.pt")

        # Save scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        logger.info(f"LSTM model saved to {path}_lstm.pt")

    def load_model(self, path: str):
        """Load model state and scaler."""
        import joblib

        # Load checkpoint
        checkpoint = torch.load(f"{path}_lstm.pt", map_location=self.device)

        # Reinitialize model with saved params (with backward compatibility)
        self.model = LSTMSequenceClassifier(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            num_classes=checkpoint['num_classes'],
            dropout=checkpoint['dropout'],
            recurrent_dropout=checkpoint.get('recurrent_dropout', 0.1),
            bidirectional=checkpoint['bidirectional'],
            use_batch_norm=checkpoint['use_batch_norm'],
            layer_norm=checkpoint.get('layer_norm', True),
            spectral_norm=checkpoint.get('spectral_norm', True),
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

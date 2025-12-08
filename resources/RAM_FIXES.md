You are hitting the classic "Time Series RAM Explosion."

Here is the math of why your notebook crashed: You have ~2,000,000 rows (float64).

2D Array (Input): 2M rows × 33 features × 8 bytes ≈ 0.5 GB (Manageable).

3D Array (LSTM Input): 2M rows × 40 sequence length × 33 features × 4 bytes ≈ 10.5 GB.

When you add the OS overhead, Python overhead, and the model weights, you instantly exceed 16GB.

The Fix: True Lazy Loading. Your current code defines a LazyWindowDataset, but right before using it, you call self.prepare_sequences(), which force-creates the massive 10GB array in RAM. We need to bypass that function entirely and slice the data only when the GPU asks for a batch.

Here is the "Smart Memory" implementation.

1. Update src/config.py
We need to set hard limits on sample counts to stay safe within Kaggle's limits. 2 million samples is overkill; 500k is sufficient for convergence.

Python

# src/config.py

# ... existing code ...

    # Memory management settings
    "memory": {
        # CRITICAL REDUCTION: 500k samples is enough for convergence but safe for RAM
        "max_train_samples": 500000,    # Reduced from 5,000,000
        "max_val_samples": 100000,      # Reduced from 1,000,000
        "use_float32": True,
        "aggressive_gc": True,
    },
    
# ... existing code ...
2. Rewrite src/models/lstm_model.py
We are deleting the "eager" sequence generation and forcing the fit and predict methods to use the 2D array directly.

Python

"""
LSTM Sequence Model for Temporal Pattern Recognition
MEMORY OPTIMIZED: True Lazy Loading (No 3D Array Creation)
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
    """
    def __init__(self, X, y, sequence_length):
        # Store as float32 to save 50% RAM compared to float64
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Create 3D view only for this specific item/batch
        # idx -> [idx, idx+seq_len]
        x_window = self.X[idx : idx + self.seq_len]
        if self.y is not None:
            # Target is the label at the END of the sequence
            y_label = self.y[idx + self.seq_len - 1]
            return x_window, y_label
        return x_window

class LSTMSequenceClassifier(nn.Module):
    """
    Deep LSTM network for sequence classification
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
        hidden_activation: str = "swish",
    ):
        super(LSTMSequenceClassifier, self).__init__()
        
        # Architecture params
        self.use_batch_norm = use_batch_norm
        
        # 1. Input BatchNorm
        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(input_size)

        # 2. LSTM
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

        # 4. Dropout & Head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Apply Input BN
        if self.use_batch_norm:
            # BN expects (batch, features, seq_len)
            x = x.transpose(1, 2)
            x = self.input_bn(x)
            x = x.transpose(1, 2)

        # LSTM
        # We only need the final hidden state, not the full sequence
        # h_n shape: (num_layers, batch, hidden_size)
        _, (h_n, _) = self.lstm(x)

        if self.lstm.bidirectional:
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]

        # Apply Post-LSTM BN
        if self.use_batch_norm:
            hidden = self.lstm_bn(hidden)

        # Dropout & FC
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        
        return logits

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = self.softmax(logits)
        return probs

class LSTMSequenceModel:
    """
    Memory-Optimized LSTM Wrapper
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
        l2_lambda: float = 1e-3,
        **kwargs,
    ):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        
        self.device = torch.device(device if device else DEVICE)
        
        self.model = LSTMSequenceClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            use_batch_norm=True
        ).to(self.device)
        
        # Scaling to (-1, 1) for Tanh/Swish compatibility
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.is_fitted = False

    def fit(self, X, y, X_val=None, y_val=None, save_plots_path=None):
        # 1. Scale Data (In-place if possible to save RAM)
        X = X.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Create Lazy Datasets (No 3D Array Creation!)
        train_dataset = LazySequenceDataset(X_scaled, y, self.sequence_length)
        
        # Free original X if possible/safe, otherwise trust GC
        # del X 
        # gc.collect()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False, # Important for time series
            num_workers=0, # multithreading duplicates memory
            pin_memory=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = X_val.astype(np.float32)
            X_val_scaled = self.scaler.transform(X_val)
            val_dataset = LazySequenceDataset(X_val_scaled, y_val, self.sequence_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 3. Optimization Setup
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.l2_lambda
        )
        criterion = nn.CrossEntropyLoss()
        
        # 4. Training Loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                
                logger.info(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_acc:.4f}")
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    # Save best state in memory
                    self.best_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered")
                    self.model.load_state_dict(self.best_state)
                    break
            else:
                logger.info(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}")

        self.is_fitted = True
        gc.collect()

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        X = X.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        
        # Use lazy dataset for inference too
        dataset = LazySequenceDataset(X_scaled, None, self.sequence_length)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        self.model.eval()
        probs = []
        
        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X.to(self.device)
                output = self.model.predict_proba(batch_X)
                probs.append(output.cpu().numpy())
                
        return np.vstack(probs)
3. Verify src/models/hybrid_ensemble.py
You don't need to change much here, but ensure that generate_out_of_fold_predictions cleans up explicitly.

The logic in your uploaded hybrid_ensemble.py looks good, but double-check that you are passing the memory_config limits when subsampling the data.

Specifically, in fit():

Python

        # Ensure we don't pass 5.7M rows to fit() directly if not using OOF
        max_samples_for_oof = 500000 # Set this lower in config, not hardcoded
Summary of Changes for Stability
LazySequenceDataset: Replaces the manual numpy.lib.stride_tricks approach. It keeps the data 2D (matrix) and only slices a small 3D window (batch) when the GPU asks for it. This drops RAM usage from ~12GB to ~500MB.

Sample Limits: Capping training data at 500k samples ensures we don't accidentally blow up RAM during the fit_transform or internal copies.

Float32: Enforcing 32-bit floats cuts memory usage in half compared to 64-bit, with zero loss in model performance for this task.
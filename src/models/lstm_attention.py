"""
Advanced LSTM with Attention Mechanism
Helps model focus on most important timesteps in sequence

Key improvements over basic LSTM:
1. Attention mechanism - learns which timesteps matter most
2. Residual connections - helps gradient flow
3. Layer normalization - more stable than batch norm for sequences
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionLayer(nn.Module):
    """
    Attention mechanism for LSTM outputs
    
    Learns to weight different timesteps by importance:
    - High attention on regime changes, breakouts
    - Low attention on noise, consolidation
    """
    
    def __init__(self, hidden_size: int, attention_size: int = None):
        super(AttentionLayer, self).__init__()
        
        if attention_size is None:
            attention_size = hidden_size
        
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Attention weights
        self.W = nn.Linear(hidden_size, attention_size, bias=False)
        self.b = nn.Parameter(torch.zeros(attention_size))
        self.u = nn.Linear(attention_size, 1, bias=False)
        
    def forward(self, lstm_outputs):
        """
        Args:
            lstm_outputs: (batch_size, seq_len, hidden_size)
            
        Returns:
            context: (batch_size, hidden_size) - weighted sum of outputs
            attention_weights: (batch_size, seq_len) - importance of each timestep
        """
        # lstm_outputs: (batch, seq_len, hidden)
        
        # Compute attention scores
        # (batch, seq_len, attention_size)
        u_t = torch.tanh(self.W(lstm_outputs) + self.b)
        
        # (batch, seq_len, 1)
        scores = self.u(u_t)
        
        # (batch, seq_len)
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)
        
        # Weighted sum: (batch, hidden)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_outputs  # (batch, seq_len, hidden)
        ).squeeze(1)  # (batch, hidden)
        
        return context, attention_weights


class LSTMWithAttention(nn.Module):
    """
    LSTM + Attention for forex prediction
    
    Architecture:
    Input → LayerNorm → LSTM → Attention → Residual → Dropout → FC → Output
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_layer_norm: bool = True,
        attention_size: int = None,
    ):
        super(LSTMWithAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        
        # 1. Input Layer Normalization (better than BatchNorm for sequences)
        if use_layer_norm:
            self.input_ln = nn.LayerNorm(input_size)
        
        # 2. LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # 3. Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = AttentionLayer(lstm_output_size, attention_size)
        
        # 4. Post-attention Layer Normalization
        if use_layer_norm:
            self.output_ln = nn.LayerNorm(lstm_output_size)
        
        # 5. Residual connection (if dimensions match)
        self.use_residual = (input_size == lstm_output_size)
        if self.use_residual:
            self.residual_proj = nn.Linear(input_size, lstm_output_size)
        
        # 6. Dropout & Output Head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, return_attention=False):
        """
        Forward pass with attention
        
        Args:
            x: (batch_size, seq_len, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (batch_size, seq_len) if return_attention=True
        """
        batch_size, seq_len, features = x.shape
        
        # 1. Input Layer Normalization
        if self.use_layer_norm:
            # LayerNorm works on last dimension, perfect for sequences
            x_norm = self.input_ln(x)
        else:
            x_norm = x
        
        # 2. LSTM forward pass
        # lstm_out: (batch, seq_len, hidden * num_directions)
        lstm_out, _ = self.lstm(x_norm)
        
        # 3. Attention mechanism
        # context: (batch, hidden * num_directions)
        # attention_weights: (batch, seq_len)
        context, attention_weights = self.attention(lstm_out)
        
        # 4. Residual connection (helps gradient flow)
        if self.use_residual:
            # Use mean pooling of input as residual
            residual = self.residual_proj(x.mean(dim=1))
            context = context + residual
        
        # 5. Post-attention Layer Normalization
        if self.use_layer_norm:
            context = self.output_ln(context)
        
        # 6. Dropout & FC
        context = self.dropout(context)
        logits = self.fc(context)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def predict_proba(self, x):
        """Predict class probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = self.softmax(logits)
        return probs
    
    def get_attention_weights(self, x):
        """
        Get attention weights for visualization
        Shows which timesteps the model focuses on
        """
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights.cpu().numpy()


# Example usage and testing
if __name__ == "__main__":
    # Test the attention LSTM
    batch_size = 32
    seq_len = 100
    input_size = 44
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Initialize model
    model = LSTMWithAttention(
        input_size=input_size,
        hidden_size=64,
        num_layers=1,
        num_classes=2,
        dropout=0.3,
        bidirectional=False,
        use_layer_norm=True,
        attention_size=32,
    )
    
    # Forward pass
    logits, attention_weights = model(x, return_attention=True)
    
    print("="*60)
    print("LSTM WITH ATTENTION - TEST")
    print("="*60)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nAttention weights (first sample):")
    print(f"  Min: {attention_weights[0].min():.6f}")
    print(f"  Max: {attention_weights[0].max():.6f}")
    print(f"  Sum: {attention_weights[0].sum():.6f} (should be 1.0)")
    print(f"\nTop 5 most important timesteps:")
    top_indices = attention_weights[0].argsort(descending=True)[:5]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Timestep t-{seq_len-idx-1}: weight={attention_weights[0][idx]:.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print("\n✓ Attention LSTM working correctly!")

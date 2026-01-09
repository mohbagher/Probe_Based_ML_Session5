"""
Advanced Model Architectures for RIS Probe-Based ML System.

Implements 8 complete model architectures:
1. MLP - Multi-Layer Perceptron (baseline)
2. CNN - 1D Convolutional Neural Network
3. LSTM - Bidirectional Long Short-Term Memory
4. GRU - Bidirectional Gated Recurrent Unit
5. Attention MLP - MLP with multi-head attention
6. Transformer - Full transformer encoder
7. ResNet MLP - MLP with residual connections
8. Hybrid CNN-LSTM - Combined architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class AdvancedMLP(nn.Module):
    """
    Enhanced Multi-Layer Perceptron with configurable architecture.
    
    Input: [masked_powers, mask] ∈ ℝ^{2K}
    Output: logits ∈ ℝ^K
    """
    
    def __init__(self, K: int, hidden_sizes: List[int] = [512, 256, 128],
                 dropout_prob: float = 0.1, use_batch_norm: bool = True):
        super().__init__()
        self.K = K
        self.input_size = 2 * K
        
        layers = []
        prev_size = self.input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, K))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for probe sequences.
    
    Treats the K probes as a 1D sequence with 2 channels (power, mask).
    Uses convolutions to capture local patterns in probe space.
    """
    
    def __init__(self, K: int, num_filters: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [5, 5, 3], dropout_prob: float = 0.1):
        super().__init__()
        self.K = K
        
        # Reshape input from (batch, 2K) to (batch, 2, K)
        conv_layers = []
        in_channels = 2
        
        for num_filter, kernel_size in zip(num_filters, kernel_sizes):
            conv_layers.append(nn.Conv1d(in_channels, num_filter, kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(2))
            conv_layers.append(nn.Dropout(dropout_prob))
            in_channels = num_filter
        
        self.conv_net = nn.Sequential(*conv_layers)
        
        # Calculate the size after convolutions and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, K)
            conv_output = self.conv_net(dummy_input)
            flattened_size = conv_output.view(1, -1).shape[1]
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, K)
        )
    
    def forward(self, x):
        # x shape: (batch, 2K)
        batch_size = x.shape[0]
        # Reshape to (batch, 2, K): [powers, mask]
        x = x.view(batch_size, 2, self.K)
        x = self.conv_net(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for sequential probe modeling.
    
    Treats probes as a sequence, using bidirectional LSTM to capture
    dependencies in both directions.
    """
    
    def __init__(self, K: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout_prob: float = 0.1):
        super().__init__()
        self.K = K
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input: (batch, K, 2) - treat each probe as timestep with 2 features
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Output from bidirectional LSTM: 2 * hidden_size
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size * K, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, K)
        )
    
    def forward(self, x):
        # x shape: (batch, 2K)
        batch_size = x.shape[0]
        # Reshape to (batch, K, 2): each probe is a timestep with [power, mask]
        x = x.view(batch_size, self.K, 2)
        
        # LSTM output: (batch, K, 2*hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Flatten for fully connected layers
        x = lstm_out.reshape(batch_size, -1)
        x = self.fc(x)
        return x


class BiGRU(nn.Module):
    """
    Bidirectional GRU for sequential probe modeling.
    
    Similar to LSTM but with simpler gating mechanism (faster training).
    """
    
    def __init__(self, K: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout_prob: float = 0.1):
        super().__init__()
        self.K = K
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size * K, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, K)
        )
    
    def forward(self, x):
        # x shape: (batch, 2K)
        batch_size = x.shape[0]
        # Reshape to (batch, K, 2)
        x = x.view(batch_size, self.K, 2)
        
        # GRU output: (batch, K, 2*hidden_size)
        gru_out, _ = self.gru(x)
        
        # Flatten and classify
        x = gru_out.reshape(batch_size, -1)
        x = self.fc(x)
        return x


class AttentionMLP(nn.Module):
    """
    MLP with Multi-Head Attention mechanism.
    
    Uses attention to focus on important probe features before classification.
    """
    
    def __init__(self, K: int, d_model: int = 256, num_heads: int = 4,
                 hidden_sizes: List[int] = [512, 256], dropout_prob: float = 0.1):
        super().__init__()
        self.K = K
        self.d_model = d_model
        
        # Project input to d_model dimensions
        self.input_proj = nn.Linear(2 * K, d_model)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        
        # Feed-forward network
        fc_layers = []
        prev_size = d_model
        for hidden_size in hidden_sizes:
            fc_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            prev_size = hidden_size
        fc_layers.append(nn.Linear(prev_size, K))
        
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        # x shape: (batch, 2K)
        batch_size = x.shape[0]
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, d_model)
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        
        # Self-attention
        attn_output, _ = self.attention(x, x, x)  # (batch, 1, d_model)
        x = attn_output.squeeze(1)  # (batch, d_model)
        
        # Classification
        x = self.fc(x)
        return x


class TransformerModel(nn.Module):
    """
    Transformer Encoder for probe sequence modeling.
    
    Uses multi-head self-attention to capture complex relationships
    between probes. State-of-the-art architecture.
    """
    
    def __init__(self, K: int, d_model: int = 256, num_heads: int = 8,
                 num_layers: int = 3, dim_feedforward: int = 512,
                 dropout_prob: float = 0.1):
        super().__init__()
        self.K = K
        self.d_model = d_model
        
        # Input embedding: project 2 features to d_model
        self.input_embedding = nn.Linear(2, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(K, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, K)
        )
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, x):
        # x shape: (batch, 2K)
        batch_size = x.shape[0]
        
        # Reshape to (batch, K, 2): each probe is a token with 2 features
        x = x.view(batch_size, self.K, 2)
        
        # Embed tokens
        x = self.input_embedding(x)  # (batch, K, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :self.K, :].to(x.device)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, K, d_model)
        
        # Global pooling (mean over sequence)
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        x = self.fc(x)
        return x


class ResBlock(nn.Module):
    """Residual block for ResNet-style MLP."""
    
    def __init__(self, size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.relu(out)
        return out


class ResNetMLP(nn.Module):
    """
    ResNet-style MLP with skip connections.
    
    Enables training of very deep networks through residual connections.
    """
    
    def __init__(self, K: int, hidden_size: int = 512, num_blocks: int = 4,
                 dropout_prob: float = 0.1):
        super().__init__()
        self.K = K
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(2 * K, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_size, dropout_prob) for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, K)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_proj(x)
        return x


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM architecture.
    
    Combines CNN for local feature extraction with LSTM for sequential modeling.
    Best for time-varying or structured probe patterns.
    """
    
    def __init__(self, K: int, num_filters: int = 64, kernel_size: int = 5,
                 lstm_hidden: int = 128, lstm_layers: int = 2, dropout_prob: float = 0.1):
        super().__init__()
        self.K = K
        
        # CNN layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, num_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.Dropout(dropout_prob),
            nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.Dropout(dropout_prob)
        )
        
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if lstm_layers > 1 else 0
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, K)
        )
    
    def forward(self, x):
        # x shape: (batch, 2K)
        batch_size = x.shape[0]
        
        # Reshape to (batch, 2, K) for CNN
        x = x.view(batch_size, 2, self.K)
        
        # CNN feature extraction
        x = self.conv_layers(x)  # (batch, num_filters, K)
        
        # Reshape for LSTM: (batch, K, num_filters)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, K, 2*lstm_hidden)
        
        # Use last hidden state
        x = lstm_out[:, -1, :]  # (batch, 2*lstm_hidden)
        
        # Classification
        x = self.fc(x)
        return x


def create_advanced_model(model_type: str, K: int, config: dict = None):
    """
    Factory function to create advanced model architectures.
    
    Args:
        model_type: Type of model to create
        K: Number of probes (output size)
        config: Optional configuration dictionary with model-specific parameters
        
    Returns:
        PyTorch model instance
    """
    if config is None:
        config = {}
    
    model_type = model_type.lower()
    
    if model_type == "mlp":
        return AdvancedMLP(
            K=K,
            hidden_sizes=config.get('hidden_sizes', [512, 256, 128]),
            dropout_prob=config.get('dropout_prob', 0.1),
            use_batch_norm=config.get('use_batch_norm', True)
        )
    
    elif model_type == "cnn":
        return CNN1D(
            K=K,
            num_filters=config.get('num_filters', [32, 64, 128]),
            kernel_sizes=config.get('kernel_sizes', [5, 5, 3]),
            dropout_prob=config.get('dropout_prob', 0.1)
        )
    
    elif model_type == "lstm":
        return BiLSTM(
            K=K,
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            dropout_prob=config.get('dropout_prob', 0.1)
        )
    
    elif model_type == "gru":
        return BiGRU(
            K=K,
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            dropout_prob=config.get('dropout_prob', 0.1)
        )
    
    elif model_type == "attention":
        return AttentionMLP(
            K=K,
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 4),
            hidden_sizes=config.get('hidden_sizes', [512, 256]),
            dropout_prob=config.get('dropout_prob', 0.1)
        )
    
    elif model_type == "transformer":
        return TransformerModel(
            K=K,
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 3),
            dim_feedforward=config.get('dim_feedforward', 512),
            dropout_prob=config.get('dropout_prob', 0.1)
        )
    
    elif model_type == "resnet":
        return ResNetMLP(
            K=K,
            hidden_size=config.get('hidden_size', 512),
            num_blocks=config.get('num_blocks', 4),
            dropout_prob=config.get('dropout_prob', 0.1)
        )
    
    elif model_type == "hybrid":
        return HybridCNNLSTM(
            K=K,
            num_filters=config.get('num_filters', 64),
            kernel_size=config.get('kernel_size', 5),
            lstm_hidden=config.get('lstm_hidden', 128),
            lstm_layers=config.get('lstm_layers', 2),
            dropout_prob=config.get('dropout_prob', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: mlp, cnn, lstm, gru, attention, transformer, resnet, hybrid")

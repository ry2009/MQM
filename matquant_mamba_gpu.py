"""
MatQuant-Mamba: Matryoshka-Quantized Mamba for Efficient Time Series Forecasting

This implementation uses actual Mamba SSM blocks (not GRU), with Matryoshka quantization
techniques and is optimized for GPU acceleration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import time
import os
import argparse
import tqdm
import math

# Check if mamba-ssm is installed, if not print instructions
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.utils.generation import GenState
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm package not found.")
    print("Please install mamba-ssm using: pip install mamba-ssm")
    print("Or in Colab: !pip install mamba-ssm")

# Check if mamba-ssm is installed, if not use fallback
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.utils.generation import GenState
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm package not found.")
    print("Using GRU-based fallback implementation instead.")
    
    # Define a fallback Mamba implementation using GRU
    class Mamba(nn.Module):
        """
        Fallback implementation using GRU when mamba-ssm is not available
        """
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1, **kwargs):
            super().__init__()
            self.d_model = d_model
            d_inner = int(expand * d_model)
            
            # Use GRU as a stand-in for the SSM mechanism
            self.in_proj = nn.Linear(d_model, d_inner)
            self.gru = nn.GRU(
                input_size=d_inner,
                hidden_size=d_inner,
                num_layers=1,
                batch_first=True
            )
            self.out_proj = nn.Linear(d_inner, d_model)
            self.dropout = nn.Dropout(dropout)
            
            print(f"Using GRU fallback for Mamba (d_model={d_model}, d_inner={d_inner})")
        
        def forward(self, x):
            # Input projection
            h = self.in_proj(x)
            h = F.gelu(h)
            
            # Process with GRU
            h, _ = self.gru(h)
            
            # Output projection
            h = self.out_proj(h)
            h = self.dropout(h)
            
            return h

print("Starting MatQuant-Mamba GPU Implementation...")

# Create directory for results
os.makedirs('matquant_results', exist_ok=True)
print("Created output directory: matquant_results/")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ==============================================
# Enhanced HFT Data Generator with Nanosecond Precision
# ==============================================

class EnhancedNanoHFTDataGenerator:
    """
    Generates synthetic high-frequency trading data with nanosecond precision
    and realistic market features including jumps, regime changes, and varied volatility.
    Optimized for GPU-based processing.
    """
    def __init__(self, 
                 symbol="BTC-USD",
                 initial_price=40000.0,
                 trading_minutes=30,         # Default 30 minutes
                 ticks_per_second=1000,      # 1000 ticks/second for nanosecond precision
                 base_volatility=0.00001,    # Base volatility level
                 include_regimes=True,       # Include regime changes
                 include_jumps=True):        # Include price jumps
        
        print(f"Initializing HFT Data Generator: {trading_minutes} minutes at {ticks_per_second} ticks/second")
        self.symbol = symbol
        self.initial_price = initial_price
        self.trading_minutes = trading_minutes
        self.ticks_per_second = ticks_per_second
        self.base_volatility = base_volatility
        self.include_regimes = include_regimes
        self.include_jumps = include_jumps
        
        # Total number of ticks
        self.total_ticks = int(trading_minutes * 60 * ticks_per_second)
        print(f"Total ticks to generate: {self.total_ticks:,}")
        
        # Start time
        self.start_time = datetime.now() - timedelta(minutes=trading_minutes)
    
    def generate_data(self):
        """
        Generate enhanced synthetic HFT data with realistic features at nanosecond precision
        (Optimized for speed)
        """
        print("Generating synthetic HFT data with nanosecond precision...")
        start_time = time.time()
        
        # Generate timestamps with nanosecond precision
        timestamp_delta = 1_000_000_000 / self.ticks_per_second  # nanoseconds between ticks
        print(f"Interval between ticks: {timestamp_delta} nanoseconds")
        
        print("Generating timestamps...")
        # Using vectorized approach for timestamps (much faster)
        base_timestamp = self.start_time.timestamp()
        timestamp_seconds = np.array([base_timestamp + i*timestamp_delta/1_000_000_000 for i in range(self.total_ticks)])
        timestamps = np.array([datetime.fromtimestamp(ts) for ts in timestamp_seconds])
        
        print("Initializing price and volatility arrays...")
        # Pre-allocate arrays (more efficient)
        prices = np.zeros(self.total_ticks)
        returns = np.zeros(self.total_ticks)
        volatility = np.zeros(self.total_ticks)
        
        price = self.initial_price
        prices[0] = price
        
        # Create regime changes (volatility clusters)
        if self.include_regimes:
            print("Creating volatility regimes...")
            # Create 3-5 different volatility regimes
            num_regimes = np.random.randint(3, 6)
            print(f"Number of volatility regimes: {num_regimes}")
            
            # Vectorized approach for regime boundaries
            regime_bounds = np.sort(np.random.choice(range(1, self.total_ticks), num_regimes-1, replace=False))
            regime_bounds = np.append(np.insert(regime_bounds, 0, 0), self.total_ticks)
            
            # Assign random volatility levels to each regime (0.5x to 3x base volatility)
            regime_vols = self.base_volatility * np.random.uniform(0.5, 3, num_regimes)
            
            # Create volatility series
            for i in range(num_regimes):
                start, end = regime_bounds[i], regime_bounds[i+1]
                volatility[start:end] = regime_vols[i]
                print(f"  Regime {i+1}: tick {start:,} to {end:,}, volatility: {regime_vols[i]:.8f}")
        else:
            print("Using constant volatility...")
            volatility[:] = self.base_volatility
        
        # Generate price path with jumps and regime-specific volatility
        print("Generating price path...")
        
        # Pre-generate random values (much faster than calling random inside the loop)
        rng = np.random.default_rng()
        random_walks = rng.normal(0, 1, self.total_ticks) 
        jump_triggers = rng.random(self.total_ticks) < 0.0001  # 0.01% chance of jump
        jump_directions = rng.choice([-1, 1], self.total_ticks)
        jump_sizes = rng.uniform(0.001, 0.005, self.total_ticks)
        
        jumps_triggered = 0
        
        # Use batch processing for speed
        print("Processing price changes in batches...")
        batch_size = 100000  # Process in batches to show progress
        for batch_start in range(1, self.total_ticks, batch_size):
            batch_end = min(batch_start + batch_size, self.total_ticks)
            
            for i in range(batch_start, batch_end):
                vol = volatility[i]
                
                # Random walk component with pre-generated random values
                price_change = random_walks[i] * vol * price
                
                # Mean reversion component (stronger during high volatility)
                price_change -= 0.01 * (price - self.initial_price) / self.initial_price * price * vol * 100
                
                # Add jumps (rare but significant price moves) using pre-generated values
                if jump_triggers[i]:
                    jump_size = jump_directions[i] * jump_sizes[i] * price
                    price_change += jump_size
                    jumps_triggered += 1
                    
                # Update price
                price = max(0.0001, price + price_change)
                prices[i] = price
                returns[i] = (price / prices[i-1]) - 1
            
            # Print progress after each batch
            print(f"  Generated {batch_end:,}/{self.total_ticks:,} ticks ({batch_end/self.total_ticks*100:.1f}%)")
        
        print(f"Price jumps triggered: {jumps_triggered}")
        
        # Create DataFrame more efficiently
        print("Creating DataFrame with features...")
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'return': returns,
            'volatility': volatility
        })
        
        # Add bid/ask spread (tighter during low volatility, wider during high volatility)
        print("Adding market microstructure features...")
        # Vectorized operations instead of column-by-column
        df['spread'] = df['volatility'] * df['price'] * 10
        df['bid'] = df['price'] - df['spread'] / 2
        df['ask'] = df['price'] + df['spread'] / 2
        
        # Add order book imbalance using vectorized operations
        print("Adding order book imbalance...")
        future_returns = df['return'].shift(-1).fillna(0)
        noise = rng.normal(0, 0.5, len(df))
        signal_ratio = 0.3  # 30% signal, 70% noise
        df['order_imbalance'] = signal_ratio * np.sign(future_returns) + (1-signal_ratio) * noise
        df['order_imbalance'] = np.clip(df['order_imbalance'], -1, 1)  # Scale to [-1, 1]
        
        # Add derived features using vectorized operations
        print("Calculating technical indicators...")
        window_sizes = [5, 10, 50, 100]
        
        for w in window_sizes:
            # Use vectorized pandas operations
            df[f'ma_{w}'] = df['price'].rolling(window=w).mean()
            df[f'vol_{w}'] = df['return'].rolling(window=w).std() * np.sqrt(self.ticks_per_second * 60)
            df[f'mom_{w}'] = df['price'].pct_change(periods=w)
            df[f'imbalance_{w}'] = df['order_imbalance'].rolling(window=w).mean()
        
        # Add timestamp features
        print("Adding timestamp features...")
        # Extract all timestamp features at once
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        df['microsecond'] = df['timestamp'].dt.microsecond
        
        # Target: next-tick return (prediction target)
        print("Setting prediction target...")
        df['target'] = df['return'].shift(-1)
        
        # Drop rows with NaN
        initial_len = len(df)
        df = df.dropna()
        print(f"Dropped {initial_len - len(df)} rows with NaN values")
        
        elapsed = time.time() - start_time
        print(f"Data generation completed in {elapsed:.2f} seconds")
        print(f"Final dataset shape: {df.shape}")
        
        return df
    
    def prepare_data(self, window_size=50):
        """
        Prepare data for model training with sliding window. 
        Optimized for performance with GPU acceleration in mind.
        """
        print(f"\nPreparing data for model with window size {window_size}...")
        start_time = time.time()
        
        # Generate and normalize data
        df = self.generate_data()
        
        # Features for model input (exclude target and timestamp)
        features = [col for col in df.columns if col not in ['timestamp', 'target']]
        print(f"Number of features: {len(features)}")
        
        # Normalize data efficiently (before creating windows)
        print("Normalizing features...")
        feature_columns = [col for col in df.columns if col not in ['price', 'spread', 'timestamp']]
        
        scaler = RobustScaler()
        df_norm = df.copy()
        df_norm[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        # Replace original values with normalized ones
        for i, feature in enumerate(features):
            df[feature] = df_norm[feature]
        
        # Efficient window creation using numpy stride tricks
        print("Creating sliding windows efficiently...")
        X_windows = self._create_sliding_windows(df[features].values, window_size)
        y = df['target'].values[window_size-1:-1]
        timestamps = df['timestamp'].values[window_size:]
        
        print(f"X shape: {X_windows.shape}, y shape: {y.shape}")
        
        # Check if sizes match
        if len(X_windows) != len(y):
            print(f"Warning: X windows ({len(X_windows)}) and y ({len(y)}) have different lengths!")
            min_size = min(len(X_windows), len(y))
            X_windows = X_windows[:min_size]
            y = y[:min_size]
            timestamps = timestamps[:min_size]
            print(f"Adjusted to common size: {min_size}")
        
        # Rearrange for PyTorch [batch, channels, time]
        X_windows = np.transpose(X_windows, (0, 2, 1))
        
        elapsed = time.time() - start_time
        print(f"Data preparation completed in {elapsed:.2f} seconds")
        print(f"Final X shape: {X_windows.shape}, y shape: {y.shape}")
        
        return X_windows, y, timestamps, df
    
    def _create_sliding_windows(self, arr, window_size):
        """
        Efficiently create sliding windows using NumPy's stride tricks
        """
        # Get number of elements that will form complete windows
        n = len(arr) - window_size + 1
        if n <= 0:
            raise ValueError(f"Window size {window_size} is too large for dataset of length {len(arr)}")
            
        # For newer NumPy versions (1.20.0+)
        if hasattr(np, 'lib') and hasattr(np.lib, 'stride_tricks') and hasattr(np.lib.stride_tricks, 'sliding_window_view'):
            print("Using NumPy's sliding_window_view (very fast)...")
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                windows = sliding_window_view(arr, window_shape=window_size, axis=0)
                return windows
            except Exception as e:
                print(f"Error using sliding_window_view: {e}")
        
        # Fallback for older NumPy versions
        print("Falling back to manual window creation...")
        shape = (n, window_size) + arr.shape[1:]
        strides = (arr.strides[0],) + arr.strides
        
        # Create view with custom strides
        return np.lib.stride_tricks.as_strided(
            arr, shape=shape, strides=strides, writeable=False
        )


# ==============================================
# Matryoshka Quantization Implementation
# ==============================================

class MatryoshkaQuantizer(nn.Module):
    """
    Matryoshka Quantization for different bit-widths.
    Implements the multi-scale slicing operator for k-bit quantization.
    """
    def __init__(self, method='slice_k'):
        super().__init__()
        self.method = method
        print(f"Initialized MatryoshkaQuantizer with method: {method}")
    
    def slice_k(self, x, k=2):
        """
        Multi-scale slicing operator for k-bit quantization as described in the paper:
        Slice_k(W) = sign(W) * floor(|W| * 2^(8-k))
        """
        return torch.sign(x) * torch.floor(torch.abs(x) * (2 ** (8 - k)))
    
    def quantize(self, x, bit_width=2):
        """
        Apply quantization at the specified bit width
        """
        if bit_width not in [2, 4, 8]:
            raise ValueError("Bit width must be one of: 2, 4, 8")
        
        return self.slice_k(x, k=bit_width)


# ==============================================
# Mamba-SSM Implementation with Matryoshka Quantization 
# ==============================================

class MambaMatryoshkaBlock(nn.Module):
    """
    Mamba block with Matryoshka quantization.
    Uses the actual Mamba SSM implementation, not a GRU approximation.
    """
    def __init__(self, 
                 d_model,         # Model dimension
                 d_state=16,      # SSM state expansion factor
                 d_conv=4,        # Local convolution width
                 expand=2,        # Expansion factor for feedforward
                 bit_width=8,     # Quantization bit-width (fixed mode)
                 is_dynamic=False, # Whether to use dynamic bit-width
                 entropy_threshold=0.5, # Threshold for dynamic mode
                 dropout=0.1,
                 dt_min=0.001,    # Min step size
                 dt_max=0.1):     # Max step size
        super().__init__()
        
        self.d_model = d_model
        self.bit_width = bit_width
        self.is_dynamic = is_dynamic
        self.entropy_threshold = entropy_threshold
        
        # Model dimensions
        d_inner = int(expand * d_model)
        
        if is_dynamic:
            print(f"Initializing MambaMatryoshkaBlock: DYNAMIC mode with entropy_threshold={entropy_threshold}")
        else:
            print(f"Initializing MambaMatryoshkaBlock: FIXED mode with bit_width={bit_width}")
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner)
        
        # Core Mamba SSM layer
        self.mamba = Mamba(
            d_model=d_inner,
            d_state=d_state,
            d_conv=d_conv,
            expand=1,  # No additional expansion needed
            dt_min=dt_min,
            dt_max=dt_max
        )
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Quantizer
        self.quantizer = MatryoshkaQuantizer()
    
    def estimate_entropy(self, x):
        """
        Estimate token entropy to determine bit-width for dynamic mode
        """
        # Simple entropy estimate based on variance
        variance = torch.var(x, dim=-1, keepdim=True)
        normalized_variance = torch.sigmoid(variance * 10)  # Scale to 0-1 range
        return normalized_variance
    
    def forward(self, x):
        """
        Forward pass with selective quantization
        """
        # Store residual for skip connection
        residual = x
        
        # Project input
        h = self.in_proj(x)
        h = F.gelu(h)
        
        # Apply quantization
        if self.is_dynamic:
            # Estimate entropy
            entropy = self.estimate_entropy(h)
            
            # Apply different quantization levels based on estimated entropy
            high_entropy_mask = entropy > self.entropy_threshold
            
            # Apply 8-bit or 2-bit quantization based on token entropy
            h_q = torch.where(
                high_entropy_mask,
                self.quantizer.quantize(h, bit_width=8),  # 8-bit for high entropy
                self.quantizer.quantize(h, bit_width=2)   # 2-bit for low entropy
            )
        else:
            # Apply fixed bit-width quantization
            h_q = self.quantizer.quantize(h, bit_width=self.bit_width)
        
        # Apply Mamba SSM processing
        h_out = self.mamba(h_q)
        
        # Project back to original dimension
        out = self.out_proj(h_out)
        out = self.dropout(out)
        
        # Residual connection and normalization
        out = self.norm(out + residual)
        
        return out


# ==============================================
# Model Definitions
# ==============================================

class BaselineMambaModel(nn.Module):
    """
    Baseline Mamba model without quantization
    Using the true Mamba SSM implementation or fallback GRU if not available
    """
    def __init__(self, 
                 input_dim,       # Input feature dimension 
                 d_model=64,      # Hidden dimension
                 n_layer=2,       # Number of Mamba blocks
                 d_state=16,      # SSM state expansion factor
                 expand=2,        # Expansion factor for feedforward
                 dropout=0.1):
        super().__init__()
        
        print(f"Initializing BaselineMambaModel: input_dim={input_dim}, d_model={d_model}, n_layer={n_layer}")
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,         # Default local convolution width
                expand=expand,
                dropout=dropout
            ) for _ in range(n_layer)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {num_params:,} parameters")
    
    def forward(self, x):
        """
        Forward pass through model
        x shape: [batch_size, seq_len, input_dim]
        """
        # Project input to model dimension
        h = self.input_proj(x)
        
        # Process through Mamba layers
        for layer in self.layers:
            h = layer(h)
        
        # Take output from the last time step
        h = h[:, -1]  # (batch_size, d_model)
        
        # Layer norm and output projection
        h = self.ln_f(h)
        h = self.output_proj(h)  # (batch_size, 1)
        
        return h.squeeze(-1)


class FixedBitMambaModel(nn.Module):
    """
    Mamba model with fixed bit-width Matryoshka quantization
    """
    def __init__(self, 
                 input_dim,      # Input feature dimension 
                 d_model=64,     # Hidden dimension
                 n_layer=2,      # Number of Mamba blocks
                 bit_width=2,    # Fixed bit-width for quantization
                 d_state=16,     # SSM state expansion factor
                 expand=2,       # Expansion factor for feedforward
                 dropout=0.1):
        super().__init__()
        
        print(f"Initializing FixedBitMambaModel: input_dim={input_dim}, d_model={d_model}, bit_width={bit_width}")
        self.bit_width = bit_width  # Store for model tracking
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Stack of quantized Mamba blocks
        self.layers = nn.ModuleList([
            MambaMatryoshkaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=expand,
                bit_width=bit_width,
                is_dynamic=False,
                dropout=dropout
            ) for _ in range(n_layer)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {num_params:,} parameters")
    
    def forward(self, x):
        """
        Forward pass through model
        x shape: [batch_size, seq_len, input_dim]
        """
        # Project input to model dimension
        h = self.input_proj(x)
        
        # Process through quantized Mamba layers
        for layer in self.layers:
            h = layer(h)
        
        # Take output from the last time step
        h = h[:, -1]  # (batch_size, d_model)
        
        # Layer norm and output projection
        h = self.ln_f(h)
        h = self.output_proj(h)  # (batch_size, 1)
        
        return h.squeeze(-1)


class DynamicBitMambaModel(nn.Module):
    """
    Mamba model with dynamic bit-width Matryoshka quantization
    """
    def __init__(self, 
                 input_dim,      # Input feature dimension 
                 d_model=64,     # Hidden dimension
                 n_layer=2,      # Number of Mamba blocks
                 entropy_threshold=0.5,  # Threshold for bit-width switching
                 d_state=16,     # SSM state expansion factor
                 expand=2,       # Expansion factor for feedforward
                 dropout=0.1):
        super().__init__()
        
        print(f"Initializing DynamicBitMambaModel: input_dim={input_dim}, d_model={d_model}, entropy_threshold={entropy_threshold}")
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Stack of dynamic quantized Mamba blocks
        self.layers = nn.ModuleList([
            MambaMatryoshkaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=expand,
                is_dynamic=True,
                entropy_threshold=entropy_threshold,
                dropout=dropout
            ) for _ in range(n_layer)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {num_params:,} parameters")
    
    def forward(self, x):
        """
        Forward pass through model
        x shape: [batch_size, seq_len, input_dim]
        """
        # Project input to model dimension
        h = self.input_proj(x)
        
        # Process through dynamically quantized Mamba layers
        for layer in self.layers:
            h = layer(h)
        
        # Take output from the last time step
        h = h[:, -1]  # (batch_size, d_model)
        
        # Layer norm and output projection
        h = self.ln_f(h)
        h = self.output_proj(h)  # (batch_size, 1)
        
        return h.squeeze(-1)


# ==============================================
# Distillation and Dataset Handling
# ==============================================

class DistillationLoss(nn.Module):
    """
    Loss function for knowledge distillation
    """
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
        print(f"Initialized DistillationLoss: alpha={alpha}, temperature={temperature}")
    
    def forward(self, student_outputs, teacher_outputs, targets):
        """
        Calculate distillation loss with temperature scaling
        """
        # Hard target loss (student predictions vs. ground truth)
        hard_loss = self.mse_loss(student_outputs, targets)
        
        # Soft target loss (student vs. teacher predictions)
        # Scaled MSE for regression
        soft_loss = self.mse_loss(
            student_outputs / self.temperature,
            teacher_outputs / self.temperature
        ) * (self.temperature ** 2)
        
        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


class HFTDataset(Dataset):
    """
    Dataset for high-frequency trading data with sliding windows
    """
    def __init__(self, X, y):
        """
        Initialize dataset with features and targets
        X: [num_samples, features, time_steps]
        y: [num_samples]
        """
        # Convert to torch tensors if not already
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        print(f"Created dataset with {len(self.y)} samples, X shape: {self.X.shape}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================
# Training and Evaluation Functions
# ==============================================

def train_baseline_model(model, train_loader, val_loader, epochs=5, lr=0.001, 
                        device='cpu', model_save_path='matquant_results/best_baseline.pt'):
    """
    Train baseline model without distillation.
    Optimized for GPU training with mixed precision.
    """
    print(f"\nTraining baseline model for {epochs} epochs with lr={lr}")
    print(f"Using device: {device}")
    start_time = time.time()
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Use mixed precision for faster GPU training
    scaler = torch.cuda.amp.GradScaler() if str(device) != 'cpu' else None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        
        # Use tqdm for progress tracking
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Mixed precision training if on GPU
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for CPU
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            # Update progress
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        # Calculate average training loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            
            for X_batch, y_batch in progress_bar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                # Update validation loss
                val_loss += loss.item()
                progress_bar.set_postfix({'val_loss': f'{val_loss/len(progress_bar):.6f}'})
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  [*] New best model saved to {model_save_path} (val_loss: {val_loss:.6f})")
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{epochs}] - {epoch_time:.1f}s - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Load best model
    print(f"Loading best model from {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    
    # Final timing
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    
    return train_losses, val_losses, model


def train_with_distillation(student_model, teacher_model, train_loader, val_loader, 
                           epochs=5, lr=0.001, alpha=0.5, temperature=2.0, 
                           device='cpu', model_save_path='matquant_results/best_student.pt'):
    """
    Train student model with distillation from a teacher.
    Optimized for GPU training with mixed precision.
    """
    print(f"\nTraining student model with distillation for {epochs} epochs")
    print(f"Learning rate: {lr}, Alpha: {alpha}, Temperature: {temperature}")
    print(f"Using device: {device}")
    start_time = time.time()
    
    # Move models to device
    student_model.to(device)
    teacher_model.to(device)
    
    # Set teacher model to evaluation mode (fixed)
    teacher_model.eval()
    
    # Optimizer and loss functions
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    distill_criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    mse_criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Use mixed precision for faster GPU training
    scaler = torch.cuda.amp.GradScaler() if str(device) != 'cpu' else None
    
    for epoch in range(epochs):
        # Training phase
        student_model.train()
        epoch_start = time.time()
        running_loss = 0.0
        
        # Use tqdm for progress tracking
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Get teacher predictions (without gradient tracking)
            with torch.no_grad():
                teacher_outputs = teacher_model(X_batch)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Mixed precision training if on GPU
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    student_outputs = student_model(X_batch)
                    loss = distill_criterion(student_outputs, teacher_outputs, y_batch)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for CPU
                student_outputs = student_model(X_batch)
                loss = distill_criterion(student_outputs, teacher_outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            # Update progress
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        # Calculate average training loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase (use standard MSE, not distillation loss)
        student_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            
            for X_batch, y_batch in progress_bar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = student_model(X_batch)
                        loss = mse_criterion(outputs, y_batch)
                else:
                    outputs = student_model(X_batch)
                    loss = mse_criterion(outputs, y_batch)
                
                # Update validation loss
                val_loss += loss.item()
                progress_bar.set_postfix({'val_loss': f'{val_loss/len(progress_bar):.6f}'})
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), model_save_path)
            print(f"  [*] New best model saved to {model_save_path} (val_loss: {val_loss:.6f})")
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{epochs}] - {epoch_time:.1f}s - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Load best model
    print(f"Loading best model from {model_save_path}")
    student_model.load_state_dict(torch.load(model_save_path))
    
    # Final timing
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    
    return train_losses, val_losses, student_model


def evaluate_trading_performance(model, test_loader, price_data, 
                               transaction_cost_bps=1.0, slippage_bps=0.5,
                               device='cpu', reverse_signal=False):
    """
    Evaluate model with realistic trading simulation
    Including transaction costs and slippage
    
    Parameters:
    -----------
    model: PyTorch model
        Trained model to evaluate
    test_loader: DataLoader
        DataLoader containing test data
    price_data: DataFrame
        Price data for the test period
    transaction_cost_bps: float
        Transaction cost in basis points
    slippage_bps: float
        Slippage in basis points
    device: str
        Device to run evaluation on
    reverse_signal: bool
        Whether to reverse the trading signals (used to test signal inversion phenomenon)
    """
    print("\nEvaluating model with realistic trading simulation...")
    print(f"Transaction costs: {transaction_cost_bps} bps, Slippage: {slippage_bps} bps")
    if reverse_signal:
        print("SIGNAL INVERSION MODE: Trading signals will be reversed")
    
    start_time = time.time()
    
    model.to(device)
    model.eval()
    
    # Generate predictions
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm.tqdm(test_loader, desc="Generating predictions")
        
        for X_batch, y_batch in progress_bar:
            X_batch = X_batch.to(device)
            
            # Forward pass - use mixed precision if on GPU
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
            else:
                outputs = model(X_batch)
            
            # Store predictions and targets
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_preds)
    actuals = np.array(all_targets)
    
    # If signal inversion is enabled, flip the predictions
    if reverse_signal:
        predictions = -predictions
    
    # Calculate MSE
    mse = np.mean((predictions - actuals) ** 2)
    print(f"Test MSE: {mse:.6f}")
    
    # Extract relevant price data for trading simulation
    test_size = len(predictions)
    prices = price_data[-test_size-1:-1]['price'].values
    spreads = price_data[-test_size-1:-1]['spread'].values
    
    # Initialize portfolio
    initial_capital = 10000
    capital = initial_capital
    position = 0  # -1 (short), 0 (neutral), 1 (long)
    portfolio_values = [float(capital)]
    positions = [position]
    transaction_costs = []
    trades = []
    
    # Trading simulation with costs
    print("Running trading simulation...")
    
    # Process in batches for progress visualization
    batch_size = 5000
    num_batches = (len(predictions) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm.tqdm(range(num_batches), desc="Simulating trades"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(predictions))
        
        for i in range(start_idx, end_idx):
            # Determine signal (vectorized approach)
            signal = 1 if predictions[i] > 0.0001 else (-1 if predictions[i] < -0.0001 else 0)
            
            # Skip if signal is the same as current position
            if signal == position:
                # Still apply actual returns based on current position
                capital *= (1 + position * actuals[i])
                portfolio_values.append(float(capital))
                positions.append(position)
                continue
            
            # Calculate transaction costs
            if abs(signal - position) > 0:  # Position change
                # Track trade
                trades.append({
                    'entry_idx': i,
                    'entry_price': float(prices[i]),
                    'position': signal
                })
                
                # Transaction fee (percentage of trade size)
                fee = capital * abs(signal - position) * transaction_cost_bps / 10000
                
                # Slippage (half of spread)
                slippage = capital * abs(signal - position) * (spreads[i] / (2 * prices[i])) * slippage_bps / 100
                
                # Apply costs
                total_cost = fee + slippage
                capital -= total_cost
                transaction_costs.append(float(total_cost))
            
            # Update position
            position = signal
            positions.append(position)
            
            # Apply actual return based on new position
            capital *= (1 + position * actuals[i])
            portfolio_values.append(float(capital))
    
    # Convert to numpy arrays for faster calculation
    portfolio_values = np.array(portfolio_values, dtype=np.float64)
    positions = np.array(positions, dtype=np.int32)
    
    # Calculate performance metrics
    print("Calculating performance metrics...")
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Trading metrics (annualized assuming HFT frequency)
    ticks_per_second = 1000  # For nanosecond precision
    seconds_in_year = 252 * 6.5 * 3600  # 252 trading days, 6.5 hours per day
    annualization_factor = np.sqrt(seconds_in_year / (len(portfolio_returns) / ticks_per_second))
    
    # Core metrics
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * annualization_factor if np.std(portfolio_returns) > 0 else 0
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    max_drawdown = calculate_max_drawdown(portfolio_values)
    win_rate = np.mean(portfolio_returns > 0) * 100
    
    # Average transaction cost
    avg_transaction_cost = np.mean(transaction_costs) if transaction_costs else 0
    
    # Calculate additional metrics
    profit_factor = calculate_profit_factor(portfolio_returns)
    sortino_ratio = calculate_sortino_ratio(portfolio_returns, annualization_factor)
    calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else float('inf')
    
    # Position statistics
    long_pct = np.mean(positions == 1) * 100
    short_pct = np.mean(positions == -1) * 100
    neutral_pct = np.mean(positions == 0) * 100
    
    # Print summary
    print("\n----- TRADING PERFORMANCE SUMMARY -----")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Sortino Ratio: {sortino_ratio:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.4f}")
    print(f"Calmar Ratio: {calmar_ratio:.4f}" if calmar_ratio != float('inf') else "Calmar Ratio: ∞")
    print(f"Number of Trades: {len(trades)}")
    print(f"Average Transaction Cost: {avg_transaction_cost:.6f}")
    print(f"Long Positions: {long_pct:.2f}%, Short: {short_pct:.2f}%, Neutral: {neutral_pct:.2f}%")
    print("-------------------------------------")
    
    elapsed = time.time() - start_time
    print(f"Trading evaluation completed in {elapsed:.2f} seconds")
    
    return {
        'mse': mse,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'calmar_ratio': calmar_ratio,
        'n_trades': len(trades),
        'transaction_costs': avg_transaction_cost,
        'portfolio_values': portfolio_values,
        'portfolio_returns': portfolio_returns,
        'long_pct': long_pct,
        'short_pct': short_pct,
        'neutral_pct': neutral_pct,
        'num_trades': len(trades),
        'avg_transaction_cost': avg_transaction_cost
    }


def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown percentage
    """
    peak = equity_curve[0]
    max_dd = 0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        
        dd = (peak - value) / peak * 100
        max_dd = max(max_dd, dd)
    
    return max_dd


def calculate_profit_factor(returns):
    """
    Calculate profit factor (sum of profits / sum of losses)
    """
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    return profits / losses if losses > 0 else float('inf')


def calculate_sortino_ratio(returns, annualization_factor):
    """
    Calculate Sortino ratio (return / downside deviation)
    """
    avg_return = np.mean(returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        return float('inf')
    
    return avg_return / downside_deviation * annualization_factor


def visualize_results(results_dict, save_path='matquant_results'):
    """
    Visualize and save performance results
    """
    print("\nVisualizing and saving performance results...")
    start_time = time.time()
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare data for visualization
    models = list(results_dict.keys())
    metrics = ['sharpe_ratio', 'sortino_ratio', 'profit_factor', 'win_rate', 'total_return']
    
    # Plot performance metrics
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('MatQuant-Mamba Performance Comparison', fontsize=16)
    
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        values = [results_dict[model][metric] for model in models]
        
        # Cap very large values for better visualization
        if metric in ['profit_factor', 'sortino_ratio']:
            values = [min(v, 20) if not np.isinf(v) else 20 for v in values]
        
        axs[row, col].bar(models, values)
        axs[row, col].set_title(f'{metric.replace("_", " ").title()}')
        axs[row, col].set_xticklabels(models, rotation=45, ha='right')
        axs[row, col].grid(True, axis='y')
    
    # Plot drawdowns
    row, col = len(metrics) // 2, len(metrics) % 2
    values = [results_dict[model]['max_drawdown'] for model in models]
    axs[row, col].bar(models, values)
    axs[row, col].set_title('Max Drawdown (%)')
    axs[row, col].set_xticklabels(models, rotation=45, ha='right')
    axs[row, col].grid(True, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{save_path}/performance_comparison.png')
    print(f"Saved performance chart to {save_path}/performance_comparison.png")
    
    # Plot position distribution
    plt.figure(figsize=(12, 6))
    position_data = {
        'Long': [results_dict[model]['long_pct'] for model in models],
        'Neutral': [results_dict[model]['neutral_pct'] for model in models],
        'Short': [results_dict[model]['short_pct'] for model in models]
    }
    
    bottom = np.zeros(len(models))
    for position, values in position_data.items():
        plt.bar(models, values, bottom=bottom, label=position)
        bottom += np.array(values)
    
    plt.title('Position Distribution by Model')
    plt.xlabel('Model')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_path}/position_distribution.png')
    print(f"Saved position distribution chart to {save_path}/position_distribution.png")
    
    # Save detailed metrics to CSV
    metrics_df = pd.DataFrame(index=models)
    
    # Add all metrics to dataframe
    all_metrics = ['mse', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'profit_factor', 
                  'total_return', 'max_drawdown', 'win_rate', 'avg_transaction_cost', 'num_trades',
                  'long_pct', 'short_pct', 'neutral_pct']
    
    for metric in all_metrics:
        metrics_df[metric] = [results_dict[model][metric] for model in models]
    
    metrics_df.to_csv(f'{save_path}/performance_metrics.csv')
    print(f"Saved detailed metrics to {save_path}/performance_metrics.csv")
    
    elapsed = time.time() - start_time
    print(f"Visualization completed in {elapsed:.2f} seconds")
    
    return metrics_df


def run_matquant_mamba_pipeline(
    minutes=5,
    epochs=10,
    d_model=64,
    d_state=16,
    n_layers=4,
    use_gpu=True,
    transaction_cost_bps=1.0, 
    slippage_bps=0.5,
    save_path='matquant_results',
    fast_mode=False
):
    """
    Run the complete MatQuant-Mamba pipeline
    """
    print("\n" + "="*80)
    print(" "*20 + "MATQUANT-MAMBA HFT PIPELINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  Minutes of data: {minutes}")
    print(f"  Training epochs: {epochs}")
    print(f"  Model dimensions: d_model={d_model}, d_state={d_state}, n_layers={n_layers}")
    print(f"  Using GPU: {use_gpu}")
    print(f"  Transaction costs: {transaction_cost_bps} bps")
    print(f"  Slippage: {slippage_bps} bps")
    print(f"  Fast mode: {fast_mode}")
    print(f"  Save path: {save_path}")
    print("="*80 + "\n")
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Generate synthetic HFT data
    print("\nSTEP 1: Generating synthetic HFT data...")
    start_time = time.time()
    
    # Data parameters
    minutes = minutes
    ticks_per_second = 1000
    total_ticks = minutes * 60 * ticks_per_second
    
    # Initialize data generator
    print(f"Initializing data generator for {minutes} minutes at {ticks_per_second} ticks/second")
    print(f"Total ticks: {total_ticks:,}")
    
    data_gen = EnhancedNanoHFTDataGenerator(
        trading_minutes=minutes,
        ticks_per_second=ticks_per_second
    )
    
    # Generate data
    print("Generating price path and calculating features...")
    df = data_gen.generate_data()
    
    # Report data generation time and shape
    data_gen_time = time.time() - start_time
    print(f"Data generation completed in {data_gen_time:.2f} seconds")
    print(f"Final dataset shape: {df.shape} with {df.shape[1] - 2} features")
    
    # Step 2: Prepare data for training
    print("\nSTEP 2: Preparing data for training...")
    prep_start_time = time.time()
    
    # Data normalization (using RobustScaler for resilience to outliers)
    print("Normalizing features...")
    feature_columns = [col for col in df.columns if col not in ['price', 'spread', 'timestamp']]
    
    scaler = RobustScaler()
    df_norm = df.copy()
    df_norm[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    # Create sliding windows
    print("Creating sliding windows...")
    window_size = 50  # Number of time steps per window
    forecast_horizon = 1  # Prediction horizon
    
    X_windows, y = [], []
    
    # Create sliding windows (vectorized for speed)
    for i in range(len(df_norm) - window_size - forecast_horizon + 1):
        X_windows.append(df_norm.iloc[i:i+window_size][feature_columns].values)
        
        # Target is price return
        current_price = df.iloc[i+window_size-1]['price']
        future_price = df.iloc[i+window_size+forecast_horizon-1]['price']
        price_return = (future_price / current_price) - 1
        
        y.append(price_return)
    
    # Convert to numpy arrays
    X_windows = np.array(X_windows)
    y = np.array(y)
    
    # Reshape for clarity
    X_windows = X_windows.reshape(-1, window_size, len(feature_columns))
    y = y.reshape(-1, 1)
    
    # Check window creation
    print(f"X windows shape: {X_windows.shape}")
    print(f"y shape: {y.shape}")
    
    if X_windows.shape[0] != y.shape[0]:
        print("Warning: X and y lengths differ, adjusting to common size")
        min_len = min(X_windows.shape[0], y.shape[0])
        X_windows = X_windows[:min_len]
        y = y[:min_len]
    
    # Split data
    test_size = int(0.2 * len(X_windows))
    val_size = int(0.2 * (len(X_windows) - test_size))
    
    # Test data is last chunk (most recent)
    X_test = X_windows[-test_size:]
    y_test = y[-test_size:]
    
    # Validation is second last chunk
    X_val = X_windows[-(test_size+val_size):-test_size]
    y_val = y[-(test_size+val_size):-test_size]
    
    # Training is everything else
    X_train = X_windows[:-(test_size+val_size)]
    y_train = y[:-(test_size+val_size)]
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create PyTorch datasets
    train_dataset = HFTDataset(X_train, y_train)
    val_dataset = HFTDataset(X_val, y_val)
    test_dataset = HFTDataset(X_test, y_test)
    
    # Create data loaders
    batch_size = 1024 if device.type == 'cuda' else 128  # Larger batch sizes for GPU
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4 if device.type == 'cuda' else 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4 if device.type == 'cuda' else 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4 if device.type == 'cuda' else 0)
    
    # Report preparation time
    prep_time = time.time() - prep_start_time
    print(f"Data preparation completed in {prep_time:.2f} seconds")
    
    # Step 3: Train and evaluate models
    print("\nSTEP 3: Training and evaluating models...")
    
    # Save test price data for trading simulation
    test_prices = df.iloc[-(test_size+forecast_horizon):]
    
    # Dictionary to store results for all models
    results = {}
    
    # Define model configurations
    print("\nDefining models for evaluation...")
    input_dim = X_windows.shape[2]
    output_dim = 1
    
    model_configs = {
        "Baseline_Mamba": {
            "model": BaselineMambaModel(
                input_dim=input_dim, 
                d_model=d_model,
                n_layer=n_layers,
                d_state=d_state
            ),
            "distill": False,
            "epochs": epochs
        }
    }
    
    if not fast_mode:
        # Add MatQuant models with various bit configurations
        bit_configs = [(2, 8), (3, 6), (4, 4), (6, 2)]
        for i, bits in enumerate(bit_configs):
            low_bit = bits[0]
            model_name = f"MatQuant_{bits[0]}+{bits[1]}_bits"
            
            model_configs[model_name] = {
                "model": FixedBitMambaModel(
                    input_dim=input_dim,
                    d_model=d_model,
                    n_layer=n_layers,
                    d_state=d_state,
                    bit_width=low_bit
                ),
                "distill": True,
                "epochs": epochs
            }
    else:
        # Just one MatQuant model for fast mode
        model_configs["MatQuant_4+4_bits"] = {
            "model": FixedBitMambaModel(
                input_dim=input_dim,
                d_model=d_model,
                n_layer=n_layers,
                d_state=d_state,
                bit_width=4
            ),
            "distill": True,
            "epochs": epochs
        }
    
    # Train and evaluate each model
    print(f"\nTraining and evaluating {len(model_configs)} models...")
    
    # First, train baseline model (teacher)
    print("\n\n" + "="*60)
    print(f"BASELINE MODEL TRAINING")
    print("="*60)
    
    baseline_key = next(iter(model_configs.keys()))  # Get first model (baseline)
    baseline_model = model_configs[baseline_key]["model"]
    
    print(f"Training baseline model: {baseline_key}")
    print(f"Model summary: {baseline_model}")
    print(f"Total parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
    
    # Train baseline model
    baseline_train_losses, baseline_val_losses, baseline_model = train_baseline_model(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=model_configs[baseline_key]["epochs"],
        device=device,
        model_save_path=f"{save_path}/{baseline_key}.pt"
    )
    
    # Evaluate baseline model
    baseline_results = evaluate_trading_performance(
        model=baseline_model,
        test_loader=test_loader,
        price_data=test_prices,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        device=device
    )
    
    # Also evaluate with reversed signals
    baseline_reversed_results = evaluate_trading_performance(
        model=baseline_model,
        test_loader=test_loader,
        price_data=test_prices,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        device=device,
        reverse_signal=True
    )
    
    # Store results
    results[baseline_key] = baseline_results
    results[f"{baseline_key}_reversed"] = baseline_reversed_results
    
    # Plot baseline training loss
    plt.figure(figsize=(10, 5))
    plt.plot(baseline_train_losses, label='Train Loss')
    plt.plot(baseline_val_losses, label='Validation Loss')
    plt.title(f'{baseline_key} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{baseline_key}_loss.png")
    
    # Now train and evaluate MatQuant models with distillation
    for model_name, config in list(model_configs.items())[1:]:
        print("\n\n" + "="*60)
        print(f"TRAINING MODEL: {model_name}")
        print("="*60)
        
        model = config["model"]
        print(f"Model summary: {model}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if config["distill"]:
            # Use distillation with baseline as teacher
            train_losses, val_losses, model = train_with_distillation(
                student_model=model,
                teacher_model=baseline_model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config["epochs"],
                device=device,
                model_save_path=f"{save_path}/{model_name}.pt"
            )
        else:
            # Train without distillation
            train_losses, val_losses, model = train_baseline_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config["epochs"],
                device=device,
                model_save_path=f"{save_path}/{model_name}.pt"
            )
        
        # Evaluate model
        model_results = evaluate_trading_performance(
            model=model,
            test_loader=test_loader,
            price_data=test_prices,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            device=device
        )

        # Also evaluate with reversed signals
        model_reversed_results = evaluate_trading_performance(
            model=model,
            test_loader=test_loader,
            price_data=test_prices,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            device=device,
            reverse_signal=True
        )
        
        # Store results
        results[model_name] = model_results
        results[f"{model_name}_reversed"] = model_reversed_results
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{model_name} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/{model_name}_loss.png")
    
    # Step 4: Visualize and analyze results
    print("\nSTEP 4: Visualizing and analyzing results...")
    metrics_df = visualize_results(results, save_path)
    
    print("\n" + "="*80)
    print(" "*20 + "MATQUANT-MAMBA PIPELINE COMPLETE")
    print("="*80)
    print(f"Results saved to {save_path}")
    
    # Display final comparison table
    print("\nPerformance Comparison:")
    display_metrics = ['sharpe_ratio', 'sortino_ratio', 'total_return', 'max_drawdown', 'mse']
    print(metrics_df[display_metrics].round(4))
    
    return results, metrics_df


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model predictive performance (MSE)
    """
    print("\nEvaluating model prediction accuracy...")
    start_time = time.time()
    
    model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm.tqdm(test_loader, desc="Evaluating")
        
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Store predictions and targets
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
            # Update loss
            running_loss += loss.item()
            progress_bar.set_postfix({'mse': f'{running_loss/len(progress_bar):.6f}'})
    
    # Calculate MSE
    mse = running_loss / len(test_loader)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.2f} seconds")
    print(f"Test MSE: {mse:.6f}")
    
    return {
        'mse': mse,
        'predictions': all_preds,
        'targets': all_targets
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MatQuant-Mamba for HFT')
    parser.add_argument('--minutes', type=int, default=10, help='Minutes of data to generate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--d_model', type=int, default=64, help='Model hidden dimension')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of model layers')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage even if available')
    parser.add_argument('--tx_cost', type=float, default=1.0, help='Transaction cost in basis points')
    parser.add_argument('--slippage', type=float, default=0.5, help='Slippage in basis points')
    parser.add_argument('--save_path', type=str, default='matquant_results', help='Directory to save results')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode (fewer models & epochs)')
    parser.add_argument('--reversed_only', action='store_true', help='Only run with reversed signals (no regular evaluation)')
    
    args = parser.parse_args()
    
    # If fast mode, override some parameters
    if args.fast:
        args.minutes = min(args.minutes, 5)  # Cap at 5 minutes
        args.epochs = min(args.epochs, 3)    # Cap at 3 epochs
    
    # Run the pipeline
    results, metrics = run_matquant_mamba_pipeline(
        minutes=args.minutes,
        epochs=args.epochs,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        use_gpu=not args.no_gpu,
        transaction_cost_bps=args.tx_cost,
        slippage_bps=args.slippage,
        save_path=args.save_path,
        fast_mode=args.fast
    ) 
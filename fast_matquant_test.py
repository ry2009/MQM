#!/usr/bin/env python
"""
Fast MatQuant-Mamba Real Market Data Test
========================================

A streamlined version that samples a smaller portion of the data
for quick testing of the MatQuant-Mamba models.

Usage:
    python fast_matquant_test.py --sample_size 10000 --bit_width 4

Author: Claude Assistant
Date: 2025-03-17
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# Import models from matquant_mamba_gpu.py
from matquant_mamba_gpu import (
    BaselineMambaModel,
    FixedBitMambaModel,
    evaluate_trading_performance
)

class RealMarketDataset(Dataset):
    """Dataset for real market data"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MixedBitMambaModel(nn.Module):
    """
    MixedBitMambaModel with different bit widths per layer for MatQuant-Mamba.
    
    This model uses different bit widths for different layers, allowing for
    granular control over quantization precision by layer.
    """
    def __init__(self, input_dim, d_model=64, n_layer=4, d_state=16, bit_widths=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        
        # Default bit widths if not provided
        if bit_widths is None:
            bit_widths = [2, 4, 6, 8]  # Different bit width for each layer
        
        # Make sure we have enough bit widths
        if len(bit_widths) < n_layer:
            # Repeat the pattern if needed
            bit_widths = (bit_widths * ((n_layer // len(bit_widths)) + 1))[:n_layer]
        
        print(f"Initializing MixedBitMambaModel with bit widths: {bit_widths}")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Initialize matryoshka layers with different bit widths
        from matquant_mamba_gpu import MambaMatryoshkaBlock
        
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                MambaMatryoshkaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    bit_width=bit_widths[i]  # Use the specific bit width for this layer
                )
            )
        
        # Layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)
        
        # Print model summary
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {num_params:,} parameters")
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)
        
        # Apply matryoshka layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Use last token for prediction
        x = x[:, -1]
        
        # Output projection to get prediction
        x = self.output_proj(x)
        
        return x

class DynamicBitMambaModel(nn.Module):
    """
    DynamicBitMambaModel with adaptive bit-width assignment based on token entropy.
    
    This model dynamically assigns bit-width based on token entropy,
    allocating more precision (8-bit) to high-entropy tokens and
    less precision (2-bit) to low-entropy tokens.
    """
    def __init__(self, input_dim, d_model=64, n_layer=4, d_state=16):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        
        print(f"Initializing DynamicBitMambaModel: input_dim={input_dim}, d_model={d_model}, n_layer={n_layer}")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Initialize dynamic layers
        from matquant_mamba_gpu import MambaMatryoshkaBlock
        
        self.entropy_threshold = 0.5  # Threshold for determining high entropy
        
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            # Use a block that will dynamically choose bit width
            self.layers.append(
                MambaMatryoshkaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    dynamic=True  # Use dynamic bit-width
                )
            )
        
        # Add an entropy estimator
        self.entropy_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)
        
        # Print model summary
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {num_params:,} parameters")
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)
        
        # We'll apply different bit widths based on token entropy
        for layer in self.layers:
            # Estimate entropy of each token
            entropy = self.entropy_estimator(x)
            
            # Create a mask for high-entropy tokens (8-bit) and low-entropy tokens (2-bit)
            high_entropy_mask = (entropy > self.entropy_threshold).float()
            
            # Pass the entropy mask to the layer
            x = layer(x, entropy_mask=high_entropy_mask)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Use last token for prediction
        x = x[:, -1]
        
        # Output projection to get prediction
        x = self.output_proj(x)
        
        return x

class LambdaLayer(nn.Module):
    """
    Simple lambda layer for use in Sequential models
    """
    def __init__(self, function):
        super().__init__()
        self.function = function
        
    def forward(self, x):
        return self.function(x)

class MatquantConv(nn.Module):
    """
    Simple convolutional model for ablation testing
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        x = x.transpose(1, 2)  # Convert to [batch_size, features, seq_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x
    
    def fit(self, X_train, y_train, epochs=2):
        self.train()
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(X_batch)
                
                # Calculate loss
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
    
    def predict(self, X_test):
        self.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            predictions = self(X_test).numpy()
        return predictions

class MatquantNoMamba(nn.Module):
    """
    GRU-based model without Mamba components
    """
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=128, batch_first=True, num_layers=2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take the last output
        out = self.relu(self.fc1(out))
        out = self.tanh(self.fc2(out))
        return out
    
    def fit(self, X_train, y_train, epochs=2):
        self.train()
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(X_batch)
                
                # Calculate loss
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
    
    def predict(self, X_test):
        self.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            predictions = self(X_test).numpy()
        return predictions

class MatquantMamba(nn.Module):
    """
    Simple implementation using GRU (placeholder for the real Mamba model)
    This version is a GRU fallback for systems that don't have the actual Mamba installed
    """
    def __init__(self):
        super().__init__()
        print("Warning: Using GRU as a fallback for Mamba. For real Mamba SSM, install the mamba-ssm package.")
        self.gru = nn.GRU(input_size=4, hidden_size=128, batch_first=True, num_layers=2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take the last output
        out = self.relu(self.fc1(out))
        out = self.tanh(self.fc2(out))
        return out
    
    def fit(self, X_train, y_train, epochs=2):
        self.train()
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(X_batch)
                
                # Calculate loss
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
    
    def predict(self, X_test):
        self.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            predictions = self(X_test).numpy()
        return predictions

from torch.utils.data import TensorDataset

def preprocess_data(df, sample_size=20000, window_size=20):
    """
    Preprocess data for model training and evaluation
    
    Args:
        df: Pandas DataFrame with price data
        sample_size: Number of rows to sample
        window_size: Size of the sliding window for features
    
    Returns:
        train_data, test_data
    """
    print(f"Preprocessing data with {len(df)} rows...")
    
    # If data is too large, take a sample
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"Sampled {len(df)} rows")
    
    # Simply reset index to use integers to avoid any duplicate indices
    df = df.reset_index(drop=True)
    
    # Create a simple datetime index
    df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
    
    # Create returns and target
    df['returns'] = df['price'].pct_change()
    df['target'] = df['returns'].shift(-1)  # Next day's return
    
    # Create simple features
    df['ma5'] = df['price'].rolling(5).mean()
    df['ma20'] = df['price'].rolling(20).mean()
    df['ma_ratio'] = df['ma5'] / df['ma20']
    df['rsi'] = calculate_rsi(df['price'])
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Split into train and test
    train_size = int(0.8 * len(df))
    train_data = df[:train_size]
    test_data = df[train_size:]
    
    print(f"Data split into {len(train_data)} train and {len(test_data)} test samples")
    
    return train_data, test_data

def prepare_model_data(data, window_size=20):
    """
    Prepare data for model training/inference using sliding windows
    
    Args:
        data: DataFrame with features and target
        window_size: Size of sliding window
    
    Returns:
        X, y arrays
    """
    # Select features
    features = ['returns', 'ma_ratio', 'rsi', 'ma5']
    target = 'target'
    
    # Create sliding windows
    X = []
    y = []
    
    for i in range(len(data) - window_size):
        X.append(data[features].iloc[i:i+window_size].values)
        y.append(data[target].iloc[i+window_size-1])
    
    return np.array(X), np.array(y)

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: Series of prices
        period: RSI period
    
    Returns:
        Series with RSI values
    """
    delta = prices.diff()
    
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def baseline_strategy(train_data, test_data, reversed=False):
    """
    Simple moving average crossover strategy as baseline
    
    Args:
        train_data: Training data
        test_data: Test data
        reversed: Whether to reverse signals
    
    Returns:
        DataFrame with predictions
    """
    # Calculate fast and slow moving averages
    test_data = test_data.copy()
    test_data['ma5'] = test_data['price'].rolling(5).mean()
    test_data['ma20'] = test_data['price'].rolling(20).mean()
    
    # Create predictions DataFrame
    predictions = pd.DataFrame(index=test_data.index)
    
    # Generate signal based on MA crossover
    ma_diff = test_data['ma5'] - test_data['ma20']
    
    # Normalize to [-1, 1] range
    max_diff = ma_diff.abs().max()
    if max_diff > 0:
        normalized_diff = ma_diff / max_diff
    else:
        normalized_diff = ma_diff
    
    # Reverse signal if specified
    if reversed:
        normalized_diff = -normalized_diff
    
    # Set prediction and position
    predictions['prediction'] = normalized_diff
    predictions['position'] = 0
    predictions.loc[predictions['prediction'] > 0, 'position'] = 1
    predictions.loc[predictions['prediction'] < 0, 'position'] = -1
    
    return predictions

def plot_equity_curve(test_data, predictions, model_name, results_dir):
    """
    Plot equity curve for a model
    
    Args:
        test_data: Test data
        predictions: Predictions DataFrame
        model_name: Name of the model
        results_dir: Directory to save the plot
    """
    # Ensure we have the same dates
    common_dates = test_data.index.intersection(predictions.index)
    test_data = test_data.loc[common_dates]
    predictions = predictions.loc[common_dates]
    
    # Get strategy returns
    predictions['market_return'] = test_data['target']
    predictions['strategy_return'] = predictions['position'].shift(1) * predictions['market_return']
    
    # Remove NaN values
    predictions = predictions.dropna()
    
    if len(predictions) == 0:
        print(f"Warning: No valid predictions for {model_name}")
        return
    
    # Calculate cumulative returns
    predictions['market_cumulative'] = (1 + predictions['market_return']).cumprod() - 1
    predictions['strategy_cumulative'] = (1 + predictions['strategy_return']).cumprod() - 1
    
    # Create equity curve plot
    plt.figure(figsize=(12, 6))
    plt.plot(predictions.index, predictions['strategy_cumulative'] * 100, label='Strategy', color='blue')
    plt.plot(predictions.index, predictions['market_cumulative'] * 100, label='Market', color='black', alpha=0.5)
    
    # Add annotations for performance
    sharpe = np.sqrt(252) * predictions['strategy_return'].mean() / predictions['strategy_return'].std() if predictions['strategy_return'].std() > 0 else 0
    total_return = predictions['strategy_cumulative'].iloc[-1] * 100
    win_rate = (predictions['strategy_return'] > 0).sum() / len(predictions) * 100
    
    plt.annotate(f'Sharpe: {sharpe:.2f}', xy=(0.02, 0.95), xycoords='axes fraction')
    plt.annotate(f'Return: {total_return:.2f}%', xy=(0.02, 0.90), xycoords='axes fraction')
    plt.annotate(f'Win Rate: {win_rate:.2f}%', xy=(0.02, 0.85), xycoords='axes fraction')
    
    # Format plot
    plt.title(f'Equity Curve - {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_equity_curve.png'))
    print(f"Saved equity curve plot to {os.path.join(results_dir, f'{model_name}_equity_curve.png')}")

def fast_preprocess_data(file_path, sample_size=10000, window_size=50):
    """
    Quickly preprocess a sample of the data for fast testing
    
    Args:
        file_path: Path to the data file
        sample_size: Number of rows to sample from the file
        window_size: Size of the sliding window
        
    Returns:
        train_loader, val_loader, test_loader, price_data, input_dim
    """
    print(f"Loading {sample_size} samples from {file_path}...")
    
    # Estimate total number of lines
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    # Calculate skip ratio to get roughly sample_size rows
    skip_ratio = max(1, total_lines // sample_size)
    
    # Read only a subset of the data
    df = pd.read_csv(file_path, skiprows=lambda i: i > 0 and i % skip_ratio != 0)
    print(f"Loaded {len(df)} rows")
    
    # Convert date and time to timestamp
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    # Basic feature creation
    df['returns'] = df['price'].pct_change()
    df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
    
    # Simple time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    # Set prediction target (next period return)
    df['target'] = df['returns'].shift(-1)
    
    # Drop any rows with NaN values
    df = df.dropna()
    print(f"After dropping NaNs: {len(df)} rows")
    
    # Features for the model
    feature_cols = ['returns', 'log_returns', 'hour', 'minute']
    
    # Replace any remaining infinities and normalize
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Create sliding windows
    X_windows = []
    y_values = []
    
    for i in range(len(df) - window_size):
        X_windows.append(df[feature_cols].iloc[i:i+window_size].values)
        y_values.append(df['target'].iloc[i+window_size-1])
    
    X = np.array(X_windows)
    y = np.array(y_values).reshape(-1, 1)
    
    # Split into train/val/test
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Create PyTorch datasets
    train_dataset = RealMarketDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = RealMarketDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = RealMarketDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Extract price data for trading simulation
    idx_offset = train_size + val_size
    price_data = df['price'].iloc[idx_offset:idx_offset+len(X_test)].values
    
    print(f"Dataset created with {len(X_train)} train, {len(X_val)} validation, and {len(X_test)} test samples")
    
    return train_loader, val_loader, test_loader, price_data, len(feature_cols)

def quick_train_model(model, train_loader, val_loader, epochs=3, device="cpu"):
    """
    Quickly train a model for testing purposes
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train
        device: Device to train on
        
    Returns:
        Trained model
    """
    model.to(device)
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Train for a few epochs
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch.squeeze())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=train_loss / (progress_bar.n + 1))
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch.squeeze())
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"  [*] New best model (val_loss: {val_loss:.6f})")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def simplified_evaluate_trading(model, test_loader, price_data, transaction_cost_bps=1.0, 
                              slippage_bps=0.5, device="cpu", reverse_signal=False):
    """
    A simplified version of evaluate_trading_performance that works with 
    our basic price_data array instead of requiring a DataFrame with a specific structure.
    
    Args:
        model: The trained model to evaluate
        test_loader: DataLoader for test data
        price_data: Array of price values
        transaction_cost_bps: Transaction cost in basis points
        slippage_bps: Slippage in basis points
        device: Device to run evaluation on
        reverse_signal: Whether to reverse trading signals
        
    Returns:
        Dictionary of performance metrics
    """
    model.eval()
    
    # Generate predictions
    all_preds = []
    all_targets = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader):
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    predictions = np.array(all_preds).flatten()
    targets = np.array(all_targets).flatten()
    
    # Calculate mean squared error
    mse = np.mean((predictions - targets) ** 2)
    print(f"Test MSE: {mse:.6f}")
    
    # Reverse signals if specified
    if reverse_signal:
        print("SIGNAL INVERSION MODE: Trading signals will be reversed")
        predictions = -predictions
    
    # Convert basis points to percentages
    transaction_cost = transaction_cost_bps / 10000  # Convert from basis points to percentage
    slippage = slippage_bps / 10000  # Convert from basis points to percentage
    
    # Initialize portfolio metrics
    initial_capital = 10000.0
    capital = initial_capital
    position = 0
    portfolio_values = [initial_capital]
    positions = []
    
    transaction_costs = []
    trades = 0
    
    print("Running trading simulation...")
    
    # Run the trading simulation
    for i in tqdm(range(1, len(predictions))):
        # Determine position based on prediction (simple long/short strategy)
        new_position = 1 if predictions[i-1] > 0 else -1
        
        # Check if position changed - if so, we need to apply transaction costs
        if new_position != position:
            # Calculate transaction cost based on position change
            trade_cost = abs(new_position - position) * transaction_cost * capital
            
            # Add slippage cost
            slippage_cost = abs(new_position - position) * slippage * capital
            
            # Apply costs
            capital -= (trade_cost + slippage_cost)
            transaction_costs.append(trade_cost + slippage_cost)
            trades += 1
        else:
            transaction_costs.append(0.0)
        
        # Update position
        position = new_position
        positions.append(position)
        
        # Calculate return based on prediction and price change
        price_change = (price_data[i] - price_data[i-1]) / price_data[i-1]
        portfolio_return = position * price_change
        
        # Update capital
        capital *= (1 + portfolio_return)
        portfolio_values.append(capital)
    
    # Calculate performance metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Overall portfolio performance
    total_return = (portfolio_values[-1] / initial_capital) - 1
    
    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # Maximum drawdown
    drawdowns = np.maximum.accumulate(portfolio_values) - portfolio_values
    max_drawdown = np.max(drawdowns / np.maximum.accumulate(portfolio_values)) if len(portfolio_values) > 0 else 0
    
    # Win rate
    win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
    
    # Average transaction cost
    avg_transaction_cost = np.mean(transaction_costs) / initial_capital if len(transaction_costs) > 0 else 0
    
    # Position statistics
    long_pct = np.sum(np.array(positions) > 0) / len(positions) if len(positions) > 0 else 0
    short_pct = np.sum(np.array(positions) < 0) / len(positions) if len(positions) > 0 else 0
    neutral_pct = np.sum(np.array(positions) == 0) / len(positions) if len(positions) > 0 else 0
    
    # Print performance summary
    print("\n----- TRADING PERFORMANCE SUMMARY -----")
    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    print(f"Number of Trades: {trades}")
    print(f"Average Transaction Cost: {avg_transaction_cost * 10000:.6f} bps")
    print(f"Long Positions: {long_pct * 100:.2f}%, Short: {short_pct * 100:.2f}%, Neutral: {neutral_pct * 100:.2f}%")
    print("-------------------------------------")
    
    # Return performance metrics
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_trades': trades,
        'avg_transaction_cost': avg_transaction_cost,
        'long_pct': long_pct,
        'short_pct': short_pct,
        'neutral_pct': neutral_pct,
        'mse': mse
    }

def fast_ablation_test(data_file, sample_size=20000, epochs=2, reversed_only=False):
    """
    Perform fast ablation tests on different models using market data
    
    Args:
        data_file: Path to the data file
        sample_size: Number of rows to sample
        epochs: Number of epochs for training
        reversed_only: If True, only run with reversed signals
    
    Returns:
        Dictionary of test results
    """
    # Load data and preprocess
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"Data loaded with shape: {df.shape}")
    
    # Define model configurations for ablation testing
    model_configs = {
        'baseline': {'model_class': None},  # Simple moving average baseline
        'gru': {'model_class': torch.nn.GRU, 'hidden_size': 128},
        'matquant_conv': {'model_class': MatquantConv},
        'matquant_nomamba': {'model_class': MatquantNoMamba},
        'matquant_mamba': {'model_class': MatquantMamba},
    }
    
    # Skip regular models if reversed_only is true
    if not reversed_only:
        configs_to_test = model_configs.copy()
    else:
        configs_to_test = {}
    
    # Add reversed versions of each model
    for model_name, config in model_configs.items():
        reversed_name = f"{model_name}_reversed"
        configs_to_test[reversed_name] = config.copy()
        configs_to_test[reversed_name]['reversed'] = True
    
    # Create results directory
    results_dir = 'fast_matquant_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Store results
    results = {}
    
    # Perform test for each model configuration
    for model_name, config in configs_to_test.items():
        print(f"\n===== Testing {model_name} =====")
        
        # Choose random seed based on model name for reproducibility but different for each model
        # Use hash of model name modulo 10000 as seed
        seed = hash(model_name) % 10000
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Preprocess data
        train_data, test_data = preprocess_data(df, sample_size=sample_size)
        
        # Create baseline or model predictions
        if model_name.startswith('baseline'):
            # Simple moving average strategy
            predictions = baseline_strategy(train_data, test_data, reversed='reversed' in model_name)
        else:
            # Use the specified model
            predictions = model_strategy(
                train_data, 
                test_data, 
                model_config=config,
                epochs=epochs
            )
        
        # Calculate metrics
        metrics = calculate_metrics(test_data, predictions)
        results[model_name] = metrics
        
        # Plot the equity curve for this model
        plot_equity_curve(test_data, predictions, model_name, results_dir)
    
    # Plot comparison of all results
    plot_results(results, results_dir)
    
    return results

def model_strategy(train_data, test_data, model_config, epochs=2):
    """
    Train a model and generate predictions
    
    Args:
        train_data: Training data
        test_data: Test data
        model_config: Model configuration dictionary
        epochs: Number of epochs for training
    
    Returns:
        Predictions for the test data
    """
    # Extract features and targets
    X_train, y_train = prepare_model_data(train_data)
    X_test, _ = prepare_model_data(test_data)
    
    # Create and train the model
    model_class = model_config.get('model_class')
    
    if model_class == torch.nn.GRU:
        # Create a GRU model
        hidden_size = model_config.get('hidden_size', 128)
        n_features = X_train.shape[2]
        
        model = torch.nn.Sequential(
            torch.nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True, num_layers=1),
            LambdaLayer(lambda x: x[0][:, -1, :]),  # Take the last output
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Tanh()
        )
        
        # Training parameters
        batch_size = 128
        
        # Create data loaders
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                    torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Train the model
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass - dealing with GRU's dual output
                pred = model(X_batch)
                
                # Reshape target to match prediction shape
                y_batch = y_batch.view(-1, 1)
                
                # Calculate loss
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            predictions = model(X_test_tensor).numpy()
            
    else:
        # Use custom models (Matquant variants)
        if model_class:
            model = model_class()
            
            # Train the model
            model.fit(X_train, y_train, epochs=epochs)
            
            # Generate predictions
            predictions = model.predict(X_test)
        else:
            raise ValueError("Model class not specified or not supported")
    
    # Convert predictions to trading signals (-1, 0, 1)
    # Flatten the predictions array if needed
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
    
    # Check if we should reverse the signals
    if model_config.get('reversed', False):
        predictions = -predictions
        
    # Create a DataFrame with the predictions
    pred_df = pd.DataFrame({
        'prediction': predictions
    }, index=test_data.index[len(test_data) - len(predictions):])
    
    # Apply threshold to get position: 1 for long, -1 for short, 0 for neutral
    threshold = 0.0  # Neutral zone threshold
    pred_df['position'] = 0
    pred_df.loc[pred_df['prediction'] > threshold, 'position'] = 1
    pred_df.loc[pred_df['prediction'] < -threshold, 'position'] = -1
    
    return pred_df

def calculate_metrics(test_data, predictions):
    """
    Calculate performance metrics for the strategy
    
    Args:
        test_data: Test data DataFrame
        predictions: DataFrame with predictions and positions
    
    Returns:
        Dictionary of metrics
    """
    # Ensure we have the same dates
    common_dates = test_data.index.intersection(predictions.index)
    test_data = test_data.loc[common_dates]
    predictions = predictions.loc[common_dates]
    
    # Calculate returns based on positions
    predictions['market_return'] = test_data['target']
    predictions['strategy_return'] = predictions['position'].shift(1) * predictions['market_return']
    
    # Remove NaN values
    predictions = predictions.dropna()
    
    if len(predictions) == 0:
        return {
            'sharpe_ratio': 0,
            'total_return': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'mse': 0,
            'n_trades': 0,
            'long_pct': 0,
            'short_pct': 0,
            'neutral_pct': 0
        }
    
    # Calculate metrics
    
    # MSE between prediction and actual return
    mse = np.mean((predictions['prediction'] - predictions['market_return'])**2)
    
    # Calculate cumulative returns
    predictions['cumulative_return'] = (1 + predictions['strategy_return']).cumprod() - 1
    
    # Sharpe ratio (assuming daily data, annualized)
    daily_returns = predictions['strategy_return']
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe_ratio = 0
    
    # Total return
    total_return = predictions['cumulative_return'].iloc[-1]
    
    # Win rate
    win_rate = (predictions['strategy_return'] > 0).sum() / len(predictions)
    
    # Maximum drawdown
    cum_returns = (1 + predictions['strategy_return']).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = abs(drawdown.min())
    
    # Position statistics
    n_trades = (predictions['position'] != predictions['position'].shift(1)).sum()
    long_pct = (predictions['position'] == 1).sum() / len(predictions)
    short_pct = (predictions['position'] == -1).sum() / len(predictions)
    neutral_pct = (predictions['position'] == 0).sum() / len(predictions)
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'mse': mse,
        'n_trades': n_trades,
        'long_pct': long_pct,
        'short_pct': short_pct,
        'neutral_pct': neutral_pct,
        'predictions': predictions  # Store predictions for further analysis
    }

def plot_results(results, results_dir):
    """
    Plot the results of the ablation tests
    
    Args:
        results: Dictionary of test results
        results_dir: Directory to save plots
    """
    # Extract metrics for plotting
    models = list(results.keys())
    
    # Create metrics lists
    sharpe_ratios = [results[model].get('sharpe_ratio', 0) for model in models]
    total_returns = [results[model].get('total_return', 0) * 100 for model in models]  # Convert to percentage
    win_rates = [results[model].get('win_rate', 0) * 100 for model in models]  # Convert to percentage
    max_drawdowns = [results[model].get('max_drawdown', 0) * 100 for model in models]  # Convert to percentage
    mse_values = [results[model].get('mse', 0) for model in models]
    n_trades = [results[model].get('n_trades', 0) for model in models]
    
    # Create a 3x2 subplot for all metrics
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # Sharpe Ratio
    axs[0, 0].bar(models, sharpe_ratios)
    axs[0, 0].set_title('Sharpe Ratio by Model')
    axs[0, 0].set_ylabel('Sharpe Ratio')
    axs[0, 0].set_xticklabels(models, rotation=45, ha='right')
    
    # Total Return
    axs[0, 1].bar(models, total_returns)
    axs[0, 1].set_title('Total Return by Model')
    axs[0, 1].set_ylabel('Total Return (%)')
    axs[0, 1].set_xticklabels(models, rotation=45, ha='right')
    
    # Win Rate
    axs[1, 0].bar(models, win_rates)
    axs[1, 0].set_title('Win Rate by Model')
    axs[1, 0].set_ylabel('Win Rate (%)')
    axs[1, 0].set_xticklabels(models, rotation=45, ha='right')
    
    # Max Drawdown
    axs[1, 1].bar(models, max_drawdowns)
    axs[1, 1].set_title('Max Drawdown by Model')
    axs[1, 1].set_ylabel('Max Drawdown (%)')
    axs[1, 1].set_xticklabels(models, rotation=45, ha='right')
    
    # MSE
    axs[2, 0].bar(models, mse_values)
    axs[2, 0].set_title('Mean Squared Error by Model')
    axs[2, 0].set_ylabel('MSE')
    axs[2, 0].set_xticklabels(models, rotation=45, ha='right')
    
    # Number of Trades
    axs[2, 1].bar(models, n_trades)
    axs[2, 1].set_title('Number of Trades by Model')
    axs[2, 1].set_ylabel('# Trades')
    axs[2, 1].set_xticklabels(models, rotation=45, ha='right')
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ablation_results.png'))
    print(f"Saved ablation results plot to {os.path.join(results_dir, 'ablation_results.png')}")
    
    # Generate a positions chart
    plt.figure(figsize=(12, 6))
    
    # Extract position metrics
    long_positions = [results[model].get('long_pct', 0) * 100 for model in models]
    short_positions = [results[model].get('short_pct', 0) * 100 for model in models]
    neutral_positions = [results[model].get('neutral_pct', 0) * 100 for model in models]
    
    # Create stacked bar chart for positions
    plt.bar(models, long_positions, label='Long')
    plt.bar(models, short_positions, bottom=long_positions, label='Short')
    plt.bar(models, neutral_positions, bottom=[long + short for long, short in zip(long_positions, short_positions)], label='Neutral')
    
    plt.title('Position Distribution by Model')
    plt.ylabel('Position Distribution (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'position_distribution.png'))
    print(f"Saved position distribution chart to {os.path.join(results_dir, 'position_distribution.png')}")

def main():
    parser = argparse.ArgumentParser(description='Fast MatQuant-Mamba Real Market Data Ablation Test')
    parser.add_argument('--data_file', type=str, default='data/SP 2.csv', help='Path to the data file')
    parser.add_argument('--sample_size', type=int, default=20000, help='Number of rows to sample')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--reversed_only', action='store_true', help='Only run with reversed signals (no regular evaluation)')
    args = parser.parse_args()
    
    # Run fast ablation test
    results = fast_ablation_test(
        data_file=args.data_file,
        sample_size=args.sample_size,
        epochs=args.epochs,
        reversed_only=args.reversed_only
    )
    
    # Print summary
    print("\n===== Fast Ablation Test Results Summary =====\n")
    print("Model\t\t\tSharpe Ratio\tTotal Return\tWin Rate")
    for model, result in results.items():
        sharpe = result.get('sharpe_ratio', 0)
        total_return = result.get('total_return', 0) * 100  # Convert to percentage
        win_rate = result.get('win_rate', 0) * 100  # Convert to percentage
        print(f"{model:20s}\t{sharpe:.4f}\t\t{total_return:.2f}%\t\t{win_rate:.2f}%")
    
    # Create a table comparing regular vs reversed models
    if not args.reversed_only:
        regular_models = [model for model in results.keys() if not model.endswith('_reversed')]
        
        print("\n===== Regular vs Reversed Signal Comparison =====\n")
        print("Model\t\t\tReg Return\tRev Return\tReg Sharpe\tRev Sharpe")
        
        for model in regular_models:
            reversed_model = f"{model}_reversed"
            if reversed_model in results:
                reg_return = results[model].get('total_return', 0) * 100
                rev_return = results[reversed_model].get('total_return', 0) * 100
                reg_sharpe = results[model].get('sharpe_ratio', 0)
                rev_sharpe = results[reversed_model].get('sharpe_ratio', 0)
                
                print(f"{model:20s}\t{reg_return:.2f}%\t\t{rev_return:.2f}%\t\t{reg_sharpe:.4f}\t\t{rev_sharpe:.4f}")
    
    print("\nResults have been saved to the 'fast_matquant_results' directory")

if __name__ == "__main__":
    # Filter warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main() 
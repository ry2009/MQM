#!/usr/bin/env python
"""
MatQuant-Market: Market Data Analysis with Matryoshka-Quantized Mamba
=================================================

This script implements a market data analysis pipeline using the MatQuant-Mamba architecture.
It focuses on raw price/volume data and market microstructure features for HFT applications.

Author: Ryan Mathieu
Date: 2025-02-18
"""

import os
import math
import argparse
from datetime import datetime
from typing import Tuple, List, Optional, Dict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np

# For logging
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field

# ---------------------------
# 1. Quantization & Utility Functions
# ---------------------------
def slice_operator(W: torch.Tensor, target_bits: int) -> torch.Tensor:
    """Applies MatQuant multi-scale slicing operator.
       Slice_k(W) = sign(W) * floor(|W| * 2^(8 - k))
    """
    assert target_bits in [2, 4, 8], "Target bits must be 2, 4, or 8."
    scale_factor = 2 ** (8 - target_bits)
    return torch.sign(W) * torch.floor(torch.abs(W) * scale_factor)

def co_distillation_loss(student_output, teacher_output, temperature=2.0):
    """
    Calculate KL divergence loss for co-distillation
    
    Args:
        student_output: Output from student model
        teacher_output: Output from teacher model
        temperature: Temperature for softening probability distributions
    
    Returns:
        KL divergence loss
    """
    # Ensure outputs have the same shape
    if student_output.shape != teacher_output.shape:
        # Reshape if needed
        if len(student_output.shape) == 1 and len(teacher_output.shape) == 1:
            student_output = student_output.view(-1, 1)
            teacher_output = teacher_output.view(-1, 1)
    
    # Soft targets
    student_output = student_output / temperature
    teacher_output = teacher_output / temperature
    
    # Calculate KL divergence
    loss = nn.MSELoss()(student_output, teacher_output) * (temperature ** 2)
    return loss

def compute_token_entropy(token_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Computes the entropy of a probability distribution."""
    token_probs = torch.clamp(token_probs, min=eps)
    return -torch.sum(token_probs * torch.log(token_probs), dim=-1)

def assign_bitwidths(entropy: torch.Tensor, threshold: float) -> torch.Tensor:
    """Dynamically assigns bit-width: 8 if entropy > threshold else 2."""
    return torch.where(entropy > threshold,
                       torch.tensor(8, device=entropy.device),
                       torch.tensor(2, device=entropy.device))

# ---------------------------
# 2. Market Data Processing
# ---------------------------
class MarketDataProcessor:
    def __init__(self, window_size: int, prediction_horizon: int):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        
    def process_market_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process raw market data into features and targets."""
        # Extract basic features
        features = []
        targets = []
        
        # Price and volume features
        prices = df['close'].values
        volumes = df['volume'].values
        
        # Calculate returns and volume changes
        returns = np.diff(prices) / prices[:-1]
        volume_changes = np.diff(volumes) / volumes[:-1]
        
        # Create sliding windows
        for i in range(len(returns) - self.window_size - self.prediction_horizon):
            # Input features
            window_returns = returns[i:i+self.window_size]
            window_volumes = volume_changes[i:i+self.window_size]
            
            # Combine features
            window_features = np.column_stack([
                window_returns,
                window_volumes
            ])
            
            # Target: future return
            target = returns[i+self.window_size:i+self.window_size+self.prediction_horizon]
            
            features.append(window_features)
            targets.append(target)
            
        return np.array(features), np.array(targets)

# ---------------------------
# 3. MatQuant-Market Model Components
# ---------------------------
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = int(expand * d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        residual = x
        x = self.norm(x)
        
        # Project to inner dimension
        x = self.in_proj(x)  # (batch_size, seq_len, d_inner)
        
        # Reshape for convolution
        batch_size, seq_len, d_inner = x.shape
        x = x.transpose(1, 2)  # (batch_size, d_inner, seq_len)
        
        # Apply convolution
        x = self.conv(x)  # (batch_size, d_inner, seq_len)
        
        # Reshape back
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_inner)
        
        # Project back to model dimension
        x = self.out_proj(x)  # (batch_size, seq_len, d_model)
        
        return x + residual

# --- Model Configuration ---
@dataclass
class ModelConfig:
    name: str
    hidden_dim: int = 128 # Default value
    num_layers: int = 6   # Default value
    quantization_type: str = 'none' # 'none', 'fixed', 'dynamic', 'distilled'
    bit_width: Optional[int] = None
    inference_latency_target: Optional[float] = None # Added for dynamic
    use_distillation: bool = False
    teacher_model: Optional[nn.Module] = None # Added for distillation

# --- Financial Metrics Tracking ---
class FinancialMetrics:
    def __init__(self, initial_capital=1_000_000, transaction_cost_pct=0.0005):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.positions = [] # Track position changes (e.g., +1 for long, -1 for short, 0 for flat)
        self.trade_log = [] # Log individual trades: (timestamp, type, size, price, pnl, holding_period)
        self.pnl_history = [initial_capital] # Track portfolio value over time
        self.latencies = {'inference': [], 'venue_ack': []} # Track latencies
        self.num_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.trade_pnls = [] # Store PnL for each closed trade

    def add_trade(self, predicted_return: float, actual_log_return: float, trade_size: float):
        # Simple strategy: Go long if predicted return > 0, short if < 0
        # For HFT, this would be much more complex (e.g., market making, arb)
        # This is a placeholder for demonstrating metric tracking
        
        self.num_trades += 1
        entry_price_multiplier = 1.0 # Assume entry at current price for simplicity
        exit_price_multiplier = np.exp(actual_log_return) # Price change based on actual log return

        # Calculate PnL for this single trade
        # Assuming we enter and exit within one time step for simplicity here
        cost = abs(trade_size) * self.transaction_cost_pct * 2 # Entry and exit cost
        
        # Determine direction based on prediction
        direction = 1 if predicted_return > 0 else -1
        
        # Calculate Gross PnL based on actual move and direction
        gross_pnl = direction * (exit_price_multiplier - entry_price_multiplier) * trade_size
        net_pnl = gross_pnl - cost
        
        self.trade_pnls.append(net_pnl)
        self.total_pnl += net_pnl
        self.current_capital += net_pnl
        self.pnl_history.append(self.current_capital)

        if net_pnl > 0:
            self.winning_trades += 1
        elif net_pnl < 0:
            self.losing_trades += 1
            
        # Basic trade logging (can be expanded)
        self.trade_log.append({
            'timestamp': time.time(), # Placeholder timestamp
            'direction': direction,
            'size': trade_size,
            'actual_log_return': actual_log_return,
            'predicted_return': predicted_return,
            'net_pnl': net_pnl
        })

    def add_latency(self, inference_latency: float, venue_ack_latency: float):
        self.latencies['inference'].append(inference_latency)
        self.latencies['venue_ack'].append(venue_ack_latency)

    def calculate_hft_metrics(self) -> Dict[str, float]:
        metrics = {}
        
        # --- Basic PnL & Return Metrics ---
        metrics['total_pnl'] = self.total_pnl
        metrics['final_capital'] = self.current_capital
        metrics['total_pnl_pct'] = (self.current_capital / self.initial_capital - 1) * 100 if self.initial_capital > 0 else 0
        metrics['initial_capital'] = self.initial_capital # Add for CSV compatibility

        # --- Calculate returns for Sharpe/Sortino ---
        returns = pd.Series(self.pnl_history).pct_change().dropna()
        
        # Add total/avg simple return for CSV compatibility
        metrics['total_simple_return'] = 0.0
        metrics['avg_simple_return'] = 0.0
        
        if not returns.empty and returns.std() != 0:
            # Assuming risk-free rate = 0, daily data -> sqrt(252) scaling
            # For HFT, scaling depends on trade frequency - let's use unscaled for now
            scaling_factor = 1 # Or adjust based on typical holding period/frequency
            
            # Portfolio Sharpe/Sortino
            metrics['portfolio_sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(scaling_factor) if returns.std() > 0 else 0
            
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else 1e-9
            metrics['portfolio_sortino_ratio'] = (returns.mean() / downside_std) * np.sqrt(scaling_factor) if downside_std > 0 else 0
            
            # Legacy metrics for CSV compatibility
            metrics['sharpe_ratio'] = metrics['portfolio_sharpe_ratio']
            metrics['sortino_ratio'] = metrics['portfolio_sortino_ratio']
        else:
            metrics['portfolio_sharpe_ratio'] = 0
            metrics['portfolio_sortino_ratio'] = 0
            metrics['sharpe_ratio'] = 0
            metrics['sortino_ratio'] = 0

        # Max Drawdown
        if len(self.pnl_history) > 1:
            pnl_series = pd.Series(self.pnl_history)
            cumulative_max = pnl_series.cummax()
            drawdown = (pnl_series - cumulative_max) / cumulative_max
            metrics['portfolio_max_drawdown'] = drawdown.min() * 100 if not drawdown.empty else 0
            metrics['max_drawdown'] = metrics['portfolio_max_drawdown']  # Legacy name for CSV
        else:
            metrics['portfolio_max_drawdown'] = 0
            metrics['max_drawdown'] = 0

        # --- Trade Metrics ---
        metrics['total_trades'] = self.num_trades
        metrics['winning_trades'] = self.winning_trades
        metrics['losing_trades'] = self.losing_trades
        
        # Calculate hit_rate directly (don't reference self.hit_rate)
        hit_rate = (self.winning_trades / self.num_trades) * 100 if self.num_trades > 0 else 0
        metrics['hit_rate'] = hit_rate
        metrics['win_rate'] = hit_rate  # Alias for clarity

        if self.trade_pnls:
            metrics['avg_trade_pnl'] = np.mean(self.trade_pnls)
            metrics['std_trade_pnl'] = np.std(self.trade_pnls)
             
            positive_pnls = [p for p in self.trade_pnls if p > 0]
            negative_pnls = [p for p in self.trade_pnls if p < 0]
             
            avg_win = np.mean(positive_pnls) if positive_pnls else 0
            avg_loss = np.mean(negative_pnls) if negative_pnls else 0
            metrics['avg_win_pnl'] = avg_win
            metrics['avg_loss_pnl'] = avg_loss # Typically negative
             
            total_profit = sum(positive_pnls)
            total_loss = abs(sum(negative_pnls)) if negative_pnls else 1
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else 0
            metrics['win_loss_ratio'] = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            metrics['avg_trade_pnl'] = 0
            metrics['std_trade_pnl'] = 0
            metrics['avg_win_pnl'] = 0
            metrics['avg_loss_pnl'] = 0
            metrics['profit_factor'] = 0
            metrics['win_loss_ratio'] = 0
            
        # Set default trade size for CSV compatibility
        metrics['avg_trade_size'] = 1000.0  # Default value

        # --- Latency Metrics ---
        # New format
        if self.latencies['inference']:
            metrics['avg_inference_latency_ms'] = np.mean(self.latencies['inference']) * 1000
            metrics['p95_inference_latency_ms'] = np.percentile(self.latencies['inference'], 95) * 1000
            metrics['max_inference_latency_ms'] = np.max(self.latencies['inference']) * 1000
            
            # Legacy format for CSV compatibility
            metrics['avg_latency'] = np.mean(self.latencies['inference'])
            metrics['max_latency'] = np.max(self.latencies['inference'])
            metrics['min_latency'] = np.min(self.latencies['inference'])
        else:
            metrics['avg_inference_latency_ms'] = 0
            metrics['p95_inference_latency_ms'] = 0
            metrics['max_inference_latency_ms'] = 0
            metrics['avg_latency'] = 0
            metrics['max_latency'] = 0
            metrics['min_latency'] = 0

        # Ensure no NaN/inf values are returned
        for key, value in metrics.items():
            if pd.isna(value) or np.isinf(value):
                metrics[key] = 0

        return metrics

class S6Layer(nn.Module):
    def __init__(self, hidden_dim):
        super(S6Layer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # S6 parameters
        self.Lambda = nn.Parameter(torch.zeros(hidden_dim))
        self.log_step = nn.Parameter(torch.zeros(()))
        
        # Initialize parameters
        nn.init.normal_(self.Lambda, mean=0.0, std=0.1)
        
        # Linear projections
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_dim]
        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_dim]
        """
        batch_size, seq_length, _ = x.shape
        
        # Project input
        input_projected = self.input_proj(x)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Initialize output
        outputs = []
        
        # Step size (discretization)
        step = torch.exp(self.log_step).clamp(min=1e-3, max=0.1)
        
        # Loop through sequence
        for t in range(seq_length):
            # Current input
            u_t = input_projected[:, t]
            
            # S6 state update
            delta = F.sigmoid(self.Lambda) * h + u_t
            h = h + step * delta
            
            # Apply output projection
            y_t = self.output_proj(h)
            outputs.append(y_t)
        
        # Stack outputs along sequence dimension
        return torch.stack(outputs, dim=1)

class MatQuantMambaBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(MatQuantMambaBlock, self).__init__()
        
        # Initialize S6 layer
        self.s6 = S6Layer(hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        
        # Skip connection
        self.skip = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_dim]
        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_dim]
        """
        # Layer norm 1
        h = self.norm1(x)
        
        # S6 layer
        h = self.s6(h)
        
        # Gating
        g = torch.sigmoid(self.gate(h))
        h = g * h
        
        # Skip connection
        skip = self.skip(x)
        
        # Residual connection
        h = h + skip
        
        # Layer norm 2
        h = self.norm2(h)
        
        return h

# --- MatQuant-Market Model ---
class MatQuantMarket(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, model_config: ModelConfig, num_symbols: Optional[int] = None):
        super().__init__()
        self.config = model_config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_symbols = num_symbols # Keep num_symbols
        
        # --- Restore Original Layers ---
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Symbol embedding layer (if multi-stock)
        if num_symbols is not None and num_symbols > 0:
            self.symbol_embedding = nn.Embedding(num_symbols, hidden_dim)
            # Need projection if concatenating embeddings
            self.symbol_projection = nn.Linear(hidden_dim * 2, hidden_dim) 
        else:
            self.symbol_embedding = None
            
        # MatQuant Mamba layers (using the original block)
        self.layers = nn.ModuleList([
            MatQuantMambaBlock(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output layer (original name)
        self.output = nn.Linear(hidden_dim, 1)
        
        # --- Teacher Model Handling (if distillation is used) ---
        # This part is mostly for setting up the teacher within the student
        # The actual teacher loading happens in the training script
        self.teacher_model = None
        if model_config.use_distillation and model_config.teacher_model:
            self.teacher_model = model_config.teacher_model
            # Freeze teacher model parameters if passed in
            # Note: The training script should handle loading the trained teacher
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def quantize(self, x: torch.Tensor, bit_width_signal: Optional[torch.Tensor] = None, force_bits: Optional[int] = None) -> torch.Tensor:
        """
        Quantize input tensor based on config, dynamic signal, or forced bits.
        Uses standard fixed/dynamic quantization logic for activations.
        For MatQuant weight slicing/interpolation, that logic would be applied
        to the model's weight parameters directly, not activations here.
        """
        effective_bits = force_bits if force_bits is not None else self.config.bit_width
        quant_type = self.config.quantization_type
        
        # If forcing bits for interpolation, treat as fixed quantization type for this step
        if force_bits is not None:
            quant_type = 'fixed'
            
        # --- No Quantization --- 
        if quant_type == 'none':
            return x
            
        # --- Dynamic Quantization --- 
        elif quant_type == 'dynamic':
            # Simplified dynamic logic using the provided signal (e.g., entropy-based)
            # Expects bit_width_signal shape: (batch_size, num_bit_options)
            if bit_width_signal is None:
                print("Warning: Dynamic quantization selected but no bit_width_signal provided. Applying default fixed bit-width.")
                quant_type = 'fixed' # Fallback to fixed
                effective_bits = self.config.bit_width
            else:
                # Example: Use softmax scores to weight different quantized versions
                bit_probs = F.softmax(bit_width_signal, dim=-1) # (batch_size, num_bit_options)
                possible_bits = [2, 4, 8] # Assume these options correspond to the signal
                num_options = bit_probs.shape[-1]
                
                quantized_xs = []
                for i, bits in enumerate(possible_bits):
                     if i >= num_options: break
                     # Apply fixed quantization for this bit option
                     max_val = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
                     scale = max_val / (2**(bits-1) - 1)
                     scale = torch.clamp(scale, min=1e-9)
                     x_q = torch.round(x / scale) * scale # Quantize-Dequantize
                     quantized_xs.append(x_q)
                     
                quantized_stack = torch.stack(quantized_xs, dim=-1) # (batch, ..., hidden_dim, num_options)
                bit_probs_expanded = bit_probs.view(*bit_probs.shape, 1).expand(*bit_probs.shape, x.shape[-1]) # Expand probs
                bit_probs_expanded = bit_probs_expanded.permute(0, 2, 1) # Match stack: (batch, hidden_dim, num_options)
                
                # Ensure dimensions match for broadcasting if needed (e.g., if x has seq_len)
                if quantized_stack.dim() > bit_probs_expanded.dim():
                     # Unsqueeze batch dim for broadcasting over seq_len etc.
                     bit_probs_expanded = bit_probs_expanded.unsqueeze(1).expand_as(quantized_stack)
                     
                x_quantized = torch.sum(quantized_stack * bit_probs_expanded, dim=-1) # Weighted sum
                return x_quantized

        # --- Fixed Quantization (or fallback from dynamic/interpolation) ---
        if quant_type == 'fixed':
            if effective_bits is None or effective_bits < 2:
                # print(f"Warning: Fixed quantization requested but invalid bits ({effective_bits}). Returning original.")
                return x
                
            # Standard fixed quantization for activations
            max_val = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] # Per-token max
            scale = max_val / (2**(effective_bits-1) - 1)
            scale = torch.clamp(scale, min=1e-9)
            x_quant = torch.round(x / scale) * scale # Quantize-Dequantize
            return x_quant
            
        # --- Distilled Quantization --- (Apply fixed quantization)
        elif quant_type == 'distilled':
            if effective_bits is None or effective_bits < 2:
                 return x # No quantization if bits not specified
            max_val = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
            scale = max_val / (2**(effective_bits-1) - 1)
            scale = torch.clamp(scale, min=1e-9)
            x_quant = torch.round(x / scale) * scale
            return x_quant
            
        else:
            print(f"Warning: Unknown quantization type '{quant_type}'. Returning original tensor.")
            return x

    def forward(self, x: torch.Tensor, symbol_ids: Optional[torch.Tensor] = None, force_bits: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass using the original MatQuantMambaBlock structure.
        Handles quantization of activations after embedding and layer outputs.
        """
        # Ensure input has feature dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # --- Dynamic Bit-width Signal Calculation (Placeholder) ---
        # This calculation depends on the specific dynamic strategy
        # We'll pass None for now, quantize function handles fallback if needed
        dynamic_bit_width_signal = None 
        # Add actual calculation here if dynamic quantization is used
        # e.g., based on input statistics or intermediate activations

        # Input embedding
        h = self.input_embedding(x) # (batch, seq, hidden)
        
        # Add symbol embedding if provided
        if self.symbol_embedding is not None and symbol_ids is not None:
            symbol_emb = self.symbol_embedding(symbol_ids) # (batch, hidden)
            symbol_emb = symbol_emb.unsqueeze(1).expand(-1, h.size(1), -1) # (batch, seq, hidden)
            h = torch.cat([h, symbol_emb], dim=-1) # (batch, seq, hidden * 2)
            h = self.symbol_projection(h) # (batch, seq, hidden)
        
        # Quantize after embedding (if applicable)
        if self.config.quantization_type != 'none' or force_bits is not None:
             h = self.quantize(h, dynamic_bit_width_signal, force_bits)
             
        # Apply MatQuant Mamba layers
        for layer in self.layers:
            h = layer(h)
            # Quantize output of each layer (if applicable)
            if self.config.quantization_type != 'none' or force_bits is not None:
                h = self.quantize(h, dynamic_bit_width_signal, force_bits) # Use same signal or recalculate
        
        # Take the last hidden state for prediction
        last_hidden_state = h[:, -1, :] # (batch, hidden)
        
        # Output layer
        output = self.output(last_hidden_state) # (batch, 1)
        
        # Ensure output shape is consistent (e.g., squeeze last dim if needed by loss)
        # Depending on the loss function, might need output.squeeze(-1) 
        # Keeping as (batch, 1) for now, compatible with MSELoss
        return output

# ---------------------------
# 4. Market Dataset
# ---------------------------
class MarketDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, prediction_horizon):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        
    def __len__(self):
        return len(self.data) - self.window_size - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.prediction_horizon, 0]  # Predict returns
        return x, y

# ---------------------------
# 5. Training Loop
# ---------------------------
def evaluate_model(model, val_loader, criterion, device):
    """Evaluate model performance on validation data."""
    model.eval()
    val_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            if model.model_config.use_distillation:
                outputs, _ = model(batch_x)
            else:
                outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    val_loss /= len(val_loader)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate financial metrics
    returns = predictions.flatten()  # Assuming predictions are returns
    actual_returns = actuals.flatten()
    
    metrics = {
        'validation_loss': val_loss,
        'sharpe_ratio': FinancialMetrics.calculate_sharpe_ratio(returns),
        'sortino_ratio': FinancialMetrics.calculate_sortino_ratio(returns),
        'hit_rate': FinancialMetrics.calculate_hit_rate(returns, actual_returns),
        'max_drawdown': FinancialMetrics.calculate_max_drawdown(returns)
    }
    
    return metrics

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, writer):
    """Enhanced training loop with comprehensive metrics tracking."""
    best_metrics = None
    temperature = 2.0  # For distillation
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            if model.model_config.use_distillation:
                student_output, teacher_output = model(batch_x)
                loss = criterion(student_output, batch_y)
                distill_loss = model.co_distillation_loss(teacher_output, student_output, temperature)
                loss = loss + distill_loss
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation with comprehensive metrics
        metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        for metric_name, value in metrics.items():
            writer.add_scalar(f'Metrics/{metric_name}', value, epoch)
        
        # Save best model based on Sharpe ratio
        if best_metrics is None or metrics['sharpe_ratio'] > best_metrics['sharpe_ratio']:
            best_metrics = metrics
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model.model_config,
                'metrics': metrics
            }, 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}')
            for metric_name, value in metrics.items():
                print(f'{metric_name}: {value:.4f}')
            print('---')

# ---------------------------
# 6. Main Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to market data CSV')
    parser.add_argument('--window_size', type=int, default=24, help='Input sequence length')
    parser.add_argument('--prediction_horizon', type=int, default=1, help='Number of steps to predict')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Mamba layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load and preprocess data
    data = pd.read_csv(args.data_path)
    data['returns'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    data = data.dropna()
    
    if len(data) == 0:
        raise ValueError("No data available after preprocessing")
    
    # Scale features
    scaler = StandardScaler()
    features = ['returns', 'volume_change']
    data[features] = scaler.fit_transform(data[features])
    
    # Create datasets
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    train_dataset = MarketDataset(train_data[features].values, args.window_size, args.prediction_horizon)
    val_dataset = MarketDataset(val_data[features].values, args.window_size, args.prediction_horizon)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MatQuantMarket(
        input_dim=len(features),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_config=ModelConfig(name="model", is_quantized=False, quantization_bits=None, use_distillation=False, is_dynamic=False),
        num_symbols=None
    ).to(device)
    
    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/matquant_market_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # Train model
    train_model(model, train_loader, val_loader, optimizer, criterion, args.num_epochs, device, writer)
    
    # Save results
    results = {
        'model_params': {
            'input_dim': len(features),
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'window_size': args.window_size,
            'prediction_horizon': args.prediction_horizon
        },
        'training_params': {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate
        }
    }
    
    pd.DataFrame([results]).to_csv('results/model_config.csv', index=False)

if __name__ == '__main__':
    main() 
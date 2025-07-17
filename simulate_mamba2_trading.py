#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import json

from matquant_mamba2 import MatryoshkaMamba2Model

# Set global default dtype
torch.set_default_dtype(torch.float32)

# --- Config --- #
DATA_PATH = "data/SP 2.csv"
CHECKPOINT_DIR = "mamba2_checkpoints"
RESULTS_DIR = "mamba2_trading_results"
TEST_SIZE = 0.2
INITIAL_CAPITAL = 1_000_000

# --- Utility Functions --- #
def calculate_sharpe(returns):
    mean = np.mean(returns)
    std = np.std(returns)
    if std < 1e-9:
        return 0.0
    return mean / std * np.sqrt(252 * 6.5 * 60 * 60)  # Annualize for second-level data (6.5h trading)

def calculate_max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown)

# --- Data Loading and Preprocessing --- #
def load_and_split_data():
    print("Loading and splitting data...")
    try:
        df = pd.read_csv(DATA_PATH)
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.sort_values('timestamp')
        df = df[df['volume'] > 0].dropna()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None
    print("Calculating base features...")
    prices = df['price'].values
    volumes = df['volume'].values
    features = {}
    features['log_returns'] = np.log(prices / np.roll(prices, 1))
    features['log_returns'][0] = 0
    features['price_change'] = np.diff(prices, prepend=prices[0])
    features['volume_change'] = np.diff(volumes, prepend=volumes[0])
    window_sizes = [5, 10, 20]
    for w in window_sizes:
        features[f'price_ma{w}'] = pd.Series(prices).rolling(window=w).mean().fillna(method='bfill').fillna(method='ffill').values
        features[f'volume_ma{w}'] = pd.Series(volumes).rolling(window=w).mean().fillna(method='bfill').fillna(method='ffill').values
        features[f'price_std{w}'] = pd.Series(prices).rolling(window=w).std().fillna(method='bfill').fillna(method='ffill').values
        features[f'return_consistency{w}'] = pd.Series(np.sign(features['log_returns'])).rolling(window=w).sum().fillna(0).values / w
    features['prices'] = prices
    features['volumes'] = volumes
    features['return_direction'] = np.sign(features['log_returns'])
    feature_list_pre_z = [
        features['log_returns'], features['price_change'], features['volume_change'],
        np.zeros_like(prices), np.zeros_like(volumes),
        features['return_direction'],
        features['return_consistency5'], features['return_consistency10'], features['return_consistency20'],
        features['price_ma5'], features['price_ma10'], features['price_ma20'],
        features['volume_ma5'], features['volume_ma10'], features['volume_ma20'],
        features['price_std5'], features['price_std20']
    ]
    feature_matrix_pre_z = np.column_stack(feature_list_pre_z)
    y = features['log_returns']
    original_prices_full = features['prices']
    original_volumes_full = features['volumes']
    valid_indices = np.all(np.isfinite(feature_matrix_pre_z), axis=1) & np.isfinite(y)
    X_pre_z = feature_matrix_pre_z[valid_indices]
    y_clean = y[valid_indices]
    original_prices_clean = original_prices_full[valid_indices]
    original_volumes_clean = original_volumes_full[valid_indices]
    print("Splitting data into Train/Val/Test...")
    X_train_pre_z, X_temp_pre_z, y_train, y_temp, prices_train, prices_temp, vols_train, vols_temp = train_test_split(
        X_pre_z, y_clean, original_prices_clean, original_volumes_clean, test_size=TEST_SIZE, shuffle=False
    )
    X_val_pre_z, X_test_pre_z, y_val, y_test, prices_val, prices_test, vols_val, vols_test = train_test_split(
        X_temp_pre_z, y_temp, prices_temp, vols_temp, test_size=0.5, shuffle=False
    )
    print(f"Test set shape (pre-zscore): X={X_test_pre_z.shape}, y={y_test.shape}")
    return X_test_pre_z, y_test, prices_test, vols_test

# --- Model/Scaler/Config Loading --- #
def load_model_and_scalers(config_name, device):
    model_path = os.path.join(CHECKPOINT_DIR, f'{config_name}_best.pth')
    scaler_path = os.path.join(CHECKPOINT_DIR, f'{config_name}_scaler.joblib')
    zscore_params_path = os.path.join(CHECKPOINT_DIR, f'{config_name}_zscore_params.joblib')
    config_path = os.path.join(CHECKPOINT_DIR, f'{config_name}_hyperparams.json')
    if not all(os.path.exists(p) for p in [model_path, scaler_path, zscore_params_path, config_path]):
        print(f"Error: Model, scaler, zscore params, or config not found for {config_name}")
        return None, None, None, None
    checkpoint = torch.load(model_path, map_location=device)
    scaler = joblib.load(scaler_path)
    zscore_params = joblib.load(zscore_params_path)
    with open(config_path, 'r') as f:
        mamba2_config = json.load(f)
    return checkpoint, scaler, zscore_params, mamba2_config

# --- Simulation --- #
def run_simulation(X_test_pre_z, y_test, prices_test, vols_test, args):
    config_name = f"test_run_final_norm_hack_d32_nL2_q5b"
    print(f"\n--- Running Trading Simulation for {config_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, scaler, zscore_params, mamba2_config = load_model_and_scalers(config_name, device)
    if checkpoint is None:
        return None
    print("Applying Z-score transformation to test data...")
    X_test = X_test_pre_z.copy()
    p_mean, p_std = zscore_params['price_mean'], zscore_params['price_std']
    v_mean, v_std = zscore_params['volume_mean'], zscore_params['volume_std']
    if p_std > 1e-9: X_test[:, 3] = (prices_test - p_mean) / p_std
    else: X_test[:, 3] = 0
    if v_std > 1e-9: X_test[:, 4] = (vols_test - v_mean) / v_std
    else: X_test[:, 4] = 0
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    print("Applying StandardScaler to test features...")
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    print("Initializing Mamba2 model...")
    input_dim = X_test_tensor.shape[1]
    output_dim = 1
    d_model = mamba2_config.get('d_state', 32)  # fallback to 32 if not present
    n_layer = 2  # Set to 2 as a default, or add to config if needed
    quantize_bit_width = args.bit_width
    model = MatryoshkaMamba2Model(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=d_model,
        n_layer=n_layer,
        mamba2_config=mamba2_config,
        quantize_bit_width=quantize_bit_width,
        device=device
    ).to(device)
    print("Loading weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Starting backtest simulation...")
    n_samples = len(X_test_tensor)
    positions = np.zeros(n_samples)
    portfolio_log_returns = np.zeros(n_samples)
    portfolio_value_history = np.zeros(n_samples + 1)
    portfolio_value_history[0] = INITIAL_CAPITAL
    trade_count = 0
    predictions_list = []
    min_holding_period = 5
    last_entry_time = -min_holding_period
    current_position = 0
    trade_threshold = 0.0001
    log_returns = y_test
    for t in range(n_samples):
        features_t = X_test_tensor[t].unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
        with torch.no_grad():
            pred = model(features_t).cpu().item()
        predictions_list.append(pred)
        if args.flip_strategy:
            pred = -pred
        if pred > trade_threshold and current_position <= 0 and t - last_entry_time >= min_holding_period:
            current_position = 1
            last_entry_time = t
            trade_count += 1
        elif pred < -trade_threshold and current_position >= 0 and t - last_entry_time >= min_holding_period:
            current_position = -1
            last_entry_time = t
            trade_count += 1
        positions[t] = current_position
        portfolio_log_returns[t] = current_position * log_returns[t]
        portfolio_value_history[t+1] = portfolio_value_history[t] * np.exp(portfolio_log_returns[t])
    returns = portfolio_log_returns
    sharpe = calculate_sharpe(returns)
    total_return = (portfolio_value_history[-1] / INITIAL_CAPITAL) - 1
    win_rate = np.mean(returns > 0)
    max_drawdown = calculate_max_drawdown(portfolio_value_history)
    print(f"Sharpe: {sharpe:.4f}, Total Return: {total_return*100:.2f}%, Win Rate: {win_rate*100:.2f}%, Max Drawdown: {max_drawdown*100:.2f}%")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, f"{config_name}_results.npz"),
             positions=positions, returns=returns, portfolio=portfolio_value_history, predictions=predictions_list)
    with open(os.path.join(RESULTS_DIR, f"{config_name}_metrics.txt"), "w") as f:
        f.write(f"Sharpe: {sharpe:.4f}\n")
        f.write(f"Total Return: {total_return*100:.2f}%\n")
        f.write(f"Win Rate: {win_rate*100:.2f}%\n")
        f.write(f"Max Drawdown: {max_drawdown*100:.2f}%\n")
        f.write(f"Trades: {trade_count}\n")
    plt.figure(figsize=(12,6))
    plt.plot(portfolio_value_history, label="Portfolio Value")
    plt.title(f"Mamba2 Trading Simulation: {config_name}")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{config_name}_portfolio.png"))
    print(f"Results saved to {RESULTS_DIR}")
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'trades': trade_count
    }

def main():
    parser = argparse.ArgumentParser(description='Simulate Mamba2 Trading on SP 2.csv')
    parser.add_argument('--bit-width', type=int, required=True, help='Quantization bit width of the model (e.g., 2, 4, 5, 8)')
    parser.add_argument('--flip-strategy', action='store_true', help='Flip trading signals (for ablation)')
    args = parser.parse_args()
    X_test_pre_z, y_test, prices_test, vols_test = load_and_split_data()
    if X_test_pre_z is None:
        print("Data loading failed.")
        return
    run_simulation(X_test_pre_z, y_test, prices_test, vols_test, args)

if __name__ == "__main__":
    main() 
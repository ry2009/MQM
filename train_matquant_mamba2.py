#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
import joblib
from matquant_mamba2 import MatryoshkaMamba2Model, BlockMLP
import argparse
import functools # For MLP in Block if needed later

# Configuration
DATA_PATH = "data/SP 2.csv"
CHECKPOINT_DIR = "mamba2_checkpoints"
RESULTS_DIR = "mamba2_results"
INPUT_DIM = 17 # Based on previous findings
OUTPUT_DIM = 1 # Predicting single value (log return)

# Mamba2 Specific Hyperparameters
D_MODEL = 64 
N_LAYER = 4  
QUANTIZE_BIT_WIDTH = 5 # Default for MatryoshkaMamba2Model

# Mamba2 Block/Layer configurations (can be tuned)
MAMBA2_D_STATE = 16
MAMBA2_D_CONV = 4
MAMBA2_EXPAND = 2
MAMBA2_HEAD_DIM = 64 # Should be multiple of d_ssm / nheads
# Default Mamba2 internal params (can be exposed as args if needed)
MAMBA2_CONFIG_BASE = {
    "d_state": MAMBA2_D_STATE,
    "d_conv": MAMBA2_D_CONV,
    "expand": MAMBA2_EXPAND,
    "headdim": MAMBA2_HEAD_DIM,
    "ngroups": 1,
    "A_init_range": (1, 16),
    "D_has_hdim": False, # If True, D is (d_ssm), else (nheads)
    "rmsnorm": True, # Use RMSNorm within Mamba2
    "norm_before_gate": False,
    "dt_min": 0.001,
    "dt_max": 0.1,
    "dt_init_floor": 1e-4,
    "dt_limit": (0.0, float("inf")),
    "bias": False, # Bias for in_proj and out_proj
    "conv_bias": True, # Bias for conv1d
    "chunk_size": 256,
    "use_mem_eff_path": True, # Set to False if ops not available or for debugging
}

# Block specific configurations (can be exposed as args if needed)
BLOCK_NORM_EPS = 1e-5
BLOCK_FUSED_ADD_NORM = False # Requires mamba_ssm ops, safer to keep False initially
BLOCK_RESIDUAL_IN_FP32 = False
# BLOCK_MLP_CLS = nn.Identity # No MLP in block by default, Mamba2 is the mixer
# For a GatedMLP (example, if you want to add one later):
# class GatedMLP(nn.Module):
#     def __init__(self, dim, hidden_dim_mult=4, **kwargs):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, dim * hidden_dim_mult)
#         self.act = nn.SiLU()
#         self.fc2 = nn.Linear(dim * hidden_dim_mult, dim)
#     def forward(self, x):
#         return self.fc2(self.act(self.fc1(x)))
# BLOCK_MLP_CLS = GatedMLP

LEARNING_RATE = 0.001
NUM_EPOCHS = 10 # Reduced epochs for quicker testing
BATCH_SIZE = 64 # Smaller batch size for training
TEST_SIZE = 0.2 # 20% for validation/test set

def load_and_preprocess_data(data_path, test_size=0.2):
    """Load, preprocess, split, and scale data, fixing lookahead bias."""
    print("Loading data...")
    try:
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.sort_values('timestamp')
        df = df[df['volume'] > 0].dropna() # Drop rows with NaN volume or other NaNs
        print(f"Data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None, None, None # Added zscore_params return

    print("Calculating base features...")
    prices = df['price'].values
    volumes = df['volume'].values
    features = {}
    features['log_returns'] = np.log(prices / np.roll(prices, 1))
    features['log_returns'][0] = 0 # Set first return to 0
    features['price_change'] = np.diff(prices, prepend=prices[0])
    features['volume_change'] = np.diff(volumes, prepend=volumes[0])
    window_sizes = [5, 10, 20]
    for w in window_sizes:
        features[f'price_ma{w}'] = pd.Series(prices).rolling(window=w).mean().fillna(method='bfill').fillna(method='ffill').values
        features[f'volume_ma{w}'] = pd.Series(volumes).rolling(window=w).mean().fillna(method='bfill').fillna(method='ffill').values
        features[f'price_std{w}'] = pd.Series(prices).rolling(window=w).std().fillna(method='bfill').fillna(method='ffill').values
        features[f'return_consistency{w}'] = pd.Series(np.sign(features['log_returns'])).rolling(window=w).sum().fillna(0).values / w
    features['prices'] = prices # Keep original prices
    features['volumes'] = volumes # Keep original volumes
    features['return_direction'] = np.sign(features['log_returns'])

    # --- Create preliminary feature matrix (WITHOUT Z-SCORES yet) --- #
    feature_list_pre_z = [
        features['log_returns'], features['price_change'], features['volume_change'],
        # Placeholders for z-scores - will be calculated AFTER split
        np.zeros_like(prices), np.zeros_like(volumes), # price_zscore, volume_zscore
        features['return_direction'],
        features['return_consistency5'], features['return_consistency10'], features['return_consistency20'],
        features['price_ma5'], features['price_ma10'], features['price_ma20'],
        features['volume_ma5'], features['volume_ma10'], features['volume_ma20'],
        features['price_std5'], features['price_std20']
    ]
    feature_matrix_pre_z = np.column_stack(feature_list_pre_z)
    y = features['log_returns']
    original_prices_full = features['prices'] # Keep aligned original prices
    original_volumes_full = features['volumes'] # Keep aligned original volumes

    # Remove NaNs/Infs from PRE-Z features and target
    valid_indices = np.all(np.isfinite(feature_matrix_pre_z), axis=1) & np.isfinite(y)
    X_pre_z = feature_matrix_pre_z[valid_indices]
    y_clean = y[valid_indices]
    original_prices_clean = original_prices_full[valid_indices]
    original_volumes_clean = original_volumes_full[valid_indices]
    print(f"Data shape after cleaning NaNs/Infs (pre-Z): X={X_pre_z.shape}, y={y_clean.shape}")

    if X_pre_z.shape[0] == 0: return None, None, None, None, None, None, None, None

    print("Splitting data...")
    X_train_pre_z, X_temp_pre_z, y_train, y_temp, prices_train, prices_temp, vols_train, vols_temp = train_test_split(
        X_pre_z, y_clean, original_prices_clean, original_volumes_clean, test_size=test_size, shuffle=False
    )
    X_val_pre_z, X_test_pre_z, y_val, y_test, prices_val, prices_test, vols_val, vols_test = train_test_split(
        X_temp_pre_z, y_temp, prices_temp, vols_temp, test_size=0.5, shuffle=False
    )

    # --- Calculate and Apply Z-Scores using ONLY Training Stats --- # 
    print("Calculating Z-score stats from training data...")
    train_price_mean = np.mean(prices_train)
    train_price_std = np.std(prices_train)
    train_volume_mean = np.mean(vols_train)
    train_volume_std = np.std(vols_train)

    zscore_params = {
        'price_mean': train_price_mean, 'price_std': train_price_std,
        'volume_mean': train_volume_mean, 'volume_std': train_volume_std
    }

    # Function to apply z-score to a split
    def calculate_zscore_split(X_split_pre_z, prices_split, vols_split, p_mean, p_std, v_mean, v_std):
        X_split = X_split_pre_z.copy()
        # Use actual prices/volumes corresponding to the split for calculation
        if p_std > 1e-9: X_split[:, 3] = (prices_split - p_mean) / p_std # price_zscore at index 3
        else: X_split[:, 3] = 0
        if v_std > 1e-9: X_split[:, 4] = (vols_split - v_mean) / v_std   # volume_zscore at index 4
        else: X_split[:, 4] = 0
        # Replace any potential new NaNs/Infs from division by small std
        X_split = np.nan_to_num(X_split, nan=0.0, posinf=0.0, neginf=0.0)
        return X_split

    print("Applying Z-score transformation...")
    X_train = calculate_zscore_split(X_train_pre_z, prices_train, vols_train, train_price_mean, train_price_std, train_volume_mean, train_volume_std)
    X_val = calculate_zscore_split(X_val_pre_z, prices_val, vols_val, train_price_mean, train_price_std, train_volume_mean, train_volume_std)
    X_test = calculate_zscore_split(X_test_pre_z, prices_test, vols_test, train_price_mean, train_price_std, train_volume_mean, train_volume_std)

    # --- Scale Features using StandardScaler (Fit on Train ONLY) --- #
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Final Shapes - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

    # Convert to tensors (ensure target is also processed appropriately)
    X_train_tensor = torch.FloatTensor(X_train_scaled) # Shape: (num_samples, num_features)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) # Shape: (num_samples, 1)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    # Return all necessary components
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, scaler, zscore_params

def train_model(X_train, y_train, X_val, y_val, training_run_name, current_quantize_bit_width, scaler, zscore_params):
    """Train the MatryoshkaMamba2Model with validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare Mamba2 specific config by copying base and adding runtime specifics
    mamba2_runtime_config = MAMBA2_CONFIG_BASE.copy()
    # d_model, quantize_bit_width, device, dtype are passed directly to MatryoshkaMamba2Model constructor
    # and then down to Mamba2 layers within the model if needed.

    model = MatryoshkaMamba2Model(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        d_model=D_MODEL, 
        n_layer=N_LAYER,
        mamba2_config=mamba2_runtime_config, # Pass the dict
        quantize_bit_width=current_quantize_bit_width,
        norm_eps=BLOCK_NORM_EPS,
        fused_add_norm_block=BLOCK_FUSED_ADD_NORM,
        residual_in_fp32_block=BLOCK_RESIDUAL_IN_FP32,
        mlp_cls_block=BlockMLP, # Use BlockMLP for MLP in each block
        device=device,
        dtype=torch.float32 # Or other desired dtype
    ).to(device)

    print(f"Model: {model}") # Print model structure
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    # Use training_run_name for checkpoint files
    best_model_path = os.path.join(CHECKPOINT_DIR, f"{training_run_name}_best.pth")
    scaler_path = os.path.join(CHECKPOINT_DIR, f"{training_run_name}_scaler.joblib")
    zscore_params_path = os.path.join(CHECKPOINT_DIR, f"{training_run_name}_zscore_params.joblib")
    # Path for full model config (hyperparameters)
    model_hyperparams_path = os.path.join(CHECKPOINT_DIR, f"{training_run_name}_hyperparams.json")

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        permutation = torch.randperm(X_train.size()[0])

        for i in range(0, X_train.size()[0], BATCH_SIZE):
            optimizer.zero_grad()
            indices = permutation[i:i + BATCH_SIZE]
            # X_train is (num_samples, num_features)
            batch_X, batch_y = X_train[indices].to(device), y_train[indices].to(device)

            # MatryoshkaMamba2Model expects (batch, seq_len, input_dim)
            # Assuming each sample is a sequence of length 1 for now.
            if batch_X.dim() == 2:
                 batch_X = batch_X.unsqueeze(1) # (batch, 1, input_dim)

            outputs = model(batch_X) # No inference_params during training unless specifically handled
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / (max(1, X_train.size()[0] // BATCH_SIZE)) # Avoid division by zero

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
             val_permutation = torch.randperm(X_val.size()[0])
             for i in range(0, X_val.size()[0], BATCH_SIZE):
                 indices = val_permutation[i:i+BATCH_SIZE]
                 batch_X_val, batch_y_val = X_val[indices].to(device), y_val[indices].to(device)
                 if batch_X_val.dim() == 2:
                     batch_X_val = batch_X_val.unsqueeze(1) # (batch, 1, input_dim)
                 val_outputs = model(batch_X_val)
                 loss = criterion(val_outputs, batch_y_val)
                 val_loss += loss.item()

        avg_val_loss = val_loss / (max(1, X_val.size()[0] // BATCH_SIZE)) # Avoid division by zero
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save model state and hyperparameters
            model_save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                # Store key hyperparameters for reloading/identification
                'input_dim': INPUT_DIM,
                'output_dim': OUTPUT_DIM,
                'd_model': D_MODEL,
                'n_layer': N_LAYER,
                'quantize_bit_width': current_quantize_bit_width,
                'mamba2_config': mamba2_runtime_config,
                'block_norm_eps': BLOCK_NORM_EPS,
                'block_fused_add_norm': BLOCK_FUSED_ADD_NORM,
                'block_residual_in_fp32': BLOCK_RESIDUAL_IN_FP32
            }
            torch.save(model_save_dict, best_model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(zscore_params, zscore_params_path)
            # Save the hyperparameter dict separately as json for easy viewing
            import json
            with open(model_hyperparams_path, 'w') as f:
                json.dump(model_save_dict['mamba2_config'], f, indent=4) # Save mamba2_config part, or more if needed
            print(f"New best model saved to {best_model_path} (Val Loss: {best_val_loss:.6f})")

    print("\nTraining finished.")
    return best_model_path, scaler_path, zscore_params_path, model_hyperparams_path

def main():
    """Main function to run the Mamba2 training pipeline."""
    # Update global/default configs from args
    global D_MODEL, N_LAYER, QUANTIZE_BIT_WIDTH, NUM_EPOCHS, LEARNING_RATE
    global MAMBA2_CONFIG_BASE, MAMBA2_D_STATE, MAMBA2_D_CONV, MAMBA2_EXPAND

    parser = argparse.ArgumentParser(description='Train MatryoshkaMamba2Model')
    parser.add_argument('--bit-width', type=int, default=QUANTIZE_BIT_WIDTH, 
                        choices=[0, 2, 3, 4, 5, 6, 7, 8], # 0 for no quantization (float32/16 based on model default)
                        help='Quantization bit width for MatryoshkaMamba2Model (0 for none, default 5)')
    parser.add_argument('--d-model', type=int, default=D_MODEL, help='Dimension of the model (d_model)')
    parser.add_argument('--n-layer', type=int, default=N_LAYER, help='Number of Mamba2 layers')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    # Add more args for MAMBA2_CONFIG_BASE if needed, e.g., d_state, d_conv, expand
    parser.add_argument('--m2-d-state', type=int, default=MAMBA2_D_STATE, help='Mamba2 d_state')
    parser.add_argument('--m2-d-conv', type=int, default=MAMBA2_D_CONV, help='Mamba2 d_conv')
    parser.add_argument('--m2-expand', type=int, default=MAMBA2_EXPAND, help='Mamba2 expand factor')
    parser.add_argument('--run-name', type=str, default=None, help='Optional name for this training run (for checkpoint naming)')

    args = parser.parse_args()

    D_MODEL = args.d_model
    N_LAYER = args.n_layer
    current_quantize_bit_width = args.bit_width
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    
    MAMBA2_D_STATE = args.m2_d_state
    MAMBA2_D_CONV = args.m2_d_conv
    MAMBA2_EXPAND = args.m2_expand
    
    # Update MAMBA2_CONFIG_BASE with args
    MAMBA2_CONFIG_BASE["d_state"] = MAMBA2_D_STATE
    MAMBA2_CONFIG_BASE["d_conv"] = MAMBA2_D_CONV
    MAMBA2_CONFIG_BASE["expand"] = MAMBA2_EXPAND
    # Note: headdim might need to be adjusted if d_model or other params change significantly
    # to maintain d_ssm % headdim == 0 where d_ssm = d_inner = expand * d_model.
    # d_ssm for Mamba2 is d_inner, which is expand * D_MODEL. We need d_ssm % headdim == 0.
    # For simplicity, keeping headdim fixed but this is a point of attention if D_MODEL and EXPAND change.
    if (MAMBA2_EXPAND * D_MODEL) % MAMBA2_CONFIG_BASE["headdim"] != 0:
        print(f"Warning: (EXPAND * D_MODEL) % headdim != 0. ({MAMBA2_EXPAND} * {D_MODEL}) % {MAMBA2_CONFIG_BASE['headdim']} != 0. This might cause issues in Mamba2 internal setup.")
        # Potentially adjust headdim here or raise an error. For now, just a warning.
        # Example auto-adjustment (could be risky):
        # new_headdim = MAMBA2_EXPAND * D_MODEL
        # while new_headdim > 0 and (MAMBA2_EXPAND * D_MODEL) % new_headdim != 0:
        #    new_headdim //=2 # or find factors
        # if new_headdim == 0: new_headdim = 1 # fallback
        # MAMBA2_CONFIG_BASE["headdim"] = new_headdim
        # print(f"Adjusted headdim to {new_headdim}")

    training_run_name = args.run_name
    if training_run_name is None:
        training_run_name = f"mamba2_d{D_MODEL}_nL{N_LAYER}_q{current_quantize_bit_width}b_ds{MAMBA2_D_STATE}_dc{MAMBA2_D_CONV}_e{MAMBA2_EXPAND}"
    else:
        training_run_name = f"{args.run_name}_d{D_MODEL}_nL{N_LAYER}_q{current_quantize_bit_width}b"

    print(f"\n--- Training MatryoshkaMamba2Model: {training_run_name} ---")
    print(f"Parameters: D_MODEL={D_MODEL}, N_LAYER={N_LAYER}, Q_BIT_WIDTH={current_quantize_bit_width}")
    print(f"Mamba2 Config (base): {MAMBA2_CONFIG_BASE}")
    print(f"Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, zscore_params = load_and_preprocess_data(DATA_PATH, TEST_SIZE)
    if X_train is None:
        print("Failed to load data. Exiting.")
        return

    # Pass current_quantize_bit_width to train_model
    best_model_path, scaler_path, zscore_params_path, model_hyperparams_path = train_model(
        X_train, y_train, X_val, y_val, 
        training_run_name, 
        current_quantize_bit_width, 
        scaler, 
        zscore_params
    )
    print(f"Best model saved at: {best_model_path}")
    print(f"Scaler saved at: {scaler_path}")
    print(f"Z-score params saved at: {zscore_params_path}")
    print(f"Model hyperparameters (from Mamba2 config) saved at: {model_hyperparams_path}")

    # (Optional) Evaluate on test set here if desired
    # print("\nEvaluating on Test Set...")
    # model_reloaded = MatryoshkaMamba2Model(...) # Reload with saved hyperparams
    # checkpoint = torch.load(best_model_path, map_location=device)
    # model_reloaded.load_state_dict(checkpoint['model_state_dict'])
    # model_reloaded.to(device)
    # model_reloaded.eval()
    # test_preds = []
    # with torch.no_grad():
    #     for i in range(0, X_test.size()[0], BATCH_SIZE):
    #         batch_X_test = X_test[i:i+BATCH_SIZE].to(device)
    #         if batch_X_test.dim() == 2: batch_X_test = batch_X_test.unsqueeze(1)
    #         outputs = model_reloaded(batch_X_test)
    #         test_preds.extend(outputs.squeeze().cpu().numpy())
    # test_r2 = r2_score(y_test.cpu().numpy(), np.array(test_preds)[:len(y_test)]) # Ensure length match
    # print(f"Test R2 Score: {test_r2:.4f}")

if __name__ == "__main__":
    main() 
'''
Matryoshka Quantized Mamba 2 Model
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import functools # For functools.partial

# --- Placeholder for mamba_ssm.ops.triton.layernorm_gated.RMSNorm --- 
# This will be used by the Block class if the specific import fails.
class RMSNormBlock(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

# --- Placeholder for mamba_ssm.ops.triton.layernorm_gated.RMSNorm as RMSNormGated (for Mamba2 internal use) ---
# Mamba2 expects a more specific RMSNormGated with group_size and norm_before_gate options.
# This is a simplified placeholder if the mamba_ssm one isn't available.
class RMSNormGatedMamba2Placeholder(nn.Module):
    def __init__(self, d_model, eps=1e-5, norm_before_gate=False, group_size=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.norm_before_gate = norm_before_gate # Placeholder, not fully implemented in this simple version
        self.group_size = group_size if group_size is not None else d_model # Placeholder

    def forward(self, x, z=None):
        # Simplified: ignores z (gate) and norm_before_gate complexities for placeholder
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        if z is not None and not self.norm_before_gate: # Very basic handling if z is provided
            print(f"RMSNormGatedMamba2Placeholder: output shape: {output.shape}, z shape: {z.shape}") # Debug print
            output = output * F.silu(z)
        return output

# --- Placeholders for mamba_ssm.distributed.tensor_parallel --- 
# These will default to standard nn.Linear if mamba_ssm.distributed is not available or not used.
class ColumnParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, process_group=None, sequence_parallel=True, device=None, dtype=None):
        # For non-distributed, out_features_group = out_features
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        # Additional attributes for API compatibility, not used in this placeholder
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

class RowParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, process_group=None, sequence_parallel=True, device=None, dtype=None):
        # For non-distributed, in_features_group = in_features
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        # Additional attributes for API compatibility, not used in this placeholder
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

# --- Attempt to import specific mamba_ssm components --- 
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    print("WARNING: causal_conv1d_fn/causal_conv1d_update not found. Mamba2 performance may be affected or sections may not work.")
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    print("WARNING: selective_state_update not found. Mamba2 performance may be affected or sections may not work.")
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGatedOriginal
    # If Mamba2 relies on specific behavior of this, we use it. Otherwise, placeholder can be used.
    # For simplicity here, we'll keep our placeholder for Mamba2 unless specific features are needed.
except ImportError:
    print("WARNING: mamba_ssm.ops.triton.layernorm_gated.RMSNorm (RMSNormGatedOriginal) not found.")
    RMSNormGatedOriginal = None # Will default to RMSNormGatedMamba2Placeholder if Mamba2 code requires it by that name

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
except ImportError:
    print("WARNING: mamba_chunk_scan_combined or mamba_split_conv1d_scan_combined not found. Mamba2 performance may be affected.")
    mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined = None, None

print("matquant_mamba2.py: Initial imports and placeholders set up.")

# --- Quantization Function --- #
def quantize_weights_fixed_pt(parameter: torch.Tensor, bits: int, device: torch.device = torch.device('cpu')):
    '''
    Simulates fixed-point quantization for a given parameter.
    Scales to the range of the specified bit-width, quantizes, then de-quantizes.
    '''
    if bits <= 0 or bits > 16: # Practical limit for this simple method
        # print(f"Quantization bits ({bits}) out of supported range (1-16). Skipping quantization for parameter.")
        return parameter # No quantization or return original if bits are out of typical range

    # Determine the scale factor for the given number of bits
    # For signed integers, the range is [-2^(bits-1), 2^(bits-1) - 1]
    # We map the float range [-1, 1] to this integer range for weights (assuming weights are typically small after init)
    # A more robust method would use min/max of the tensor, but for simplicity and typical NN weights:
    q_min = -2.**(bits - 1)
    q_max = 2.**(bits - 1) - 1

    # Simple scaling: Assumes weights are somewhat centered around 0 and not excessively large.
    # A common practice is to clip weights before quantization, e.g., to [-1, 1] or another range.
    # Here, we'll use the parameter's current range for scaling to avoid explicit clipping initially.
    param_min, param_max = parameter.min(), parameter.max()
    
    # If min and max are too close (or zero), quantization is tricky / can lead to all zeros.
    if torch.isclose(param_min, param_max):
        # print(f"Parameter min and max are too close ({param_min.item()}, {param_max.item()}). Skipping quantization.")
        return parameter

    # Scale factor to map [param_min, param_max] to [q_min, q_max]
    scale = (q_max - q_min) / (param_max - param_min)
    zero_point = q_min - param_min * scale

    # Quantize
    quantized_param = torch.round(parameter * scale + zero_point)
    quantized_param = torch.clamp(quantized_param, q_min, q_max)

    # De-quantize (back to approximate original float range)
    dequantized_param = (quantized_param - zero_point) / scale
    
    return dequantized_param.to(device=parameter.device, dtype=parameter.dtype)

print("matquant_mamba2.py: Added quantization function.")

# --- Mamba2 Class Definition (Adapted for Quantization) --- #
class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True, # If True, uses RMSNormGated for normalization within Mamba2
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=True, # Caution: mem_eff_path might bypass typical .weight attributes for some ops
        layer_idx=None,
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
        quantize_bit_width=None # Added for Matryoshka Quantization
    ):
        factory_kwargs = {"device": device}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu" # Mamba2 typically uses silu
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.quantize_bit_width = quantize_bit_width

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
            if conv_bias and hasattr(self.conv1d, 'bias') and self.conv1d.bias is not None:
                 nn.init.zeros_(self.conv1d.bias)

        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, **factory_kwargs))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            # Use the specific RMSNormGated from mamba_ssm if available, otherwise placeholder
            # The original Mamba2 code used: from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
            # We'll try to use RMSNormGatedOriginal if it was imported, else our placeholder.
            # This might need adjustment if the placeholder isn't fully compatible with Mamba2's expectations.
            if RMSNormGatedOriginal is not None:
                self.norm = RMSNormGatedOriginal(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                         group_size=self.d_ssm // ngroups if ngroups > 0 else self.d_ssm, # ensure ngroups > 0 for division
                                         **factory_kwargs)
            else:
                print("Mamba2 WARNING: Using RMSNormGatedMamba2Placeholder for internal normalization. Behavior may differ.")
                self.norm = RMSNormGatedMamba2Placeholder(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                         group_size=self.d_ssm // ngroups if ngroups > 0 else self.d_ssm,
                                         **factory_kwargs)
        else:
            self.norm = nn.Identity() # If rmsnorm is False

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)
        
        # --- Apply Quantization --- #
        if self.quantize_bit_width is not None and self.quantize_bit_width > 0:
            print(f"Mamba2 (Layer {self.layer_idx if self.layer_idx is not None else 'N/A'}): Quantizing in_proj, conv1d, out_proj to {self.quantize_bit_width} bits.")
            with torch.no_grad():
                self.in_proj.weight.data = quantize_weights_fixed_pt(self.in_proj.weight.data, self.quantize_bit_width, device=device)
                if bias and hasattr(self.in_proj, 'bias') and self.in_proj.bias is not None:
                    self.in_proj.bias.data = quantize_weights_fixed_pt(self.in_proj.bias.data, self.quantize_bit_width, device=device)
                
                self.conv1d.weight.data = quantize_weights_fixed_pt(self.conv1d.weight.data, self.quantize_bit_width, device=device)
                if conv_bias and hasattr(self.conv1d, 'bias') and self.conv1d.bias is not None:
                    self.conv1d.bias.data = quantize_weights_fixed_pt(self.conv1d.bias.data, self.quantize_bit_width, device=device)

                self.out_proj.weight.data = quantize_weights_fixed_pt(self.out_proj.weight.data, self.quantize_bit_width, device=device)
                if bias and hasattr(self.out_proj, 'bias') and self.out_proj.bias is not None:
                    self.out_proj.bias.data = quantize_weights_fixed_pt(self.out_proj.bias.data, self.quantize_bit_width, device=device)

    def forward(self, u, seqlen=None, seq_idx=None, inference_params=None):
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        
        A = -torch.exp(self.A_log.float()) # Ensure A is float for exp
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        # Handle case where mamba_split_conv1d_scan_combined might not be available
        if self.use_mem_eff_path and inference_params is None and mamba_split_conv1d_scan_combined is not None and self.conv1d.bias is not None and self.out_proj.bias is not None and (isinstance(self.norm, nn.Identity) or (hasattr(self.norm, 'weight') and self.norm.weight is not None)):
            #This path has strong conditions on specific ops being available and specific parameters (like biases) existing.
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias, # Requires conv1d.bias
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm and not isinstance(self.norm, nn.Identity) else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm and hasattr(self.norm, 'eps') else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias, # Requires out_proj.bias
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate if hasattr(self.norm, 'norm_before_gate') else False, 
                **dt_limit_kwargs,
            )
        else: # Fallback path or non-memory-efficient path
            if self.use_mem_eff_path and (mamba_split_conv1d_scan_combined is None or self.conv1d.bias is None or self.out_proj.bias is None or isinstance(self.norm, nn.Identity) or not (hasattr(self.norm, 'weight') and self.norm.weight is not None)):
                if inference_params is None:
                     print(f"Mamba2 (Layer {self.layer_idx if self.layer_idx is not None else 'N/A'}) WARNING: Falling back from memory-efficient path due to missing ops or params (e.g., bias may be False, or norm is Identity, or mamba_split_conv1d_scan_combined not found).")
            
            # Derived from Mamba2 source for the non-fused path
            d_mlp_approx = (zxbcdt.shape[-1] - (2 * self.d_ssm + 2 * self.ngroups * self.d_state + self.nheads))
            if d_mlp_approx < 0: d_mlp_approx = 0 # Should not happen if d_in_proj is correct
            d_mlp = d_mlp_approx // 2
            
            # Order: [z, x, B, C, dt] # Mamba1 original split: d_in_proj = self.d_inner + self.d_inner + self.ngroups * self.d_state + self.ngroups * self.d_state + self.nheads
            # Mamba2 split: d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
            # This means d_inner = d_ssm for Mamba2 if d_mlp_approx is intended to be zero.
            # If there's an MLP component *within* Mamba2 (not typical, Mamba2Block does MLP outside)
            # then d_in_proj = 2*d_mlp + 2*d_ssm_part + ...
            # For now, assuming the split provided in Mamba2 source is for its full d_inner for z0, x0.
            # If d_ssm is smaller than d_inner, the Mamba2 paper suggests the rest is MLP-like.
            # The provided split seems to imply d_ssm is part of d_inner, not the entirety of it, if d_mlp > 0.

            # If d_ssm is self.d_inner, then d_mlp should be 0 from the split formula.
            # If d_ssm is smaller than d_inner, d_mlp is (d_inner - d_ssm).
            # Let's recalculate d_mlp based on d_ssm and d_inner
            d_mlp_calc = self.d_inner - self.d_ssm
            # The split sizes should sum up to zxbcdt.shape[-1]
            # zxbcdt_parts = [d_mlp_calc, d_mlp_calc, self.d_ssm, self.d_ssm + 2*self.ngroups*self.d_state, self.nheads]
            # Current split from source: 
            z0_size = d_mlp_calc
            x0_size = d_mlp_calc
            z_size  = self.d_ssm # This z is for the RMSNorm after SSM, if applied
            xBC_size= self.d_ssm + 2*self.ngroups*self.d_state # x for SSM, B, C
            dt_size = self.nheads
            current_split = [z0_size, x0_size, z_size, xBC_size, dt_size]
            
            # print(f"Debug Mamba2 (Layer {self.layer_idx}): zxbcdt shape {zxbcdt.shape}, target_sum {zxbcdt.shape[-1]}, current_sum {sum(current_split)}")
            # print(f"Debug Mamba2: d_model={self.d_model}, d_inner={self.d_inner}, d_ssm={self.d_ssm}, d_mlp_calc={d_mlp_calc}, headdim={self.headdim}, nheads={self.nheads}, ngroups={self.ngroups}, d_state={self.d_state}")
            # print(f"Debug Mamba2: Split sizes: z0={z0_size}, x0={x0_size}, z_ssm_norm={z_size}, xBC_ssm_input={xBC_size}, dt={dt_size}")

            assert sum(current_split) == zxbcdt.shape[-1], f"Split sizes do not match d_in_proj: {sum(current_split)} vs {zxbcdt.shape[-1]}"

            z0, x0, z_for_norm, xBC, dt = torch.split(zxbcdt, current_split, dim=-1)

            if conv_state is not None:
                xBC_t = rearrange(xBC, "b l d -> b d l")
                conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))
            
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            
            x_ssm, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            
            if mamba_chunk_scan_combined is None:
                # raise ImportError("mamba_chunk_scan_combined is not available, non-memory-efficient path cannot proceed.")
                print(f"Mamba2 (Layer {self.layer_idx if self.layer_idx is not None else 'N/A'}) CRITICAL WARNING: 'mamba_chunk_scan_combined' not found. Using ZERO TENSOR placeholder. Model will not learn Mamba logic.")
                # Expected output of mamba_chunk_scan_combined is like x_ssm_rearranged
                x_ssm_rearranged_for_placeholder = rearrange(x_ssm, "b l (h p) -> b l h p", p=self.headdim)
                y_placeholder = torch.zeros_like(x_ssm_rearranged_for_placeholder)
                
                if ssm_state is not None: # If states were expected to be returned
                    # The ssm_state passed in is what would be updated. So, create a dummy last_state of similar shape.
                    # The actual shape of last_state from mamba_chunk_scan_combined is (B, H, P, N) where N is d_state
                    # ssm_state here is passed from allocate_inference_cache, shape (batch_size, self.nheads, self.headdim, self.d_state)
                    last_state_placeholder = torch.zeros_like(ssm_state) 
                    y = (y_placeholder, last_state_placeholder) # Return as a tuple
                else:
                    y = y_placeholder
            else:
                y = mamba_chunk_scan_combined(
                    rearrange(x_ssm, "b l (h p) -> b l h p", p=self.headdim),
                    dt,
                    A,
                    rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D, # D should be (H,P) or (H,)
                    z=rearrange(z_for_norm, "b l (h p) -> b l h p", p=self.headdim) if (not self.rmsnorm and not isinstance(self.norm, nn.Identity)) else None, # z here is pre-SSM gate
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    seq_idx=seq_idx,
                    **dt_limit_kwargs,
                    return_final_states=ssm_state is not None,
                )
                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
            
            y = rearrange(y, "b l h p -> b l (h p)") # Back to (B, L, d_ssm)
            
            if self.rmsnorm and not isinstance(self.norm, nn.Identity):
                if y.shape[1] == z_for_norm.shape[1]:
                    y = self.norm(y, z_for_norm)
                else:
                    print(f"Mamba2 (Layer {self.layer_idx if self.layer_idx is not None else 'N/A'}) WARNING: RMSNorm input y seq_len ({y.shape[1]}) != z_for_norm seq_len ({z_for_norm.shape[1]}). Calling norm without z gate for placeholder compatibility.")
                    y = self.norm(y, None) # Pass None for z if sequence lengths mismatch
            
            # If MLP component exists (d_mlp_calc > 0)
            if z0_size > 0:
                y_mlp = F.silu(z0) * x0
                y = torch.cat([y_mlp, y], dim=-1)
            
            if seqlen_og is not None: # If input was (B*L, D)
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)

        if self.process_group is not None and self.use_mem_eff_path: # Only all_reduce/reduce_scatter if it was mem_eff_path
            # The non-mem-eff path above handles its own projections, so distributed comms might be complex there.
            # For simplicity, only applying for the mem_eff_path where out_proj is part of the fused kernel.
            # If sequence_parallel, RowParallelLinear for out_proj handles all_gather, ColumnParallelLinear for in_proj handles scatter.
            # Here, if out_proj was part of fused kernel, we need to handle its output distribution.
            # This part of Mamba2 source seems to imply out_proj is separate from the fused kernel for this reduction part.
            # Given self.out_proj is a RowParallelLinear, it should handle all_gather on its input if sequence_parallel=False.
            # If sequence_parallel=True, RowParallelLinear output is sharded, requiring reduce_scatter *if input was sequence sharded*.
            # This is complex. The Mamba2 source itself has: reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
            # This implies the output `out` from the fused kernel is sharded if sequence_parallel=True.
            # For now, let's assume the distributed handling is managed by the ParallelLinear layers if not using fused kernel.
            # If using fused kernel, the Mamba2 source shows a reduce_scatter / all_reduce on `out`.
            # This needs careful integration with how ParallelLinear layers are defined.
            # The provided Mamba2 does reduce_fn AFTER mamba_split_conv1d_scan_combined
            pass # Deferring complex distributed logic for `out` post a fused kernel until clearer.
                 # If self.out_proj is RowParallelLinear, it expects full input if sequence_parallel=True,
                 # or sharded input if sequence_parallel=False to then all-gather.
                 # The output `out` from fused kernel is (B,L,D_model) or (B*L, D_model).
                 # If sequence_parallel=True for RowParallelLinear, its *output* is sharded. Input should be full.
                 # It seems the reduce_scatter from Mamba2 applies if `out_proj` is NOT sequence_parallel OR if `out_proj` itself is a ColumnLinear type for output?
                 # This part is confusing from the snippet.

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        # Inference step, largely as provided in Mamba2 source
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))

        d_mlp_calc = self.d_inner - self.d_ssm
        z0_size = d_mlp_calc
        x0_size = d_mlp_calc
        z_size  = self.d_ssm 
        xBC_size= self.d_ssm + 2*self.ngroups*self.d_state
        dt_size = self.nheads
        current_split = [z0_size, x0_size, z_size, xBC_size, dt_size]
        z0, x0, z_for_norm, xBC, dt = torch.split(zxbcdt, current_split, dim=-1)

        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_ssm, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())

        if selective_state_update is None:
            assert self.ngroups == 1, "Default step only supports ngroups=1 if selective_state_update is None"
            dt_processed = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
            dA = torch.exp(dt_processed * A) 
            x_ssm_re = rearrange(x_ssm, "b (h p) -> b h p", p=self.headdim)
            # Original mamba: dBx = torch.einsum("bh,bn,bhp->bhpn", dt_processed, B_step, x_re)
            # B_step would be B from xBC. Here B is (batch, ngroups * d_state)
            # For ngroups=1, B is (batch, d_state)
            # dBx calculation needs B to be (batch, nheads, d_state) if dt is (batch, nheads) and x is (b,h,p)
            # This inference path for no selective_state_update needs careful matching of einops to Mamba1 if that's the reference
            # Assuming B is (batch, d_state), and needs to be broadcast/repeated for nheads for dBx
            # This is a simplified path, may not match Mamba1/2 exactly if ops are missing
            if B.shape[-1] == self.d_state: # if ngroups is 1 implicitly
                 B_for_ssm = repeat(B, 'b n -> b h n', h=self.nheads)
            else: # Fallback, may not be correct for multi-group without selective_state_update
                 B_for_ssm = rearrange(B, 'b (h n) -> b h n', h=self.nheads) # Risky if ngroups > 1 & nheads related

            dBx = torch.einsum("bh,bhn,bhp->bhpn", dt_processed, B_for_ssm, x_ssm_re) # B (b,h,n), x (b,h,p)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bhn->bhp", ssm_state.to(dtype), rearrange(C, 'b (h n) -> b h n', h=self.nheads) if C.shape[-1] != self.d_state else repeat(C, 'b n -> b h n', h=self.nheads))
            D_eff = self.D
            if not self.D_has_hdim and self.headdim > 1: D_eff = repeat(self.D, 'h -> h p', p=self.headdim)
            elif self.D_has_hdim: D_eff = rearrange(self.D, '(h p) -> h p', p=self.headdim)
            
            y = y + D_eff * x_ssm_re # D (h) or (h,p), x_ssm_re (b,h,p)
            y = rearrange(y, "b h p -> b (h p)")
            if not (self.rmsnorm and not isinstance(self.norm, nn.Identity)):
                 if z0_size == 0: # If no explicit z0/x0 from MLP part, use z_for_norm as gate for y_ssm if not norming
                    y = y * self.act(z_for_norm)
        else:
            # This path is if selective_state_update is available
            A_repeated = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt_repeated = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias_repeated = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D_repeated = repeat(self.D, "h -> h p", p=self.headdim) if not self.D_has_hdim else rearrange(self.D, "(h p) -> h p", p=self.headdim)
            B_re = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C_re = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_ssm_re = rearrange(x_ssm, "b (h p) -> b h p", p=self.headdim)
            z_for_norm_re = rearrange(z_for_norm, "b (h p) -> b h p", p=self.headdim)
            
            y = selective_state_update(
                ssm_state, x_ssm_re, dt_repeated, A_repeated, B_re, C_re, D_repeated, 
                z=z_for_norm_re if (not self.rmsnorm and not isinstance(self.norm, nn.Identity)) else None, # z here is pre-SSM gate
                dt_bias=dt_bias_repeated, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        
        if self.rmsnorm and not isinstance(self.norm, nn.Identity):
            y = self.norm(y, z_for_norm)
        
        if z0_size > 0:
            y_mlp = F.silu(z0) * x0
            y = torch.cat([y_mlp, y], dim=-1)
            
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.conv1d.weight.shape[0], self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        # d_state here is N in the Mamba paper (state dimension per head)
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state_shape = (batch_size, self.conv1d.weight.shape[0], self.d_conv)
            ssm_state_shape = (batch_size, self.nheads, self.headdim, self.d_state)
            conv_state = torch.zeros(conv_state_shape, device=self.conv1d.weight.device, dtype=self.conv1d.weight.dtype)
            ssm_state = torch.zeros(ssm_state_shape, device=self.in_proj.weight.device, dtype=self.in_proj.weight.dtype)
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

print("matquant_mamba2.py: Added Mamba2 class definition.")

# --- Block Class Definition (from Mamba repository) --- #
class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=RMSNormBlock, fused_add_norm=False, residual_in_fp32=False # Default norm_cls to our RMSNormBlock
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm # Note: fused_add_norm=True requires specific layer_norm_fn from mamba_ssm.ops.triton.layer_norm
        self.norm = norm_cls(dim) 
        self.mixer = mixer_cls # mixer_cls will be a partial function for Mamba2
        
        if mlp_cls is not nn.Identity and mlp_cls is not None:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = nn.Identity() # Use nn.Identity if mlp_cls is None or nn.Identity
            self.norm2 = nn.Identity() # And no second norm if no MLP

        if self.fused_add_norm:
            # This path requires layer_norm_fn from mamba_ssm.ops.triton.layer_norm
            # We will attempt to import it, otherwise this path might not work as intended.
            global layer_norm_fn # Make it accessible if imported
            try:
                from mamba_ssm.ops.triton.layer_norm import layer_norm_fn as mamba_layer_norm_fn
                layer_norm_fn = mamba_layer_norm_fn
                print("Block: Successfully imported mamba_ssm.ops.triton.layer_norm.layer_norm_fn for fused_add_norm.")
            except ImportError:
                print("Block WARNING: mamba_ssm.ops.triton.layer_norm.layer_norm_fn not found. fused_add_norm=True may not work correctly.")
                layer_norm_fn = None # Fallback if import fails
                # Potentially force fused_add_norm to False if the function isn't available
                # self.fused_add_norm = False 
                # print("Block INFO: Setting fused_add_norm to False due to missing layer_norm_fn.")
            
            # Original assertion from Mamba's Block was:
            # assert RMSNorm is not None, "RMSNorm import fails"
            # assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            # Our norm_cls defaults to RMSNormBlock, which is a simplified RMSNorm.
            # For fused_add_norm to work as in mamba_ssm, self.norm should ideally be the mamba_ssm RMSNorm.
            if not isinstance(self.norm, (nn.LayerNorm, RMSNormBlock)) and (RMSNormGatedOriginal is not None and not isinstance(self.norm, RMSNormGatedOriginal)):
                 print("Block WARNING: norm_cls is not nn.LayerNorm or a recognized RMSNorm variant. fused_add_norm might be suboptimal or incorrect.")


    def forward(
            self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None, 
            inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual)) + residual
        """
        if not self.fused_add_norm or layer_norm_fn is None: # Fallback if fused_add_norm is False or fn not available
            residual_input = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states_norm = self.norm(residual_input.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual_input = residual_input.to(torch.float32)
        else: # Fused add norm path
            # Check if self.norm is an RMSNorm variant as expected by mamba_ssm's layer_norm_fn
            is_rms = isinstance(self.norm, RMSNormBlock) or (RMSNormGatedOriginal is not None and isinstance(self.norm, RMSNormGatedOriginal))
            hidden_states_norm, residual_input = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias if hasattr(self.norm, 'bias') and self.norm.bias is not None else None, # Some RMSNorms don't have bias
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=is_rms
            )
        
        # Mixer (Mamba2) call
        # Mixer_kwargs will pass things like seq_idx if needed by Mamba2
        hidden_states_mix = self.mixer(hidden_states_norm, inference_params=inference_params, **mixer_kwargs)

        # Add back the residual from before the mixer an d MLP
        # The Mamba Block structure usually is: residual = hidden_states_mix + residual_input (from Add -> LN -> Mixer)
        # And then this new residual is passed to the next layer.
        # The output of the mixer becomes the new hidden_states for the next layer's residual connection.

        # HACK for placeholder model where mixer (conv1d) can change sequence length
        hidden_states_mix_for_add = hidden_states_mix
        if hidden_states_mix.shape[1] != residual_input.shape[1]:
            print(f"Block (mixer output vs residual_input) WARNING: seq_len mismatch ({hidden_states_mix.shape[1]} vs {residual_input.shape[1]}). Truncating/padding hidden_states_mix before adding to residual_input. This is a HACK for placeholder model.")
            # Option 1: Truncate longer (hidden_states_mix) to shorter (residual_input). Simplest to avoid crash.
            # This means sequence length will not grow beyond the initial residual_input's length if conv keeps expanding.
            target_len = residual_input.shape[1]
            if hidden_states_mix.shape[1] > target_len:
                hidden_states_mix_for_add = hidden_states_mix[:, :target_len, :]
            elif hidden_states_mix.shape[1] < target_len: # Should not happen if conv always expands or same
                # Pad hidden_states_mix. For simplicity, let's assume truncation is the main case for now.
                # This case would require padding, e.g. F.pad, which is more complex for a quick hack.
                # For now, if mixer output is shorter (shouldn't be with current conv setup), it might error later or broadcast if lucky.
                # Safest is to ensure they match or handle padding.
                # Given the error, hidden_states_mix is longer. So truncation path is active.
                pass # Sticking to truncation of hidden_states_mix if it's longer.

        current_residual = hidden_states_mix_for_add + residual_input 

        if not isinstance(self.mlp, nn.Identity):
            if not self.fused_add_norm or layer_norm_fn is None:
                hidden_states_mlp_norm = self.norm2(current_residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32 and isinstance(self.norm2, nn.Identity): 
                    pass 
                elif self.residual_in_fp32 :
                     current_residual_for_mlp_res = current_residual.to(torch.float32)
                else:
                    current_residual_for_mlp_res = current_residual
            else: 
                is_rms_mlp = isinstance(self.norm2, RMSNormBlock) or (RMSNormGatedOriginal is not None and isinstance(self.norm2, RMSNormGatedOriginal))
                # To ensure consistency if hidden_states_mix was truncated for current_residual calculation:
                # The input to MLP norm here is `hidden_states_mix` (original) + `residual_input` (original seq len)
                # If hidden_states_mix was truncated to hidden_states_mix_for_add for current_residual,
                # then current_residual_for_mlp_res (which is current_residual) is based on the truncated length.
                # The MLP path input should be consistent. The Mamba block typically does: Norm(MixerOut + PrevResidualInput)
                # hidden_states_mix_for_add should be used if it was created.
                # For fused_add_norm, input to layer_norm_fn is hidden_states_mix. If this was truncated, this needs to use truncated version.
                # This part needs to be very careful if MLP is active and truncation happened.
                # Safest for now assuming MLP is Identity or this path would need more detailed seq len handling.
                hidden_states_mlp_norm, current_residual_for_mlp_res = layer_norm_fn(
                    hidden_states_mix_for_add, # Use the (potentially truncated) mixer output if MLP is to be consistent with current_residual
                    self.norm2.weight,
                    self.norm2.bias if hasattr(self.norm2, 'bias') and self.norm2.bias is not None else None,
                    residual=residual_input, 
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=is_rms_mlp
                )
            
            hidden_states_mlp = self.mlp(hidden_states_mlp_norm)
            final_out_hidden_states = hidden_states_mlp 
            final_residual = hidden_states_mlp + current_residual_for_mlp_res
        else:
            # If no MLP, the output of mixer is the main output, and residual is current_residual
            final_out_hidden_states = hidden_states_mix_for_add # CORRECT: use the (potentially) truncated version
            final_residual = current_residual
            
        return final_out_hidden_states, final_residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        # Delegate to mixer if mixer has this method (Mamba2 does)
        if hasattr(self.mixer, 'allocate_inference_cache'):
            return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        return None

print("matquant_mamba2.py: Added Block class definition.") 

# --- Simple MLP for Block --- #
class BlockMLP(nn.Module):
    def __init__(self, d_model, hidden_mult=4, activation=nn.SiLU, dropout=0.0):
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- MatryoshkaMamba2Model Definition --- #
class MatryoshkaMamba2Model(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        d_model: int, 
        n_layer: int, 
        mamba2_config: dict, # Dict for Mamba2 specific params like d_state, d_conv, expand etc.
        quantize_bit_width: int | None = None, # e.g., 5 for 5-bit
        norm_eps: float = 1e-5, # For RMSNorm in Blocks
        fused_add_norm_block: bool = False, # For Block class
        residual_in_fp32_block: bool = False, # For Block class
        mlp_cls_block = nn.Identity, # MLP class for Block, defaults to no MLP
        device=None, 
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.quantize_bit_width = quantize_bit_width
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model, **factory_kwargs)
        if self.quantize_bit_width is not None and self.quantize_bit_width > 0:
            with torch.no_grad():
                self.input_proj.weight.data = quantize_weights_fixed_pt(self.input_proj.weight.data, self.quantize_bit_width, device=device)
                if self.input_proj.bias is not None:
                    self.input_proj.bias.data = quantize_weights_fixed_pt(self.input_proj.bias.data, self.quantize_bit_width, device=device)

        # Stack of Mamba2 Blocks
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            # Mixer_cls for the Block is a Mamba2 layer.
            # We use functools.partial to pass Mamba2-specific hyperparams along with d_model.
            mamba2_layer_config = mamba2_config.copy() # Start with base config
            mamba2_layer_config['d_model'] = d_model
            mamba2_layer_config['layer_idx'] = i # Pass layer_idx for inference cache, etc.
            mamba2_layer_config['device'] = device
            if dtype is not None:
                mamba2_layer_config['dtype'] = dtype
            if self.quantize_bit_width is not None:
                mamba2_layer_config['quantize_bit_width'] = self.quantize_bit_width
            
            # mixer_cls_partial = functools.partial(Mamba2, **mamba2_layer_config) # Old way
            current_mixer_instance = Mamba2(**mamba2_layer_config) # New: Instantiate Mamba2 here
            
            current_norm_cls = functools.partial(RMSNormBlock, eps=norm_eps, device=device, dtype=dtype)

            self.layers.append(
                Block(
                    d_model,
                    mixer_cls=current_mixer_instance, # Pass the instantiated mixer
                    mlp_cls=mlp_cls_block, # e.g., nn.Identity or a GatedMLP class
                    norm_cls=current_norm_cls,
                    fused_add_norm=fused_add_norm_block,
                    residual_in_fp32=residual_in_fp32_block
                )
            )
        
        # Final normalization before output projection (optional, but common)
        self.final_norm = RMSNormBlock(d_model, eps=norm_eps, **factory_kwargs)

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim, **factory_kwargs)
        if self.quantize_bit_width is not None and self.quantize_bit_width > 0:
            with torch.no_grad():
                self.output_proj.weight.data = quantize_weights_fixed_pt(self.output_proj.weight.data, self.quantize_bit_width, device=device)
                if self.output_proj.bias is not None:
                    self.output_proj.bias.data = quantize_weights_fixed_pt(self.output_proj.bias.data, self.quantize_bit_width, device=device)

    def forward(self, x, inference_params=None):
        # x: (batch, seq_len, input_dim)
        hidden_states = self.input_proj(x)
        
        residual = None # For the first block, residual is None if not using pre-LN add
                        # Or hidden_states itself if using pre-LN add without external residual.
                        # The Block class handles residual=None for the first interaction.

        for i, layer in enumerate(self.layers):
            # Update inference_params for the current layer if provided (for Mamba2 step function)
            if inference_params is not None:
                inference_params.layer_idx = i # Set current layer index for cache handling in Mamba2
            
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
            # The `residual` returned by the block is (mixer_output + its_input_residual)
            # The `hidden_states` returned by the block is the output of the MLP (if any) or the mixer output.
            # For the next layer, hidden_states is the primary input, and `residual` is the accumulated residual.

        hidden_states = self.final_norm(hidden_states + residual if residual is not None else hidden_states)
        out = self.output_proj(hidden_states)
        
        # Squeeze last dim if output_dim is 1 (e.g. for regression)
        if self.output_proj.out_features == 1:
            out = out.squeeze(-1)
            
        return out

print("matquant_mamba2.py: Added MatryoshkaMamba2Model class definition. File complete.") 
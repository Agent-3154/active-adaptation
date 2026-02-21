import torch
import torch.nn as nn
import einops
import contextlib
from typing import Tuple
from torch.utils._contextlib import _DecoratorContextManager

_RECURRENT_MODE = False

class set_recurrent_mode(_DecoratorContextManager):
    def __init__(self, mode: bool = True):
        super().__init__()
        self.mode = mode
        self.prev = _RECURRENT_MODE
    
    def __enter__(self):
        global _RECURRENT_MODE
        _RECURRENT_MODE = self.mode
    
    def __exit__(self, exc_type, exc_value, traceback):
        global _RECURRENT_MODE
        _RECURRENT_MODE = self.prev


def recurrent_mode():
    return _RECURRENT_MODE


class LSTM(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        burn_in: int = 0, 
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.out = nn.Sequential(nn.LazyLinear(hidden_size), nn.Mish())
        self.burn_in = burn_in

    def forward(
        self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor
    ):  
        if recurrent_mode():
            N, T = x.shape[:2]
            hx = hx[:, 0]
            cx = cx[:, 0]
            output = []
            reset = 1. - is_init.float().reshape(N, T, 1)
            for i, x_t, reset_t in zip(range(T), x.unbind(1), reset.unbind(1)):
                hx, cx = self.lstm(x_t, (hx * reset_t, cx * reset_t))
                output.append(hx)
            output = torch.stack(output, dim=1)
            output = self.out(output)
            return (
                output,
                einops.repeat(hx, "b h -> b t h", t=T),
                einops.repeat(cx, "b h -> b t h", t=T)
            )
        else:
            N = x.shape[0]
            reset = 1. - is_init.float().reshape(N, 1)
            hx, cx = self.lstm(x, (hx * reset, cx * reset))
            output = self.out(hx)
            return output, hx, cx


class GRUCore(nn.Module):
    """
    PPO-friendly GRU core using GRUCell and a Python loop, with:
      - is_init masking (episode resets),
      - optional detach at segment start (truncated BPTT),
      - LayerNorm on hidden states,
      - residual connection from inputs (with projection when D != H),
      - orthogonal recurrent init and optional gate bias trick.

    Forward supports both step and sequence modes:
      - Step:     x [B, D],     is_init [B] (bool)    -> y [B, H], h_last [B, H]
      - Sequence: x [B, T, D],  is_init [B, T] (bool) -> Y [B, T, H], h_last [B, H]

    Conventions:
      - is_init[b, t] == True means: the CURRENT observation at time t is the FIRST after an env reset.
      - mask := (~is_init).float() -> 1 keeps state; 0 resets to zero BEFORE processing x_t.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_layernorm_hidden: bool = True,
        use_layernorm_input: bool = False,   # optional; helpful if inputs are non-stationary
        residual: bool = True,
        residual_gate_init: float = 0.0,     # gate initializes near 0; learns if residual helps
        detach_init_state: bool = True,      # good default for PPO segments
        init_update_bias_neg: bool = True,   # set update gate bias negative for faster adaptation
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.detach_init_state = detach_init_state

        # Core GRU cell (step-wise to allow masking)
        self.cell = nn.GRUCell(input_size, hidden_size)

        # Normalization
        self.ln_in = nn.LayerNorm(input_size) if use_layernorm_input else nn.Identity()
        self.ln_h  = nn.LayerNorm(hidden_size) if use_layernorm_hidden else nn.Identity()

        # Residual connection from inputs to outputs (post-activation residual)
        self.use_residual = residual
        if residual:
            # Project input to hidden size if needed
            self.res_proj = nn.Identity() if input_size == hidden_size else nn.Linear(input_size, hidden_size, bias=False)
            # Per-feature learnable gate in [-] -> (0,1) by sigmoid
            self.res_gate = nn.Parameter(torch.full((hidden_size,), residual_gate_init))
        else:
            self.register_buffer("res_gate", torch.zeros(hidden_size), persistent=False)
            self.res_proj = nn.Identity()

        # --- Initialization ---
        # Recurrent weights orthogonal; input weights Xavier; biases zero
        for name, p in self.cell.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        # Optional trick: make update gate initially more "open" by negative bias
        # PyTorch GRU gate order is [reset (r), update (z), new (n)] per hidden chunk.
        if init_update_bias_neg:
            H = hidden_size
            with torch.no_grad():
                if getattr(self.cell, "bias_ih", None) is not None:
                    self.cell.bias_ih.data[H:2 * H].fill_(-1.0)
                if getattr(self.cell, "bias_hh", None) is not None:
                    self.cell.bias_hh.data[H:2 * H].fill_(-1.0)

        # If residual projection exists, init it nicely
        if residual and isinstance(self.res_proj, nn.Linear):
            nn.init.xavier_uniform_(self.res_proj.weight)

    # ---- Forward (step or sequence) ----
    def forward(
        self,
        x: torch.Tensor,      # [B, D] or [B, T, D]
        hx: torch.Tensor,     # [B, H] or [B, T, H], must be provided
        is_init: torch.Tensor, # [B] or [B, T] (bool), must be provided
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          - If x.dim()==2: y [B, H], h_last [B, H]
          - If x.dim()==3: Y [B, T, H], h_last [B, H]
        """
        dtype = x.dtype
        if x.dim() == 2:
            B, D = x.shape
            # Preprocess inputs
            x_t = self.ln_in(x)
            # Reset where current step is first after env reset
            mask = (~is_init).to(dtype).reshape(B, 1)  # [B,1]
            # Recurrent update
            hx = self.cell(x_t, hx * mask)
            # Post activ + residual
            y = self.ln_h(hx)
            if self.use_residual:
                res = self.res_proj(x)                   # use raw x for residual (or use normalized x_t; both are fine)
                gate = torch.sigmoid(self.res_gate)      # [H]
                y = y + res * gate
            return y, hx
        elif x.dim() == 3:
            B, T, D = x.shape
            hx = hx[:, 0]
            hs = []
            mask = (~is_init).to(dtype).reshape(B, T, 1)  # [B,T,1]
            for t in range(T):
                # Preprocess x_t
                x_t = self.ln_in(x[:, t])
                # Recurrent update
                hx = self.cell(x_t, hx * mask[:, t])
                hs.append(hx)
            H = torch.stack(hs, dim=1)  # [B, T, H]
            Y = self.ln_h(H)
            # Post activ + residual
            if self.use_residual:
                res = self.res_proj(x)
                gate = torch.sigmoid(self.res_gate)          # [H]
                Y = Y + res * gate
            return Y, H
        else:
            raise ValueError(f"Expected x rank 2 or 3, got {x.shape}")
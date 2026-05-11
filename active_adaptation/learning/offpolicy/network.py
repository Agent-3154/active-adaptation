import torch
import torch.nn as nn


class ConditionalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int = 0,
        norm: str | None = None, # None, "rms", "layer"
        activation: str | type[nn.Module] = nn.SiLU
    ):
        super().__init__()
        if norm is None:
            self.norm = nn.Identity()
        elif norm.lower() == "rms":
            self.norm = nn.RMSNorm(hidden_dim)
        elif norm.lower() == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Invalid norm: {norm}")

        ActClass = activation if isinstance(activation, type) else getattr(nn, activation)
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            ActClass(),
            nn.Linear(hidden_dim, hidden_dim),
            ActClass()
        )

        if condition_dim > 0:
            self.cond_proj = nn.Linear(condition_dim, 2 * hidden_dim)
        else:
            self.cond_proj = None
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if self.cond_proj is not None:
            cond = self.cond_proj(cond)
            scale, shift = cond.chunk(2, dim=-1)
            x = x * (1.0 + scale) + shift
        x = self.layers(x)
        return x + residual


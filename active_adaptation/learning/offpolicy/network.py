import torch
import torch.nn as nn


class ConditionalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int = 0,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
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


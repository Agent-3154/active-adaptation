import math
import torch
import torch.nn as nn


class PositionEmbedding1D(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # Create sinusoidal position embeddings
        position = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        
        pos_emb = torch.zeros(1, self.seq_len, self.embed_dim)
        pos_emb[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position * div_term)
        
        # Make it learnable by wrapping in nn.Parameter
        self.pos_emb = nn.Parameter(pos_emb)
    
    def forward(self):
        return self.pos_emb


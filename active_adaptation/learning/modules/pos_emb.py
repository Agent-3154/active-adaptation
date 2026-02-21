import math
import torch
import torch.nn as nn
from typing import Tuple
from jaxtyping import Float


class PositionEmbedding1D(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Create sinusoidal position embeddings
        position = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )

        pos_emb = torch.zeros(1, self.seq_len, self.embed_dim)
        pos_emb[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position * div_term)

        # Make it learnable by wrapping in nn.Parameter
        self.pos_emb = nn.Parameter(pos_emb)

    def forward(self):
        return self.pos_emb


class PositionEncodingND(nn.Module):
    """N-dimensional positional encoding using coordinate meshgrid.

    Adds positional encodings to input tensors by concatenating coordinate indices
    along each spatial dimension. The positional encoding is created using a meshgrid
    of coordinate indices, which are then concatenated to the input tensor.

    This is useful for neural networks that need to be aware of spatial positions,
    such as in image processing or 3D data processing tasks.

    Args:
        shape: Tuple of integers specifying the spatial dimensions.
            For example:
            - (H, W) for 2D data (height, width)
            - (D, H, W) for 3D data (depth, height, width)
            - (T, H, W) for video data (time, height, width)
        center: Whether to center the position encoding around the origin.
            If True, the position encoding will be centered around the origin.
            If False, the position encoding will be at the origin.

    Example:
        >>> # For 2D image data with shape (batch, channels, height, width)
        >>> pos_enc = PositionEncodingND(shape=(64, 64))  # 64x64 images
        >>> x = torch.randn(10, 3, 64, 64)  # batch=10, channels=3
        >>> x_with_pos = pos_enc(x)  # Shape: (10, 3+2, 64, 64) - adds 2 dims for (H, W) coords
        >>>
        >>> # For 3D data with shape (batch, channels, depth, height, width)
        >>> pos_enc_3d = PositionEncodingND(shape=(32, 64, 64))
        >>> x_3d = torch.randn(5, 1, 32, 64, 64)
        >>> x_3d_with_pos = pos_enc_3d(x_3d)  # Adds 3 dims for (D, H, W) coords
    """

    def __init__(self, shape: Tuple[int, ...], center: bool = True):
        super().__init__()
        self.shape = shape
        self.num_dims = len(shape)
        # Create meshgrid of coordinate indices and stack them along a new dimension
        meshgrid_tensors = torch.meshgrid(
            *[torch.arange(s) for s in shape], indexing="ij"
        )
        if center:
            meshgrid_tensors = [
                t - (s - 1) / 2 for t, s in zip(meshgrid_tensors, shape)
            ]
        pos_encoding = torch.stack(meshgrid_tensors, dim=0)  # Shape: (num_dims, *shape)
        self.register_buffer(
            "pos_encoding", pos_encoding.unsqueeze(0)
        )  # Add batch dim: (1, num_dims, *shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to input tensor.

        Expands the positional encoding to match the input's batch dimensions and
        concatenates it along the channel/feature dimension.

        Args:
            x: Input tensor of shape (..., channels, *spatial_dims) where spatial_dims matches
                the shape provided during initialization. The input must have at least
                `num_dims + 1` trailing dimensions (1 for channels, `num_dims` for spatial dims).

        Returns:
            Tensor with positional encodings concatenated. The output shape is
            (..., channels + num_dims, *spatial_dims), where `num_dims` is the
            number of spatial dimensions (length of `shape`).

        Raises:
            RuntimeError: If the trailing dimensions of `x` don't match `self.shape`.
        """
        # Expand pos_encoding from (1, num_dims, *shape) to (batch, num_dims, *shape)
        batch_shape = x.shape[
            : -(self.num_dims + 1)
        ]  # All dims except (channels, *spatial_dims)
        pos_encoding = self.pos_encoding.expand(
            *batch_shape, self.num_dims, *self.shape
        )
        return torch.cat([x, pos_encoding], dim=-(self.num_dims + 1))

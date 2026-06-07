import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase


class CatTensors(TensorDictModuleBase):
    def __init__(self, in_keys, out_key, del_keys=False, sort=True):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = [out_key]

        self.del_keys = del_keys
        self.sort = sort
        if self.sort:
            self.in_keys = sorted(self.in_keys)

    def forward(self, tensordict: TensorDictBase):
        out = torch.cat([tensordict.get(k) for k in self.in_keys], dim=-1)
        tensordict.set(self.out_keys[0], out)
        if self.del_keys:
            tensordict.exclude(*self.in_keys, inplace=True)
        return tensordict


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable layer normalization.

    A feedforward neural network that supports pre-norm or post-norm layer normalization,
    or no normalization. The network is constructed as a sequence of linear layers,
    optional layer normalization, and activation functions.

    Args:
        num_units: List of integers specifying the number of units in each layer.
            The first element is the input dimension, and the last is the output dimension.
            For example, [128, 64, 32] creates a network with input size 128,
            hidden layer size 64, and output size 32.
        activation: PyTorch activation module class (not instance). Defaults to nn.Mish.
            Examples: nn.ReLU, nn.GELU, nn.Mish.
        layer_norm: Position of layer normalization relative to activation.
            - "pre": Apply layer normalization before activation (pre-norm).
            - "post": Apply layer normalization after activation (post-norm).
            - None: No layer normalization.
            Defaults to "pre".
        first_non_muon: If True, the first linear layer's weight is marked as non-Muon.
            Defaults to False.

    Example:
        >>> mlp = MLP(num_units=[128, 64, 32], activation=nn.ReLU, layer_norm="pre")
        >>> x = torch.randn(10, 128)
        >>> output = mlp(x)  # Shape: (10, 32)
    """

    def __init__(
        self,
        num_units: List[int],
        activation: nn.Module = nn.Mish,
        layer_norm: Literal["pre", "post", None] = "pre",
        first_non_muon: bool = False,
    ):
        super().__init__()
        self.num_units = num_units
        self.activation = activation
        self.layer_norm = layer_norm
        layers = []
        for i in range(len(num_units) - 1):
            layer = nn.Linear(num_units[i], num_units[i + 1])
            if first_non_muon and i == 0:
                layer.weight._non_muon = True
            layers.append(layer)
            if layer_norm == "pre":
                layers.append(nn.LayerNorm(num_units[i + 1]))
            layers.append(activation())
            if layer_norm == "post":
                layers.append(nn.LayerNorm(num_units[i + 1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (..., input_dim) where input_dim is num_units[0].

        Returns:
            Output tensor of shape (..., output_dim) where output_dim is num_units[-1].
        """
        return self.layers(x)

    def orth(self, gain: float = 1.0) -> "MLP":
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain)
                nn.init.zeros_(module.bias)
        return self


class ResidualMLP(nn.Module):
    """Residual Multi-Layer Perceptron with skip connections.

    A feedforward neural network with residual (skip) connections between layers.
    Each layer applies: output = layer(x) + skip_connection(x), where the skip
    connection is either an identity mapping (if input/output dimensions match)
    or a linear projection (if dimensions differ).

    Each residual block consists of: Linear -> LayerNorm -> Activation.
    The skip connection is applied after the activation.

    Args:
        num_units: List of integers specifying the number of units in each layer.
            The first element is the input dimension, and the last is the output dimension.
            For example, [128, 64, 32] creates a network with input size 128,
            hidden layer size 64, and output size 32.
        activation: PyTorch activation module class (not instance). Defaults to nn.Mish.
            Examples: nn.ReLU, nn.GELU, nn.Mish.

    Example:
        >>> res_mlp = ResidualMLP(num_units=[128, 64, 32], activation=nn.ReLU)
        >>> x = torch.randn(10, 128)
        >>> output = res_mlp(x)  # Shape: (10, 32)
    """

    def __init__(
        self,
        num_units: List[int],
        activation: nn.Module = nn.Mish,
        first_non_muon: bool = False,
    ):
        super().__init__()
        self.num_units = num_units
        self.activation = activation
        self.first_non_muon = first_non_muon

        layers = []
        skip_layers = []

        for i in range(len(num_units) - 1):
            in_features = num_units[i]
            out_features = num_units[i + 1]
            if in_features != out_features:
                skip_layer = nn.Linear(in_features, out_features)
                skip_layer.weight._non_muon = True
            else:
                skip_layer = nn.Identity()
            linear = nn.Linear(in_features, out_features)
            if self.first_non_muon and i == 0:
                linear.weight._non_muon = True
            layer = nn.Sequential(
                linear,
                nn.LayerNorm(out_features),
                activation(),
            )
            layers.append(layer)
            skip_layers.append(skip_layer)
        self.layers = nn.ModuleList(layers)
        self.skip_layers = nn.ModuleList(skip_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual MLP.

        Args:
            x: Input tensor of shape (..., input_dim) where input_dim is num_units[0].

        Returns:
            Output tensor of shape (..., output_dim) where output_dim is num_units[-1].
        """
        for layer, skip_layer in zip(self.layers, self.skip_layers):
            x = layer(x) + skip_layer(x)
        return x


class SimbaMLP(nn.Module):
    """The architecture described in https://arxiv.org/pdf/2410.09754."""

    def __init__(self, num_units: int, num_blocks: int, activation=nn.SiLU):
        super().__init__()
        self.num_units = num_units
        self.num_blocks = num_blocks
        blocks = []

        for i in range(num_blocks):
            block = nn.Sequential(
                nn.LayerNorm(num_units),
                nn.Linear(num_units, num_units),
                activation(),
                nn.Linear(num_units, num_units),
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class ConditionalBlock(nn.Module):
    """Residual FiLM block with optional RMS / layer norm."""

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 1,
        condition_dim: int = 0,
        norm: str | None = None,
        activation: str | type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.expansion = int(expansion)
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
            nn.Linear(hidden_dim, hidden_dim * expansion),
            ActClass(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
            ActClass(),
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


class DtypeConversion(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        return x.to(self.dtype)


class FlattenBatch(nn.Module):
    def __init__(self, module, data_dim: int = 1):
        super().__init__()
        self.module = module
        self.data_dim = data_dim

    def forward(self, *args: torch.Tensor):
        batch_shape = args[0].shape[: -self.data_dim]
        args_flattened = (arg.flatten(0, len(batch_shape) - 1) for arg in args)
        output_flattened = self.module(*args_flattened)
        if isinstance(output_flattened, tuple):
            output = tuple(arg.unflatten(0, batch_shape) for arg in output_flattened)
        else:
            output = output_flattened.unflatten(0, batch_shape)
        return output


class SymmetryWrapper(TensorDictModuleBase):
    """
    Wrap a module to apply symmetry transformations to the input and output.
    The input is stacked with its mirrored version, and the output is averaged.

    Args:
        module: The module to wrap.
        input_transform: The input transform to apply.
        output_transform: The output transform to apply.
    """

    def __init__(
        self,
        module: TensorDictModuleBase,
        input_transform: TensorDictModuleBase,
        output_transform: TensorDictModuleBase,
    ):
        super().__init__()
        self.module = module
        self.in_keys = self.module.in_keys
        self.out_keys = self.module.out_keys
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, td: TensorDictBase):
        input = td.select(*self.in_keys)
        input_mirrored = input.empty()
        self.input_transform(input, tensordict_out=input_mirrored)
        input_mirrored = torch.stack([input, input_mirrored], dim=0)
        output_mirrored = self.module(input_mirrored).select(*self.out_keys)
        output = (output_mirrored[0] + self.output_transform(output_mirrored[1])) * 0.5
        td.update(output)
        return td

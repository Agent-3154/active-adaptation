import torch
import torch.nn as nn
from typing import List, Literal
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase


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
    ):
        super().__init__()
        self.num_units = num_units
        self.activation = activation
        self.layer_norm = layer_norm
        layers = []
        for i in range(len(num_units) - 1):
            layers.append(nn.Linear(num_units[i], num_units[i + 1]))
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
    ):
        super().__init__()
        self.num_units = num_units
        self.activation = activation
        
        layers = []
        skip_layers = []

        for i in range(len(num_units) - 1):
            in_features = num_units[i]
            out_features = num_units[i + 1]
            if in_features != out_features:
                skip_layer = nn.Linear(in_features, out_features)
            else:
                skip_layer = nn.Identity()
            layer = nn.Sequential(
                nn.Linear(in_features, out_features),
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


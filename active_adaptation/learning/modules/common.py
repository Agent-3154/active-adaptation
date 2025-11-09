import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase


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


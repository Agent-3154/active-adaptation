"""Route one PPO distribution to the environment's named control inputs."""
import torch
from tensordict.nn import TensorDictModuleBase


class ActionRouting(TensorDictModuleBase):
    def __init__(self, action_spec):
        super().__init__()
        self.keys = list(action_spec.keys())
        if "action" in self.keys and len(self.keys) != 1:
            raise ValueError("Multi-input PPO reserves 'action' for the flat sampled action")
        self.sizes = [action_spec[key].shape[-1] for key in self.keys]
        self.action_dim = sum(self.sizes)
        self.in_keys = ["action"]
        self.out_keys = self.keys

    def forward(self, tensordict):
        for key, value in zip(self.keys, torch.split(tensordict["action"], self.sizes, dim=-1)):
            tensordict.set(key, value)
        return tensordict

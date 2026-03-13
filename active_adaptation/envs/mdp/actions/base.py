import abc
import torch

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent


class Action(MDPComponent, RegistryMixin):
    action_dim: int
    action_buf: torch.Tensor

    def __init__(self, env):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]

    @abc.abstractmethod
    def process_action(self, action: torch.Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_action(self, substep: int):
        raise NotImplementedError

__all__ = [
    "Action",
]

from __future__ import annotations

import torch

from typing import List
from typing_extensions import override

from active_adaptation.utils.symmetry import SymmetryTransform

from .base import Action


class ConcatenatedAction(Action):
    """Concatenate multiple action managers into a single action space."""

    def __init__(self, env, actions: List):
        super().__init__(env)
        self.action_managers: List[Action] = []

        for spec in actions:
            cls = Action.registry[spec.pop("_target_")]
            self.action_managers.append(cls(self.env, **spec))
        self.action_dims = [
            action_manager.action_dim for action_manager in self.action_managers
        ]

    @property
    def action_dim(self):
        return sum(self.action_dims)

    @property
    def action_buf(self):
        return torch.cat(
            [action_manager.action_buf for action_manager in self.action_managers], dim=-1
        )

    @override
    def process_action(self, action: torch.Tensor):
        actions = torch.split(action, self.action_dims, dim=-1)
        for action_manager, action_chunk in zip(self.action_managers, actions):
            action_manager.process_action(action_chunk)

    @override
    def apply_action(self, substep: int):
        [action_manager.apply_action(substep) for action_manager in self.action_managers]

    @override
    def symmetry_transform(self):
        return SymmetryTransform.cat(
            [action_manager.symmetry_transform() for action_manager in self.action_managers]
        )


__all__ = ["ConcatenatedAction"]

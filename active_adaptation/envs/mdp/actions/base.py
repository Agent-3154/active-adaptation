from __future__ import annotations

import abc
import torch

from typing import TYPE_CHECKING

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase
    from active_adaptation.utils.symmetry import SymmetryTransform


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

    def diagnostics(self) -> dict:
        return {}
    
    def symmetry_transform(self) -> SymmetryTransform:
        """Return the mirror transform for this action term's output slice.

        The transform describes how an action vector should change under the
        task's left/right symmetry. It must have the same width and ordering as
        the tensor accepted by :meth:`process_action` for this component. For a
        typical joint action, implement this by permuting symmetric joints and
        flipping signs for coordinates whose positive direction changes under
        reflection.

        Composite action managers concatenate component actions in config
        order, so each component returns only its local transform; the enclosing
        manager concatenates those local transforms into the full policy-action
        transform. Components with no well-defined symmetry should override this
        method and raise ``NotImplementedError`` explicitly.
        """
        return NotImplementedError


class ActionV2(MDPComponent, RegistryMixin):
    """Environment-deferred action term.

    Like :class:`Action`, subclasses implement :meth:`process_action` and
    :meth:`apply_action`.

    Unlike :class:`Action`, instances are constructed **without** an
    environment. Environment-bound state (``env``, ``asset``) is created in
    :meth:`_initialize`, which the environment calls once at startup.
    """

    action_dim: int
    action_buf: torch.Tensor

    def __init__(self) -> None:
        self._initialized = False

    def _initialize(self, env: "_EnvBase") -> None:
        """Bind to ``env``. Called once at startup."""
        self.env = env
        self.asset = self.env.scene.articulations["robot"]
        self._initialized = True

    @property
    def initialized(self) -> bool:
        """``True`` after :meth:`_initialize` has been called."""
        return self._initialized

    @abc.abstractmethod
    def process_action(self, action: torch.Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_action(self, substep: int):
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}

    def symmetry_transform(self) -> SymmetryTransform:
        """Return the mirror transform for this action term's output slice.

        The transform describes how an action vector should change under the
        task's left/right symmetry. It must have the same width and ordering as
        the tensor accepted by :meth:`process_action` for this component. For a
        typical joint action, implement this by permuting symmetric joints and
        flipping signs for coordinates whose positive direction changes under
        reflection.

        Composite action managers concatenate component actions in config
        order, so each component returns only its local transform; the enclosing
        manager concatenates those local transforms into the full policy-action
        transform. Components with no well-defined symmetry should override this
        method and raise ``NotImplementedError`` explicitly.
        """
        return NotImplementedError


__all__ = ["Action", "ActionV2"]

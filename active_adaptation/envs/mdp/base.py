from __future__ import annotations

import torch
import abc

from typing import Tuple, TYPE_CHECKING, Generic, TypeVar
from active_adaptation.registry import RegistryMixin
from active_adaptation.utils.math import quat_mul, sample_quat_yaw
from active_adaptation.utils.symmetry import SymmetryTransform


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from mjlab.entity import Entity
    from active_adaptation.envs.env_base import _EnvBase


def is_method_implemented(obj, base_class, method_name: str):
    """Check if a method is actually implemented (not just the base class pass).
    
    This function checks if a subclass has overridden a method from the base class.
    It compares the underlying function objects to determine if the method was
    actually overridden, even if the override just contains `pass`.
    
    Args:
        obj: Instance to check
        base_class: Base class that defines the default method
        method_name: Name of the method to check (e.g., 'post_step', 'update', etc.)
        
    Returns:
        True if the method is overridden in the subclass, False otherwise
    """
    # Get the method from the instance's class (not the instance itself)
    obj_method = getattr(type(obj), method_name, None)
    base_method = getattr(base_class, method_name, None)
    
    if obj_method is None or base_method is None:
        return False
    
    # Get the underlying function objects
    # In Python 3, accessing from class gives a function directly
    # In some cases it might be a method, so we handle both
    obj_func = getattr(obj_method, '__func__', obj_method)
    base_func = getattr(base_method, '__func__', base_method)
    
    # Compare the underlying function objects
    # If they're the same, it means the method wasn't overridden
    return obj_func is not base_func


class MDPComponent:
    """
    Base class for all MDP components (Command, Observation, Reward, Termination, Randomization).
    
    Provides common initialization, properties, and lifecycle methods that are shared
    across all component types.
    
    Note: This class does not include registry functionality. Use multiple inheritance
    with `RegistryMixin` if registry functionality is needed.
    """
    
    def __init__(self, env: _EnvBase):
        """Initialize the MDP component with a reference to the environment.
        
        Args:
            env: The environment instance this component belongs to.
        """
        self.env: _EnvBase = env
    
    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self.env.num_envs
    
    @property
    def device(self) -> torch.device:
        """Device on which tensors are stored."""
        return self.env.device
    
    def reset(self, env_ids: torch.Tensor) -> None:
        """Called after episode termination.
        
        Args:
            env_ids: Indices of environments that were reset.
        """
        pass
    
    def update(self) -> None:
        """Called after all physics substeps are completed."""
        pass
    
    def pre_step(self, substep: int) -> None:
        """Called before each physics substep.
        
        Args:
            substep: The current substep index.
        """
        pass
    
    def post_step(self, substep: int) -> None:
        """Called after each physics substep.
        
        Args:
            substep: The current substep index.
        """
        pass
    
    def startup(self) -> None:
        """Called once upon initialization of the environment."""
        pass
    
    def debug_draw(self) -> None:
        """Called at each step **after** simulation, if GUI is enabled."""
        pass


class Command(MDPComponent, RegistryMixin):
    def __init__(self, env: _EnvBase, teleop: bool=False) -> None:
        super().__init__(env)
        self.asset: "Articulation" | "Entity" = env.scene["robot"]
        self.init_root_state = self.asset.data.default_root_state.clone()
        self.init_joint_pos = self.asset.data.default_joint_pos.clone()
        self.init_joint_vel = self.asset.data.default_joint_vel.clone()
        self.teleop = teleop
        
        if self.env.backend == "isaac":
            if self.env.terrain_type == "generator":
                self._origins = self.env.scene.terrain.terrain_origins.reshape(-1, 3).clone()
            else:
                self._origins = self.env.scene.env_origins.reshape(-1, 3).clone()
        elif self.env.backend == "mujoco":
            self._origins = torch.zeros(self.num_envs, 3, device=self.device)

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Called before `reset` to sample initial state for the next episodes.
        This can be used for implementing curriculum learning.
        """
        init_root_state = self.init_root_state[env_ids]
        if self.env.terrain_type == "plane":
            origins = self.env.scene.env_origins[env_ids]
        else:
            idx = torch.randint(0, len(self._origins), (len(env_ids),), device=self.device)
            origins = self._origins[idx]
        init_root_state[:, :3] += origins
        init_root_state[:, 3:7] = quat_mul(
            init_root_state[:, 3:7],
            sample_quat_yaw(len(env_ids), device=self.device)
        )
        return init_root_state


CT = TypeVar('CT', bound=Command)


class Observation(Generic[CT], MDPComponent, RegistryMixin):
    """
    Base class for all observations.
    """

    def __init__(self, env: _EnvBase):
        super().__init__(env)
        self.command_manager: CT = env.command_manager

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError
    
    def __call__(self) ->  Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.compute()
        return tensor

    def symmetry_transform(self) -> SymmetryTransform:
        """Called to apply symmetry transformations to the observation"""
        pass


class ActionManager(MDPComponent, RegistryMixin):

    action_dim: int
    action_buf: torch.Tensor

    def __init__(self, env: _EnvBase):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

    @abc.abstractmethod
    def process_action(self, action: torch.Tensor):
        """Process the action. Called once per environment step."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_action(self, substep: int):
        """Apply the action. Called at every simulation step."""
        raise NotImplementedError


class Reward(Generic[CT], MDPComponent, RegistryMixin):
    def __init__(
        self,
        env: _EnvBase,
        weight: float,
        enabled: bool = True,
    ):
        super().__init__(env)
        self.command_manager: CT = env.command_manager
        self.weight = weight
        self.enabled = enabled

    def __call__(self) -> torch.Tensor:
        result = self.compute()
        if isinstance(result, torch.Tensor):
            rew, count = result, result.numel()
        elif isinstance(result, tuple):
            rew, is_active = result
            rew = rew * is_active.float()
            count = is_active.sum()
        return self.weight * rew, count 

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError


class Termination(Generic[CT], MDPComponent, RegistryMixin):
    def __init__(
        self,
        env: _EnvBase,
        is_timeout: bool = False,
        enabled: bool = True,
    ):
        super().__init__(env)
        self.command_manager: CT = env.command_manager
        # `is_timeout=True` means the condition contributes to `truncated`,
        # otherwise it contributes to `terminated`.
        self.is_timeout = is_timeout
        self.enabled = enabled
    
    @abc.abstractmethod
    def compute(self, termination: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Randomization(MDPComponent, RegistryMixin):
    def __init__(self, env: _EnvBase):
        super().__init__(env)

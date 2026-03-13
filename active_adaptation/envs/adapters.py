"""Common adapter protocols shared by all environment backends."""

from typing import Dict, Protocol, TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveScene
    from mjlab.entity import Entity
    from mjlab.scene import Scene


class SimAdapter(Protocol):
    def get_physics_dt(self) -> float: ...

    def has_gui(self) -> bool: ...

    def step(self, render: bool = False) -> None: ...

    def render(self) -> None: ...

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None: ...


class SceneAdapter(Protocol):
    _scene: Union["InteractiveScene", "Scene"]

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    def reset(self, env_ids: torch.Tensor) -> None:
        self._scene.reset(env_ids)

    def update(self, dt: float) -> None:
        self._scene.update(dt)

    def write_data_to_sim(self) -> None:
        self._scene.write_data_to_sim()

    def zero_external_wrenches(self) -> None:
        raise NotImplementedError(
            f"Zero external wrenches is not implemented for {self.__class__.__name__}."
        )

    @property
    def articulations(self) -> Dict[str, Union["Articulation", "Entity"]]: ...

    @property
    def sensors(self) -> dict:
        return self._scene.sensors

    @property
    def env_origins(self) -> torch.Tensor:
        return self._scene.env_origins

    def get_spawn_origins(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.env_origins[env_ids]


__all__ = [
    "SimAdapter",
    "SceneAdapter",
]

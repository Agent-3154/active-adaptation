"""Common adapter protocols shared by all environment backends."""
from __future__ import annotations

from typing import Dict, Protocol, TYPE_CHECKING, Union, Any

import torch
import numpy as np

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveScene
    from mjlab.entity import Entity
    from mjlab.scene import Scene


class CameraFrustumHandle:
    """Backend-agnostic camera frustum for debug visualization.

    Wraps a Viser ``CameraFrustumHandle`` (or compatible object) and accepts
    torch / numpy assignments for pose and image.
    """

    def __init__(self, handle: Any):
        self._handle = handle

    @staticmethod
    def _as_numpy(value: torch.Tensor | np.ndarray, shape: tuple[int, ...] | None = None):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr

    @property
    def position(self) -> np.ndarray:
        return self._handle.position

    @position.setter
    def position(self, value: torch.Tensor | np.ndarray) -> None:
        self._handle.position = self._as_numpy(value, (3,)).astype(np.float32)

    @property
    def wxyz(self) -> np.ndarray:
        return self._handle.wxyz

    @wxyz.setter
    def wxyz(self, value: torch.Tensor | np.ndarray) -> None:
        self._handle.wxyz = self._as_numpy(value, (4,)).astype(np.float32)

    @property
    def image(self):
        return self._handle.image

    @image.setter
    def image(self, value: torch.Tensor | np.ndarray) -> None:
        self._handle.image = self._as_numpy(value)

    def __getattr__(self, name: str):
        return getattr(self._handle, name)


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

    def get(self, name, default=None):
        raise NotImplementedError

    @property
    def articulations(self) -> Dict[str, Union["Articulation", "Entity"]]: ...

    @property
    def entities(self) -> Dict[str, Union["Articulation", "Entity"]]: ...

    @property
    def sensors(self) -> dict:
        return self._scene.sensors

    @property
    def env_origins(self) -> torch.Tensor:
        return self._scene.env_origins

    @property
    def ground_mesh(self):
        """Warp ground mesh used for ray-based height queries.

        Backends that support ground raycasting must provide a warp-compatible
        mesh here. Backends without a concept of a shared ground can raise
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def get_spawn_origins(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.env_origins[env_ids]

    def create_sphere_marker(
        self, prim_path: str, color: tuple[float, float, float], radius: float
    ): ...

    def create_arrow_marker(
        self,
        prim_path: str,
        color: tuple[float, float, float],
        scale: tuple[float, float, float],
    ): ...

    def create_camera_frustum(
        self, name: str, *, fov_y: float, aspect: float, scale: float = 0.15
    ) -> CameraFrustumHandle: ...

    def clear_debug(self) -> None:
        """Clear per-step debug primitives (vectors / points / plots)."""
        ...

    def draw_vector(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        size: float = 2.0,
        color: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    ): ...

    def draw_point(
        self,
        x: torch.Tensor,
        color: tuple[float, ...] = (1.0, 0.0, 0.0, 1.0),
        size: float = 10.0,
    ): ...

    def draw_plot(
        self,
        x: torch.Tensor,
        size: float = 2.0,
        color: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    ): ...


__all__ = [
    "SimAdapter",
    "SceneAdapter",
    "CameraFrustumHandle",
]

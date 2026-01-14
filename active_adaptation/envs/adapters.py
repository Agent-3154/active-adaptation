"""Adapter classes to provide a unified API for different simulation backends."""

from typing import Protocol, TYPE_CHECKING, Union
from typing_extensions import override
import torch

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext
    from isaaclab.scene import InteractiveScene
    from mjlab.sim import Simulation
    from mjlab.scene import Scene


class SimAdapter(Protocol):
    """Unified interface for simulation objects across backends.

    This Protocol is used only for type checking - it has zero runtime overhead.
    """

    def get_physics_dt(self) -> float:
        """Get the physics timestep."""
        ...

    def has_gui(self) -> bool:
        """Check if GUI is available."""
        ...

    def step(self, render: bool = False) -> None:
        """Step the simulation."""
        ...

    def render(self) -> None:
        """Render the simulation."""
        ...

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        """Set camera view (optional, backend-specific)."""
        ...


class SceneAdapter(Protocol):
    """Unified interface for scene objects across backends.

    This Protocol is used only for type checking - it has zero runtime overhead.
    """

    _scene: Union["InteractiveScene", "Scene"]

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._scene.num_envs

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset environments."""
        self._scene.reset(env_ids)

    def update(self, dt: float) -> None:
        """Update scene state."""
        self._scene.update(dt)

    def write_data_to_sim(self) -> None:
        """Write data to simulation."""
        self._scene.write_data_to_sim()

    def zero_external_wrenches(self) -> None:
        """Zero external wrenches."""
        raise NotImplementedError(
            f"Zero external wrenches is not implemented for {self.__class__.__name__}."
        )

    @property
    def articulations(self) -> dict:
        """Dictionary of articulations (robots)."""
        ...

    @property
    def sensors(self) -> dict:
        """Dictionary of sensors."""
        return self._scene.sensors

    @property
    def env_origins(self) -> torch.Tensor:
        """Origins of the environments."""
        return self._scene.env_origins


class IsaacSimAdapter:
    """Adapter for IsaacLab SimulationContext."""

    def __init__(self, sim: "SimulationContext"):
        self._sim = sim

    def get_physics_dt(self) -> float:
        return self._sim.get_physics_dt()

    def has_gui(self) -> bool:
        return self._sim.has_gui()

    def step(self, render: bool = False) -> None:
        self._sim.step(render=render)

    def render(self) -> None:
        self._sim.render()

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        if eye is not None and target is not None:
            self._sim.set_camera_view(eye=eye, target=target)

    def __getattr__(self, name):
        # Forward any other attributes to the underlying sim
        return getattr(self._sim, name)


class MujocoSimAdapter:
    """Adapter for MuJoCo MJSim."""

    def __init__(self, sim):
        self._sim = sim

    def get_physics_dt(self) -> float:
        return self._sim.get_physics_dt()

    def has_gui(self) -> bool:
        return self._sim.has_gui()

    def step(self, render: bool = False) -> None:
        self._sim.step(render=render)

    def render(self) -> None:
        self._sim.render()

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        # MuJoCo doesn't have set_camera_view, but we can add it if needed
        pass

    def __getattr__(self, name):
        return getattr(self._sim, name)


class MjlabSimAdapter:
    """Adapter for mjlab Simulation."""

    def __init__(self, sim: "Simulation", viewer=None):
        self._sim = sim
        self.viewer = viewer

    def get_physics_dt(self) -> float:
        return self._sim.cfg.mujoco.timestep

    def has_gui(self) -> bool:
        # mjlab doesn't have GUI support yet
        return self.viewer is not None

    def step(self, render: bool = False) -> None:
        # mjlab's step doesn't take render parameter
        self._sim.step()

    def render(self) -> None:
        # mjlab doesn't have render method yet
        pass

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        # mjlab doesn't have set_camera_view
        pass

    def __getattr__(self, name):
        return getattr(self._sim, name)


class IsaacSceneAdapter(SceneAdapter):
    """Adapter for IsaacLab InteractiveScene."""

    def __init__(self, scene: "InteractiveScene"):
        self._scene = scene

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.articulations.values():
            if asset.has_external_wrench:
                asset._external_force_b.zero_()
                asset._external_torque_b.zero_()
                asset.has_external_wrench = False

    @property
    def articulations(self):
        return self._scene.articulations

    @property
    def rigid_objects(self):
        return self._scene.rigid_objects

    @property
    def terrain(self):
        return self._scene.terrain

    def __getitem__(self, name):
        return self._scene[name]


class MujocoSceneAdapter(SceneAdapter):
    """Adapter for MuJoCo MJScene."""

    def __init__(self, scene):
        self._scene = scene

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.articulations.values():
            if asset.has_external_wrench:
                asset._external_force_b.zero_()
                asset._external_torque_b.zero_()
                asset.has_external_wrench = False

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    def reset(self, env_ids: torch.Tensor) -> None:
        self._scene.reset(env_ids)

    def update(self, dt: float) -> None:
        self._scene.update(dt)

    def write_data_to_sim(self) -> None:
        self._scene.write_data_to_sim()

    @property
    def articulations(self) -> dict:
        return self._scene.articulations

    def __getitem__(self, name):
        return self._scene[name]


class MjlabSceneAdapter(SceneAdapter):
    """Adapter for mjlab Scene."""

    def __init__(self, scene: "Scene"):
        self._scene = scene

    @property
    def articulations(self) -> dict:
        # mjlab uses 'entities' instead of 'articulations'
        # Return entities dict for compatibility
        return self._scene.entities

    def __getitem__(self, name):
        return self._scene.entities[name]


def wrap_sim(sim, backend: str) -> SimAdapter:
    """Wrap a simulation object with the appropriate adapter."""
    if backend == "isaac":
        return IsaacSimAdapter(sim)
    elif backend == "mujoco":
        return MujocoSimAdapter(sim)
    elif backend == "mjlab":
        return MjlabSimAdapter(sim)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def wrap_scene(scene, backend: str) -> SceneAdapter:
    """Wrap a scene object with the appropriate adapter."""
    if backend == "isaac":
        return IsaacSceneAdapter(scene)
    elif backend == "mujoco":
        return MujocoSceneAdapter(scene)
    elif backend == "mjlab":
        return MjlabSceneAdapter(scene)
    else:
        raise ValueError(f"Unknown backend: {backend}")

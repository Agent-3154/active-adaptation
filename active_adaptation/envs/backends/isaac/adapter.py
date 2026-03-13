from typing import TYPE_CHECKING

import torch
from typing_extensions import override

from active_adaptation.envs.adapters import SimAdapter, SceneAdapter

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext


class IsaacSimAdapter(SimAdapter):
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
        return getattr(self._sim, name)


class IsaacSceneAdapter(SceneAdapter):
    def __init__(self, scene: "InteractiveScene"):
        self._scene: "InteractiveScene" = scene

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.articulations.values():
            if hasattr(asset, "instantaneous_wrench_composer"):
                asset.instantaneous_wrench_composer.reset()
            if hasattr(asset, "permanent_wrench_composer"):
                asset.permanent_wrench_composer.reset()
            if getattr(asset, "has_external_wrench", False):
                asset._external_force_b.zero_()
                asset._external_torque_b.zero_()
                asset.has_external_wrench = False

    @property
    def articulations(self):
        return self._scene.articulations

    @property
    def rigid_objects(self):
        return self._scene.rigid_objects

    def __getattr__(self, name):
        return getattr(self._scene, name)

    @override
    def get_spawn_origins(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self._scene.terrain is None:
            return self.env_origins[env_ids]

        terrain_origins = self._scene.terrain.terrain_origins.reshape(-1, 3)
        idx = torch.randint(
            0,
            terrain_origins.shape[0],
            (len(env_ids),),
            device=env_ids.device,
        )
        return terrain_origins[idx]


__all__ = [
    "IsaacSimAdapter",
    "IsaacSceneAdapter",
]

from typing import TYPE_CHECKING

from typing_extensions import override

from active_adaptation.envs.adapters import SceneAdapter

if TYPE_CHECKING:
    from active_adaptation.envs.backends.mjlab.viewer import MjLabViewer
    from mjlab.scene import Scene
    from mjlab.sim import Simulation


class MjlabSimAdapter:
    def __init__(self, sim: "Simulation", viewer: "MjLabViewer" = None):
        self._sim = sim
        self.viewer = viewer

    def get_physics_dt(self) -> float:
        return self._sim.cfg.mujoco.timestep

    def has_gui(self) -> bool:
        return self.viewer is not None

    def step(self, render: bool = False) -> None:
        self._sim.step()

    def render(self) -> None:
        pass

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        pass

    def __getattr__(self, name):
        return getattr(self._sim, name)


class MjlabSceneAdapter(SceneAdapter):
    def __init__(self, scene: "Scene"):
        self._scene: "Scene" = scene

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.entities.values():
            asset.data.data.xfrc_applied.zero_()

    @property
    def articulations(self):
        return self._scene.entities

    def __getattr__(self, name):
        return getattr(self._scene, name)


__all__ = [
    "MjlabSimAdapter",
    "MjlabSceneAdapter",
]

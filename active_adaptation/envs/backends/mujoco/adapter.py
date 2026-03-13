from typing_extensions import override

from active_adaptation.envs.adapters import SceneAdapter


class MujocoSimAdapter:
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
        pass

    def __getattr__(self, name):
        return getattr(self._sim, name)


class MujocoSceneAdapter(SceneAdapter):
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

    def reset(self, env_ids) -> None:
        self._scene.reset(env_ids)

    def update(self, dt: float) -> None:
        self._scene.update(dt)

    def write_data_to_sim(self) -> None:
        self._scene.write_data_to_sim()

    @property
    def articulations(self) -> dict:
        return self._scene.articulations

    def __getattr__(self, name):
        return getattr(self._scene, name)


__all__ = [
    "MujocoSimAdapter",
    "MujocoSceneAdapter",
]

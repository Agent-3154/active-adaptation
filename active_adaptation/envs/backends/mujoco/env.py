from typing import cast
from dataclasses import dataclass

from active_adaptation.envs.backends.mujoco.adapter import (
    MujocoSceneAdapter,
    MujocoSimAdapter,
)
from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.registry import Registry


class MujocoBackendEnv(_EnvBase):
    """MuJoCo backend env: only scene/sim construction."""

    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]

    def setup_scene(self):
        from active_adaptation.envs.backends.mujoco.mujoco import MJScene, MJSim
        from active_adaptation.envs.terrain import TERRAINS_MUJOCO

        registry = Registry.instance()
        asset_cfg = registry.get("asset", self.cfg.robot.name)
        asset_cfg, _ = asset_cfg(backend="mujoco")

        @dataclass
        class SceneCfg:
            robot = asset_cfg
            contact_forces = "robot"
            terrain = TERRAINS_MUJOCO.get(self.cfg.terrain, TERRAINS_MUJOCO["plane"])

        scene = MJScene(SceneCfg())
        sim = MJSim(scene)
        self.scene = MujocoSceneAdapter(scene)
        self.sim = MujocoSimAdapter(sim)
        self.terrain_type = "plane"


__all__ = ["MujocoBackendEnv"]

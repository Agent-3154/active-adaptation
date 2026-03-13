from typing import cast

try:
    from isaaclab.utils import configclass
except ModuleNotFoundError:
    def configclass(cls):
        return cls

from active_adaptation.assets import AssetCfg
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
        asset_cfg = cast(AssetCfg, registry.get("asset", self.cfg.robot.name))

        @configclass
        class SceneCfg:
            robot = asset_cfg.mujoco()
            contact_forces = "robot"
            terrain = TERRAINS_MUJOCO.get(self.cfg.terrain, TERRAINS_MUJOCO["plane"])

        scene = MJScene(SceneCfg())
        sim = MJSim(scene)
        self.scene = MujocoSceneAdapter(scene)
        self.sim = MujocoSimAdapter(sim)
        self.terrain_type = "plane"


__all__ = ["MujocoBackendEnv"]

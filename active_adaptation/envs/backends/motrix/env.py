from active_adaptation.envs.backends.motrix.adapter import (
    MotrixScene,
    MotrixSceneCfg,
    MotrixSim,
)
from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.registry import Registry


class MotrixBackendEnv(_EnvBase):
    """MotrixSim backend env: scene/sim construction only."""

    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]

    def setup_scene(self):
        registry = Registry.instance()
        asset_cfg = registry.get("asset", self.cfg.robot.name)
        sensors = []
        asset_cfg, _sensors = asset_cfg(backend="motrix")
        sensors.extend(_sensors)

        self.terrain_type = self.cfg.get("terrain", "plane")
        scene_cfg = MotrixSceneCfg(
            num_envs=self.cfg.num_envs,
            env_spacing=2.5,
            entities={"robot": asset_cfg},
            sensors=tuple(sensors)
        )
        self.scene = MotrixScene(scene_cfg)
        self.sim = MotrixSim(self.scene, headless=self.headless)


__all__ = ["MotrixBackendEnv"]

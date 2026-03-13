from typing import cast

from active_adaptation.assets import AssetCfg
from active_adaptation.envs.backends.mjlab.adapter import (
    MjlabSceneAdapter,
    MjlabSimAdapter,
)
from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.registry import Registry


class MjlabBackendEnv(_EnvBase):
    """MjLab backend env: scene/sim construction and viewer glue."""

    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]
        if self.sim.has_gui():
            self.sim.viewer.setup()
            self.sim.viewer.update()

    def setup_scene(self):
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
        from mjlab.scene import Scene, SceneCfg
        import mjlab.terrains as terrain_gen
        from mjlab.terrains import TerrainEntityCfg
        from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

        from active_adaptation.envs.backends.mjlab.viewer import MjLabViewer

        registry = Registry.instance()
        asset_cfg = cast(AssetCfg, registry.get("asset", self.cfg.robot.name))
        sensors = tuple(sensor.mjlab() for sensor in asset_cfg.sensors_mjlab)
        terrain = self.cfg.get("terrain", "plane")
        self.terrain_type = terrain

        env_spacing = 2.5
        if terrain == "plane":
            terrain_cfg = TerrainEntityCfg(terrain_type="plane")
        elif terrain == "rough":
            terrain_cfg = TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=TerrainGeneratorCfg(
                    size=(5.0, 5.0),
                    border_width=20.0,
                    num_rows=8,
                    num_cols=8,
                    sub_terrains={
                        "boxes": terrain_gen.BoxRandomGridTerrainCfg(
                            proportion=1.0,
                            grid_width=0.5,
                            grid_height_range=(0.001, 0.005),
                            platform_width=0.5,
                        )
                    },
                    add_lights=False,
                ),
                env_spacing=env_spacing,
                num_envs=self.cfg.num_envs,
            )
        else:
            raise ValueError(
                f"Unsupported terrain `{terrain}`. Expected one of: `plane`, `rough`."
            )

        scene_cfg = SceneCfg(
            num_envs=self.cfg.num_envs,
            env_spacing=env_spacing,
            entities={"robot": asset_cfg.mjlab()},
            sensors=sensors,
            terrain=terrain_cfg,
        )
        scene = Scene(scene_cfg, device=str(self.device))
        self.sim = Simulation(
            num_envs=scene.num_envs,
            cfg=SimulationCfg(
                nconmax=200,
                njmax=500,
                contact_sensor_maxmatch=80,
                mujoco=MujocoCfg(
                    timestep=0.005,
                    iterations=10,
                    ls_iterations=20,
                ),
            ),
            model=scene.compile(),
            device=str(self.device),
        )

        scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.sim.create_graph()

        self.scene = MjlabSceneAdapter(scene)
        viewer = MjLabViewer(self) if not self.headless else None
        self.sim = MjlabSimAdapter(self.sim, viewer)


__all__ = ["MjlabBackendEnv"]

import math
import mujoco
from typing import cast

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
        from mjlab.viewer import ViewerConfig
        from mjlab.utils.spec_config import CollisionCfg, _GEOM_ATTR_DEFAULTS

        def edit_spec(self: CollisionCfg, spec: mujoco.MjSpec):
            from mjlab.utils.spec import disable_collision
            from mjlab.utils.string import filter_exp, resolve_field

            self.validate()

            all_geoms: list[mujoco.MjsGeom] = spec.geoms
            all_geom_names = tuple(g.name for g in all_geoms)
            geom_subset = filter_exp(self.geom_names_expr, all_geom_names)

            resolved_fields = {
                name: resolve_field(getattr(self, name), geom_subset, default)
                for name, default in _GEOM_ATTR_DEFAULTS.items()
            }

            # raise error if any of the resolved fields are None
            if any(not len(field) for field in resolved_fields.values()):
                raise ValueError("Resolved fields cannot be empty")

            for i, geom_name in enumerate(geom_subset):
                geom = spec.geom(geom_name)

                geom.condim = resolved_fields["condim"][i]
                geom.contype = resolved_fields["contype"][i]
                geom.conaffinity = resolved_fields["conaffinity"][i]
                geom.priority = resolved_fields["priority"][i]

                CollisionCfg.set_array_field(geom.friction, resolved_fields["friction"][i])
                CollisionCfg.set_array_field(geom.solref, resolved_fields["solref"][i])
                CollisionCfg.set_array_field(geom.solimp, resolved_fields["solimp"][i])

                if resolved_fields["margin"][i] is not None:
                    geom.margin = resolved_fields["margin"][i]
                if resolved_fields["gap"][i] is not None:
                    geom.gap = resolved_fields["gap"][i]
                if resolved_fields["solmix"][i] is not None:
                    geom.solmix = resolved_fields["solmix"][i]

            other_geoms = ()
            if self.disable_other_geoms:
                other_geoms = set(all_geom_names).difference(geom_subset)
                for geom_name in other_geoms:
                    geom = spec.geom(geom_name)
                    disable_collision(geom)
        
        # replace the edit_spec method of CollisionCfg with our own
        CollisionCfg.edit_spec = edit_spec

        from active_adaptation.envs.backends.mjlab.viewer import MjLabViewer

        registry = Registry.instance()
        asset_cfg = registry.get("asset", self.cfg.robot.name)
        asset_cfg, sensors = asset_cfg(backend="mjlab")
        terrain = self.cfg.get("terrain", "plane")
        
        self.terrain_type = terrain

        env_spacing = 2.5
        if terrain == "plane":
            terrain_cfg = TerrainEntityCfg(terrain_type="plane")
        else:
            raise ValueError(
                f"Unsupported terrain `{terrain}`. Expected one of: `plane`, `rough`."
            )

        scene_cfg = SceneCfg(
            num_envs=self.cfg.num_envs,
            env_spacing=env_spacing,
            entities={"robot": asset_cfg},
            sensors=sensors,
            terrain=terrain_cfg,
        )
        scene = Scene(scene_cfg, device=str(self.device))
        sim = Simulation(
            num_envs=scene.num_envs,
            cfg=SimulationCfg(
                nconmax=self.cfg.sim.get("nconmax", 200),
                njmax=self.cfg.sim.get("njmax", 500),
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

        scene.initialize(sim.mj_model, sim.model, sim.data)
        sim.create_graph()

        viewer_cfg = self._make_viewer_cfg(ViewerConfig)
        viewer = MjLabViewer(self, sim) if not self.headless else None
        self.scene = MjlabSceneAdapter(scene, sim, viewer=viewer)
        self.sim = MjlabSimAdapter(sim, viewer, viewer_cfg=viewer_cfg, scene=scene)

    def _make_viewer_cfg(self, viewer_config_cls):
        lookat = tuple(float(v) for v in self.cfg.viewer.lookat)
        eye = tuple(float(v) for v in self.cfg.viewer.eye)
        resolution = tuple(int(v) for v in self.cfg.viewer.resolution)

        delta = [eye_i - lookat_i for eye_i, lookat_i in zip(eye, lookat)]
        distance = math.sqrt(sum(v * v for v in delta))
        if distance <= 1e-8:
            distance = 5.0
            azimuth = 90.0
            elevation = -45.0
        else:
            planar = math.hypot(delta[0], delta[1])
            azimuth = math.degrees(math.atan2(delta[1], delta[0]))
            elevation = -math.degrees(math.atan2(delta[2], planar))

        return viewer_config_cls(
            lookat=lookat,
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            width=resolution[0],
            height=resolution[1],
            env_idx=0,
            max_extra_envs=max(0, self.cfg.num_envs - 1),
        )


__all__ = ["MjlabBackendEnv"]

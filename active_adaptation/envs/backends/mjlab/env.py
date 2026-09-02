import math
import mujoco
from typing import cast

from active_adaptation.envs.backends.mjlab.adapter import (
    MjlabSceneAdapter,
    MjlabSimAdapter,
)
from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.assets.asset_cfg import AssetSpec, coerce_asset_spec
from active_adaptation.registry import Registry


class MjlabBackendEnv(_EnvBase):
    """MjLab backend env: scene/sim construction and viewer glue."""

    # TODO: simplify
    def _register_wrapper_callbacks(self, wrapper) -> None:
        if wrapper is None:
            return
        if callable(getattr(wrapper, "startup", None)):
            self._startup_callbacks.append(wrapper.startup)
        if callable(getattr(wrapper, "reset", None)):
            self._reset_callbacks.append(wrapper.reset)
        if callable(getattr(wrapper, "pre_step", None)):
            self._pre_step_callbacks.append(wrapper.pre_step)
        elif callable(getattr(wrapper, "write_data_to_sim", None)):
            self._pre_step_callbacks.append(lambda _substep: wrapper.write_data_to_sim())
        if callable(getattr(wrapper, "post_step", None)):
            self._post_step_callbacks.append(wrapper.post_step)
        if callable(getattr(wrapper, "update", None)):
            self._post_group_update_callbacks.append(wrapper.update)
        if callable(getattr(wrapper, "debug_draw", None)):
            self._debug_draw_callbacks.append(wrapper.debug_draw)

    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]
        if self.sim.has_gui():
            self._debug_draw_callbacks.insert(0, self.scene.clear_debug)
            self.sim.viewer.setup()
            self.sim.viewer.update()

    def setup_scene(self):
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
        from mjlab.scene import Scene, SceneCfg
        import mjlab.terrains as terrain_gen
        from mjlab.terrains import TerrainEntityCfg
        from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
        from mjlab.viewer import ViewerConfig
        # mjlab 1.6+ still silently zeros all collisions when geom_names_expr
        # matches nothing and disable_other_geoms=True. Keep a thin fail-fast.
        from mjlab.utils.spec_config import CollisionCfg
        from mjlab.utils.string import filter_exp

        if not getattr(CollisionCfg.edit_spec, "_aa_empty_match_guard", False):
            _collision_edit_spec = CollisionCfg.edit_spec

            def _edit_spec_require_matches(
                self: CollisionCfg, spec: mujoco.MjSpec
            ) -> None:
                matched = filter_exp(
                    self.geom_names_expr, tuple(g.name for g in spec.geoms)
                )
                if not matched:
                    raise ValueError(
                        f"CollisionCfg geom_names_expr={self.geom_names_expr!r} "
                        "matched no geoms; with disable_other_geoms=True this would "
                        "silently disable all collisions."
                    )
                _collision_edit_spec(self, spec)

            _edit_spec_require_matches._aa_empty_match_guard = True  # type: ignore[attr-defined]
            CollisionCfg.edit_spec = _edit_spec_require_matches

        from active_adaptation.envs.backends.mjlab.viewer import MjLabViewer

        registry = Registry.instance()
        robot_cfg = dict(self.cfg.robot.copy())
        asset_entry = registry.get("asset", robot_cfg.pop("name"))
        asset_spec: AssetSpec = coerce_asset_spec(
            asset_entry, backend="mjlab", **robot_cfg
        )
        asset_cfg = asset_spec.config
        sensors = {sensor.name: sensor for sensor in asset_spec.sensors}
        terrain = self.cfg.get("terrain", "plane")
        
        self.terrain_type = terrain

        if terrain == "plane":
            terrain_cfg = TerrainEntityCfg(terrain_type="plane")
        else:
            raise ValueError(
                f"Unsupported terrain `{terrain}`. Expected one of: `plane`, `rough`."
            )

        entities = {"robot": asset_cfg}
        for obj_name, obj_spec in self.cfg.get("objects", {}).items():
            obj_spec = dict(obj_spec)
            asset_entry = registry.get("asset", obj_spec.pop("_target_"))
            object_spec = coerce_asset_spec(
                asset_entry, backend="mjlab", **obj_spec
            )
            entities[obj_name] = object_spec.config

        import active_adaptation.envs.sensors  # noqa: F401  # register sensor factories
        from active_adaptation.envs.sensors.warp_base import (
            WarpSensorSpec,
            install_warp_sensors,
            is_warp_sensor_spec,
        )

        warp_sensor_specs: dict[str, WarpSensorSpec] = {}
        for sensor_name, sensor_spec in self.cfg.get("sensors", {}).items():
            sensor_spec = dict(sensor_spec)
            fn = registry.get("sensor", sensor_spec.pop("_target_"))
            result = fn(
                backend="mjlab", name=sensor_name, **sensor_spec
            )
            if is_warp_sensor_spec(result):
                warp_sensor_specs[sensor_name] = result
            else:
                sensors[sensor_name] = result

        scene_cfg = SceneCfg(
            num_envs=self.cfg.num_envs,
            env_spacing=self.cfg.get("env_spacing", 2.5),
            entities=entities,
            sensors=tuple(sensors.values()),
            terrain=terrain_cfg,
        )

        self._edit_scene_spec(scene_cfg)

        scene = Scene(scene_cfg, device=str(self.device))
        sim = Simulation(
            num_envs=scene.num_envs,
            cfg=SimulationCfg(
                nconmax=self.cfg.sim.get("nconmax", 200),
                njmax=self.cfg.sim.get("njmax", 500),
                contact_sensor_maxmatch=80,
                mujoco=MujocoCfg(
                    timestep=self.cfg.sim.get("mujoco_physics_dt", 0.005),
                    iterations=self.cfg.sim.get("mujoco_iterations", 10),
                    ls_iterations=self.cfg.sim.get("mujoco_ls_iterations", 20),
                ),
                broadphase=self.cfg.sim.get("broadphase", None), # nxn, sap_tile, sap_segmented
            ),
            model=scene.compile(),
            device=str(self.device),
        )

        scene.initialize(sim.mj_model, sim.model, sim.data)
        if scene.sensor_context is not None:
            sim.set_sensor_context(scene.sensor_context)
        sim.create_graph()

        viewer_cfg = self._make_viewer_cfg(ViewerConfig)
        viewer = MjLabViewer(self, sim) if not self.headless else None
        self.scene = MjlabSceneAdapter(scene, sim, viewer=viewer)
        install_warp_sensors(self.scene, warp_sensor_specs, env=self)
        self.sim = MjlabSimAdapter(sim, viewer, viewer_cfg=viewer_cfg, scene=scene)
        self.robot = self.scene.articulations["robot"]
        if asset_spec.wrapper is not None:
            self.robot_wrapper = asset_spec.wrapper
            self.robot_wrapper._initialize(robot=self.robot, env=self)
            self._register_wrapper_callbacks(self.robot_wrapper)
        else:
            self.robot_wrapper = None

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

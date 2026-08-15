from typing_extensions import override
from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.envs.backends.isaac.adapter import (
    IsaacSceneAdapter,
    IsaacSimAdapter,
    IsaacDebugDraw
)
from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.assets.asset_cfg import AssetSpec
from active_adaptation.registry import Registry
from tqdm import tqdm

class IsaacBackendEnv(_EnvBase):
    """Isaac backend env: scene/sim construction and viewer glue."""

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
            # Wrapper can choose to provide only the force/wrench write path.
            self._pre_step_callbacks.append(lambda _substep: wrapper.write_data_to_sim())
        if callable(getattr(wrapper, "post_step", None)):
            self._post_step_callbacks.append(wrapper.post_step)
        if callable(getattr(wrapper, "update", None)):
            self._update_callbacks.append(wrapper.update)
        if callable(getattr(wrapper, "debug_draw", None)):
            self._debug_draw_callbacks.append(wrapper.debug_draw)

    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]

        if self.sim._sim.has_gui():
            from isaaclab.envs import ViewerCfg
            from isaaclab.envs.ui import BaseEnvWindow, ViewportCameraController


            self.cfg.viewer.env_index = 0
            self.manager_visualizers = {}
            self.window = BaseEnvWindow(self, window_name="IsaacLab")
            self.viewport_camera_controller = ViewportCameraController(
                self,
                ViewerCfg(self.cfg.viewer.eye, self.cfg.viewer.lookat, origin_type="env"),
            )

    @override
    def setup_scene(self):
        import isaaclab.sim as sim_utils
        from isaaclab.sim import (
            SimulationContext,
            attach_stage_to_usd_context,
            use_stage,
        )
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        registry = Registry.instance()
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.cfg.num_envs,
            env_spacing=self.cfg.get("env_spacing", 2.5),
            replicate_physics=True,
        )
        scene_cfg.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=str(
                    ROBOT_MODEL_DIR / "scene" / "kloofendal_43d_clear_puresky_4k.hdr"
                ),
            ),
        )

        robot_cfg = dict(self.cfg.robot.copy())
        asset_factory = registry.get("asset", robot_cfg.pop("name"))
        asset_spec: AssetSpec = asset_factory(backend="isaaclab", **robot_cfg)
        scene_cfg.robot = asset_spec.config
        sensors = asset_spec.sensors
        for name, sensor_cfg in sensors.items():
            setattr(scene_cfg, name, sensor_cfg)

        scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        terrain_name = self.cfg.get("terrain", "plane")
        scene_cfg.terrain = registry.get("terrain", terrain_name)

        from isaaclab.assets import ArticulationCfg, RigidObjectCfg

        for obj_name, spec in self.cfg.get("objects", {}).items():
            fn = registry.get("asset", spec.pop("_target_"))
            cfg = fn(backend="isaaclab", **spec)
            assert isinstance(cfg, (ArticulationCfg, RigidObjectCfg)), f"Asset configuration must be an instance of ArticulationCfg or RigidObjectCfg, got {type(cfg)}"
            cfg.prim_path = "{ENV_REGEX_NS}/" + obj_name
            setattr(scene_cfg, obj_name, cfg)

        for sensor_name, sensor_spec in self.cfg.get("sensors", {}).items():
            sensor_spec = dict(sensor_spec)
            fn = registry.get("sensor", sensor_spec.pop("_target_"))
            cfg = fn(backend="isaaclab", name=sensor_name, **sensor_spec)
            setattr(scene_cfg, sensor_name, cfg)
        
        for observation in self.observation_groups.values():
            for func in observation.funcs.values():
                func.edit_spec(scene_cfg)

        sim_cfg = sim_utils.SimulationCfg(
            dt=self.cfg.sim.isaac_physics_dt,
            render=sim_utils.RenderCfg(rendering_mode="balanced"),
            physx=sim_utils.PhysxCfg(**self.cfg.sim.get("physx", {})),
            device=str(self.device),
        )

        sim = SimulationContext.instance() or SimulationContext(sim_cfg)
        with use_stage(sim.get_initial_stage()):
            self.scene = InteractiveScene(scene_cfg)
            attach_stage_to_usd_context()
        with use_stage(sim.get_initial_stage()):
            sim.reset()
        
        # Try to fix headless record
        # --------------------------
        from pxr import UsdGeom, Gf

        stage = sim.get_initial_stage()

        camera_path = "/World/RecordCamera"

        camera = UsdGeom.Camera.Define(stage, camera_path)

        xform = UsdGeom.Xformable(camera)
        xform.AddTranslateOp().Set(
            Gf.Vec3d(
                self.cfg.viewer.eye[0],
                self.cfg.viewer.eye[1],
                self.cfg.viewer.eye[2],
            )
        )
        # --------------------------
        
        # warm up the simulation
        for _ in tqdm(range(10), desc="Warming up the simulation"):
            sim.step(render=False)

        sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat, camera_prim_path=camera_path)
        try:
            import omni.replicator.core as rep

            self._render_product = rep.create.render_product(
                camera_path, tuple(self.cfg.viewer.resolution)
            )
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator(
                "rgb", device="cpu"
            )
            self._rgb_annotator.attach([self._render_product])
            for _ in range(5):
                sim.step(render=True)
        except ModuleNotFoundError:
            print("Set enable_cameras=true to use cameras.")

        if self.cfg.viewer.get("viser", False):
            from active_adaptation.envs.backends.isaac.viewer import IsaacViserViewer
            viser_viewer = IsaacViserViewer(self)
            viser_viewer.setup()
        else:
            viser_viewer = None
        
        if sim.has_gui(): # native isaac gui
            debug_draw = IsaacDebugDraw()
        else:
            debug_draw = None

        self.sim = IsaacSimAdapter(sim, viser_viewer)
        self.scene = IsaacSceneAdapter(self.scene, viser_viewer, debug_draw)
        self.terrain_type = self.scene.terrain.cfg.terrain_type
        self.robot = self.scene.articulations["robot"]

        self._debug_draw_callbacks.insert(0, self.scene.clear_debug)

        if asset_spec.wrapper is not None:
            self.robot_wrapper = asset_spec.wrapper
            self.robot_wrapper._initialize(robot=self.robot, env=self)
            self._register_wrapper_callbacks(self.robot_wrapper)
        else:
            self.robot_wrapper = None
    
    @override
    def _setup_visual(self) -> None:
        super()._setup_visual()
        if self.visual is None:
            return
        viser_viewer = self.sim._viser_viewer
        if viser_viewer is None:
            return
        # Collision mesh first: cheap Viser proxy for InteriorGS geometry.
        collision = getattr(self.visual, "collision_mesh", None)
        if collision is not None:
            viser_viewer.add_collision_mesh(collision)
        gs = getattr(self.visual, "gaussian_splat", None) or getattr(
            self.visual, "_gs", None
        )
        if gs is not None:
            # Uploaded hidden by default (browser 3DGS is costly).
            viser_viewer.add_gaussian_splat(gs)


__all__ = ["IsaacBackendEnv"]

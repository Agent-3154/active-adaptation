from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.assets import AssetCfg
from active_adaptation.envs.backends.isaac.adapter import (
    IsaacSceneAdapter,
    IsaacSimAdapter,
)
from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.registry import Registry


class IsaacBackendEnv(_EnvBase):
    """Isaac backend env: scene/sim construction and viewer glue."""

    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]

        if self.sim.has_gui():
            from isaaclab.envs import ViewerCfg
            from isaaclab.envs.ui import BaseEnvWindow, ViewportCameraController

            from active_adaptation.utils.debug import DebugDraw

            self.cfg.viewer.env_index = 0
            self.manager_visualizers = {}
            self.window = BaseEnvWindow(self, window_name="IsaacLab")
            self.viewport_camera_controller = ViewportCameraController(
                self,
                ViewerCfg(self.cfg.viewer.eye, self.cfg.viewer.lookat, origin_type="env"),
            )
            self.debug_draw = DebugDraw()

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
            env_spacing=2.5,
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

        asset_cfg = registry.get("asset", self.cfg.robot.name)
        if not isinstance(asset_cfg, AssetCfg):
            raise ValueError(
                "Asset configuration must be an instance of AssetCfg, "
                f"got {type(asset_cfg)}"
            )
        scene_cfg.robot = asset_cfg.isaaclab()
        for sensor_cfg in asset_cfg.sensors_isaaclab:
            setattr(scene_cfg, sensor_cfg.name, sensor_cfg.isaaclab())

        scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        terrain_name = self.cfg.get("terrain", "plane")
        scene_cfg.terrain = registry.get("terrain", terrain_name)

        for obj in self.cfg.get("objects", []):
            obj_cfg = registry.get("asset", obj.name).isaaclab()
            obj_cfg.prim_path = "{ENV_REGEX_NS}/" + obj.name
            setattr(scene_cfg, obj.name, obj_cfg)

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

        sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
        try:
            import omni.replicator.core as rep

            self._render_product = rep.create.render_product(
                "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
            )
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator(
                "rgb", device="cpu"
            )
            self._rgb_annotator.attach([self._render_product])
        except ModuleNotFoundError:
            print("Set enable_cameras=true to use cameras.")

        self.sim = IsaacSimAdapter(sim)
        self.scene = IsaacSceneAdapter(self.scene)
        self.terrain_type = self.scene.terrain.cfg.terrain_type


__all__ = ["IsaacBackendEnv"]

import os
import json
import torch
from isaaclab.utils import configclass

import active_adaptation
from active_adaptation.envs.base import _Env
from active_adaptation.asset import AssetCfg
from active_adaptation.registry import Registry
from typing import cast

import active_adaptation.assets
from active_adaptation.viewer import MjLabViewer


class SimpleEnvIsaac(_Env):
    """Isaac Sim backend implementation."""
    
    def __init__(self, cfg, device: str):
        super().__init__(cfg, device)
        self.robot = self.scene.articulations["robot"]
        
        if self.sim.has_gui():
            from isaaclab.envs.ui import BaseEnvWindow, ViewportCameraController
            from isaaclab.envs import ViewerCfg
            # hacks to make IsaacLab happy. we don't use them.
            self.cfg.viewer.env_index = 0
            self.manager_visualizers = {}
            self.window = BaseEnvWindow(self, window_name="IsaacLab")
            self.viewport_camera_controller = ViewportCameraController(
                self,
                ViewerCfg(self.cfg.viewer.eye, self.cfg.viewer.lookat, origin_type="env")
            )
    
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        self.stats[env_ids] = 0.
    
    def setup_scene(self):
        import active_adaptation.envs.scene as scene
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
        
        registry = Registry.instance()
        
        scene_cfg = InteractiveSceneCfg(num_envs=self.cfg.num_envs, env_spacing=2.5)
        scene_cfg.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        asset_cfg = cast(AssetCfg, registry.get("asset", self.cfg.robot.name))
        
        scene_cfg.robot = asset_cfg.isaaclab()
        scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        scene_cfg.terrain = registry.get("terrain", self.cfg.terrain)

        for sensor_cfg in asset_cfg.sensors_isaaclab:
            setattr(scene_cfg, sensor_cfg.name, sensor_cfg.isaaclab())

        sim_cfg = sim_utils.SimulationCfg(
            dt=self.cfg.sim.isaac_physics_dt,
            render=sim_utils.RenderCfg(
                rendering_mode="balanced",
                # antialiasing_mode="FXAA",
                # enable_global_illumination=True,
                # enable_reflections=True,
            ),
            device=str(self.device)
        )
        
        # slightly reduces GPU memory usage
        # sim_cfg.physx.gpu_max_rigid_contact_count = 2**21
        # sim_cfg.physx.gpu_max_rigid_patch_count = 2**21
        # sim_cfg.physx.gpu_found_lost_pairs_capacity = 2538320 # 2**20
        # sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 61999079 # 2**26
        # sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2**23
        # sim_cfg.physx.gpu_collision_stack_size = 2**25
        # sim_cfg.physx.gpu_heap_capacity = 2**24
        
        self.sim, self.scene = scene.create_isaaclab_sim_and_scene(sim_cfg, scene_cfg)
        # set camera view for "/OmniverseKit_Persp" camera
        self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
        try:
            import omni.replicator.core as rep
            # create render product
            self._render_product = rep.create.render_product(
                "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
            )
            # create rgb annotator -- used to read data from the render product
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
            # self._seg_annotator = rep.AnnotatorRegistry.get_annotator(
            #     "instance_id_segmentation_fast", 
            #     device="cpu",
            # )
            # self._seg_annotator.attach([self._render_product])
            # for _ in range(4):
            #     self.sim.render()
        except ModuleNotFoundError as e:
            print("Set enable_cameras=true to use cameras.")
        
        try:
            from active_adaptation.utils.debug import DebugDraw
            self.debug_draw = DebugDraw()
            print("[INFO] Debug Draw API enabled.")
        except ModuleNotFoundError:
            print()
        
        # asset_meta = get_asset_meta(self.scene["robot"])
        # path = os.path.join(os.getcwd(), "asset_meta.json")
        # print(f"Saving asset meta to {path}")
        # with open(path, "w") as f:
        #     json.dump(asset_meta, f, indent=4)


class SimpleEnvMujoco(_Env):
    """MuJoCo backend implementation."""
    
    def __init__(self, cfg, device: str):
        super().__init__(cfg, device)
        self.robot = self.scene.articulations["robot"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        self.stats[env_ids] = 0.
    
    def setup_scene(self):
        from active_adaptation.envs.mujoco import MJScene, MJSim
        from active_adaptation.registry import Registry
        from active_adaptation.envs.terrain import TERRAINS_MUJOCO

        registry = Registry.instance()
        asset_cfg = cast(AssetCfg, registry.get("asset", self.cfg.robot.name))

        @configclass
        class SceneCfg:
            robot = asset_cfg.mujoco()
            contact_forces = "robot"
            terrain = TERRAINS_MUJOCO.get(self.cfg.terrain, TERRAINS_MUJOCO["plane"])
        
        self.scene = MJScene(SceneCfg())
        self.sim = MJSim(self.scene)


class SimpleEnvMjlab(_Env):
    """MjLab backend implementation."""
    
    def __init__(self, cfg, device: str):
        super().__init__(cfg, device)
        self.robot = self.scene.articulations["robot"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        self.stats[env_ids] = 0.
    
    def setup_scene(self):
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
        from mjlab.terrains import TerrainImporterCfg

        registry = Registry.instance()
        asset_cfg = cast(AssetCfg, registry.get("asset", self.cfg.robot.name))
        # Initialize scene and simulation.
        sensors = tuple(sensor.mjlab() for sensor in asset_cfg.sensors_mjlab)
        scene_cfg = SceneCfg(
            num_envs=self.cfg.num_envs,
            env_spacing=2.5,
            entities={"robot": asset_cfg.mjlab()},
            sensors=sensors,
            terrain=TerrainImporterCfg(terrain_type="plane"),
        )
        self.scene = Scene(scene_cfg, device=str(self.device))
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=SimulationCfg(
                nconmax=50,
                njmax=300,
                mujoco=MujocoCfg(
                    timestep=0.005,
                    iterations=10,
                    ls_iterations=20,
                ),
            ),
            model=self.scene.compile(),
            device=str(self.device),
        )

        self.scene.initialize(
            mj_model=self.sim.mj_model,
            model=self.sim.model,
            data=self.sim.data,
        )
        self.sim.create_graph()
        # MjLabViewer(self).run_async()


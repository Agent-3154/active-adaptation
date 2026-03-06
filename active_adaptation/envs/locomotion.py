import os
import json
import torch
try:
    from isaaclab.utils import configclass
except ModuleNotFoundError:
    def configclass(cls):
        return cls

import active_adaptation
from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.assets import AssetCfg
from active_adaptation.registry import Registry
from active_adaptation.envs.adapters import (
    IsaacSimAdapter, IsaacSceneAdapter,
    MujocoSimAdapter, MujocoSceneAdapter,
    MjlabSimAdapter, MjlabSceneAdapter,
)
from typing import cast


class SimpleEnvIsaac(_EnvBase):
    """Isaac Sim backend implementation."""
    
    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]
        
        if self.sim.has_gui():
            from isaaclab.envs.ui import BaseEnvWindow, ViewportCameraController
            from isaaclab.envs import ViewerCfg
            from active_adaptation.utils.debug import DebugDraw
            # hacks to make IsaacLab happy. we don't use them.
            self.cfg.viewer.env_index = 0
            self.manager_visualizers = {}
            self.window = BaseEnvWindow(self, window_name="IsaacLab")
            self.viewport_camera_controller = ViewportCameraController(
                self,
                ViewerCfg(self.cfg.viewer.eye, self.cfg.viewer.lookat, origin_type="env")
            )
            self.debug_draw = DebugDraw()
    
    def _reset_idx(self, env_ids: torch.Tensor):
        init_state = self.command_manager.sample_init(env_ids)
        if isinstance(init_state, torch.Tensor):
            init_state = {"robot": init_state}
        if init_state is not None:
            for key, value in init_state.items():
                self.scene[key].write_root_state_to_sim(
                    value, 
                    env_ids=env_ids
                )
        self.stats[env_ids] = 0.
    
    def setup_scene(self):
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationContext, attach_stage_to_usd_context, use_stage
        # from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage
        from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
        from isaaclab.assets import AssetBaseCfg, ArticulationCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        
        registry = Registry.instance()
        
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.cfg.num_envs,
            env_spacing=2.5,
            replicate_physics=True
        )
        scene_cfg.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        asset_cfg = registry.get("asset", self.cfg.robot.name)
        if isinstance(asset_cfg, AssetCfg):
            scene_cfg.robot = asset_cfg.isaaclab()
            for sensor_cfg in asset_cfg.sensors_isaaclab:
                setattr(scene_cfg, sensor_cfg.name, sensor_cfg.isaaclab())
        elif isinstance(asset_cfg, ArticulationCfg):
            scene_cfg.robot = asset_cfg
            scene_cfg.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                track_air_time=True,
                history_length=3
            )
        else:
            raise ValueError(f"Asset configuration must be an instance of AssetCfg or ArticulationCfg, got {type(asset_cfg)}")
        
        scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        scene_cfg.terrain = registry.get("terrain", self.cfg.terrain)

        # TODO@btx0424: more general object spawning support.
        if (objects := self.cfg.get("objects", [])):
            from isaaclab.assets import RigidObjectCfg
            from omegaconf import OmegaConf

            objects = OmegaConf.to_container(objects)
            for obj_name, kwargs in objects.items():
                spawner_cfg = {
                    "box": sim_utils.CuboidCfg,
                    "sphere": sim_utils.SphereCfg,
                    "cylinder": sim_utils.CylinderCfg,
                    "capsule": sim_utils.CapsuleCfg,
                    "cone": sim_utils.ConeCfg,
                }[obj_name]
                spawner_cfg = spawner_cfg(
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=False
                    ),
                    **kwargs
                )
                obj_cfg = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + obj_name,
                    spawn=spawner_cfg
                )
                setattr(scene_cfg, obj_name, obj_cfg)

        sim_cfg = sim_utils.SimulationCfg(
            dt=self.cfg.sim.isaac_physics_dt,
            render=sim_utils.RenderCfg(
                rendering_mode="balanced",
                # antialiasing_mode="FXAA",
                # enable_global_illumination=True,
                # enable_reflections=True,
            ),
            physx=sim_utils.PhysxCfg(
                **self.cfg.sim.get("physx", {}),
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

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            # the type-annotation is required to avoid a type-checking error
            # since it gets confused with Isaac Sim's SimulationContext class
            self.sim: SimulationContext = SimulationContext(sim_cfg)
        else:
            self.sim: SimulationContext = SimulationContext.instance()
        
        with use_stage(self.sim.get_initial_stage()):
            self.scene = InteractiveScene(scene_cfg)
            attach_stage_to_usd_context()
        
        # TODO@btx0424: check if we need to perform startup randomizations before resetting 
        # the simulation.
        with use_stage(self.sim.get_initial_stage()):
            self.sim.reset()
        
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

        self.sim = IsaacSimAdapter(self.sim)
        self.scene = IsaacSceneAdapter(self.scene)
        self.terrain_type = self.scene.terrain.cfg.terrain_type


class SimpleEnvMujoco(_EnvBase):
    """MuJoCo backend implementation."""
    
    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        init_state = self.command_manager.sample_init(env_ids)
        if isinstance(init_state, torch.Tensor):
            init_state = {"robot": init_state}
        for key, value in init_state.items():
            self.scene[key].write_root_state_to_sim(
                value, 
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
        
        if (objects := self.cfg.get("objects", [])):
            raise NotImplementedError("Objects are not supported in the MuJoCo backend yet.")
        
        scene = MJScene(SceneCfg())
        sim = MJSim(scene)
        self.scene = MujocoSceneAdapter(scene)
        self.sim = MujocoSimAdapter(sim)
        self.terrain_type = "plane"


class SimpleEnvMjlab(_EnvBase):
    """MjLab backend implementation."""
    
    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if (init_root_state is not None) and (not self.robot.is_fixed_base):
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        self.stats[env_ids] = 0.
    
    def setup_scene(self):
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
        from mjlab.terrains import TerrainImporterCfg
        from active_adaptation.viewer import MjLabViewer

        registry = Registry.instance()
        asset_cfg = cast(AssetCfg, registry.get("asset", self.cfg.robot.name))
        # Initialize scene and simulation.
        sensors = tuple(sensor.mjlab() for sensor in asset_cfg.sensors_mjlab)
        
        if (objects := self.cfg.get("objects", [])):
            raise NotImplementedError("Objects are not supported in the MjLab backend yet.")
        
        scene_cfg = SceneCfg(
            num_envs=self.cfg.num_envs,
            env_spacing=2.5,
            entities={"robot": asset_cfg.mjlab()},
            sensors=sensors,
            terrain=TerrainImporterCfg(terrain_type="plane"),
        )
        scene = Scene(scene_cfg, device=str(self.device))
        self.sim = Simulation(
            num_envs=scene.num_envs,
            cfg=SimulationCfg(
                nconmax=50,
                njmax=300,
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
        if not self.headless:
            viewer = MjLabViewer(self)
            viewer.run_async()
        else:
            viewer = None
        self.sim = MjlabSimAdapter(self.sim, viewer)
        self.terrain_type = "plane"

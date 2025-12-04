"""Configuration classes for assets in active adaptation framework.

This module provides backend-agnostic configuration classes for defining
assets (robots, objects) that can be used across different simulation
backends (Isaac Sim, MuJoCo Lab, MuJoCo).
"""

from dataclasses import dataclass, field, MISSING
from typing import Dict, Tuple, List, Optional, Literal
from pathlib import Path

import torch
import active_adaptation as aa

# debugging
# if aa._BACKEND_SET is False:
#     aa.set_backend("mjlab")

if aa.get_backend() == "isaac":
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import (
        ArticulationCfg as _ArticulationCfg,
        RigidObjectCfg as IsaaclabRigidObjectCfg,
    )
    from isaaclab.utils import configclass
    from isaaclab.sensors import ContactSensorCfg as IsaaclabContactSensorCfg

    @configclass
    class ArticulationCfg(_ArticulationCfg):
        joint_symmetry_mapping: Optional[Dict[str, Tuple[int, str]]] = None
        spatial_symmetry_mapping: Optional[Dict[str, str]] = None
        joint_names_isaac: Optional[List[str]] = None
        joint_names_mjlab: Optional[List[str]] = None
        body_names_isaac: Optional[List[str]] = None
        body_names_mjlab: Optional[List[str]] = None

elif aa.get_backend() == "mjlab":
    import mujoco
    from mjlab.entity import EntityCfg as _EntityCfg, EntityArticulationInfoCfg
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.utils.spec_config import CollisionCfg
    from mjlab.sensor import ContactSensorCfg as MjlabContactSensorCfg, ContactMatch
    from active_adaptation.sensors.mjlab import CfrcContactSensorCfg

    @dataclass
    class EntityCfg(_EntityCfg):
        joint_symmetry_mapping: Optional[Dict[str, Tuple[int, str]]] = None
        spatial_symmetry_mapping: Optional[Dict[str, str]] = None
        joint_names_isaac: Optional[List[str]] = None
        joint_names_mjlab: Optional[List[str]] = None
        body_names_isaac: Optional[List[str]] = None
        body_names_mjlab: Optional[List[str]] = None

elif aa.get_backend() == "mujoco":
    import mujoco
    from active_adaptation.envs.mujoco import MJArticulationCfg


@dataclass(kw_only=True, frozen=True)
class InitialStateCfg:
    """Configuration for the initial state of an asset.
    
    Defines the initial position, orientation, and joint states (positions
    and velocities) for an asset when it is spawned in the simulation.
    
    Attributes:
        pos: Initial 3D position (x, y, z) in world coordinates. Defaults to (0, 0, 0).
        rot: Initial rotation as quaternion (w, x, y, z). Defaults to identity quaternion (1, 0, 0, 0).
        joint_pos: Dictionary mapping joint name patterns to initial joint positions.
            Supports regex patterns. Defaults to {".*": 0.0} (all joints at 0).
        joint_vel: Dictionary mapping joint name patterns to initial joint velocities.
            Supports regex patterns. Defaults to {".*": 0.0} (all joints at 0 velocity).
    """
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    joint_pos: Dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    joint_vel: Dict[str, float] = field(default_factory=lambda: {".*": 0.0})

    def isaaclab(self):
        """Convert to Isaac Sim initial state configuration.
        
        Returns:
            ArticulationCfg.InitialStateCfg: Isaac Sim compatible initial state configuration.
        """
        return ArticulationCfg.InitialStateCfg(
            pos=self.pos,
            rot=self.rot,
            joint_pos=self.joint_pos,
            joint_vel=self.joint_vel,
        )
    
    def mjlab(self):
        """Convert to MuJoCo Lab initial state configuration.
        
        Returns:
            EntityCfg.InitialStateCfg: MuJoCo Lab compatible initial state configuration.
        """
        return EntityCfg.InitialStateCfg(
            pos=self.pos,
            rot=self.rot,
            joint_pos=self.joint_pos,
            joint_vel=self.joint_vel,
        )


@dataclass(kw_only=True, frozen=True)
class ActuatorCfg:
    """Configuration for joint actuators.
    
    Defines the properties and limits for actuators controlling joints.
    Supports both individual joint specifications and pattern-based matching.
    
    Attributes:
        joint_names_expr: Joint name pattern(s) to match. Can be a regex string
            or list of strings. Defaults to ".*" (all joints).
        effort_limit: Dictionary mapping joint name patterns to maximum effort/torque limits.
            Required field.
        velocity_limit: Dictionary mapping joint name patterns to maximum velocity limits.
            Required field.
        stiffness: Dictionary mapping joint name patterns to stiffness values.
            Required field.
        damping: Dictionary mapping joint name patterns to damping values.
            Required field.
        friction: Dictionary mapping joint name patterns to friction coefficients.
            Required field.
        armature: Dictionary mapping joint name patterns to armature values.
            Required field. Note: Not used in mjlab backend.
    """
    joint_names_expr: str | List[str] = ".*"
    effort_limit: Dict[str, float] = MISSING
    velocity_limit: Dict[str, float] = MISSING
    stiffness: Dict[str, float] = MISSING
    damping: Dict[str, float] = MISSING
    friction: Dict[str, float] = MISSING
    armature: Dict[str, float] = MISSING

    def isaaclab(self):
        """Convert to Isaac Sim actuator configuration.
        
        Returns:
            ImplicitActuatorCfg: Isaac Sim compatible actuator configuration.
        """
        return ImplicitActuatorCfg(
            joint_names_expr=self.joint_names_expr,
            effort_limit_sim=self.effort_limit,
            velocity_limit_sim=self.velocity_limit,
            stiffness=self.stiffness,
            damping=self.damping,
            friction=self.friction,
            armature=self.armature,
        )
    
    def mjlab(self):
        """Convert to MuJoCo Lab actuator configuration.
                
        Returns:
            BuiltinPositionActuatorCfg: MuJoCo Lab compatible actuator configuration.
        """
        return BuiltinPositionActuatorCfg(
            joint_names_expr=self.joint_names_expr,
            effort_limit=self.effort_limit,
            stiffness=self.stiffness,
            damping=self.damping,
            frictionloss=self.friction,
            armature=self.armature,
        )


@dataclass(kw_only=True, frozen=True)
class ContactSensorCfg:
    """
    Configuration for ContactSensors.
    """
    name: str = MISSING
    primary: str = MISSING
    # for isaaclab, secondary is a list of strings
    secondary: str | Tuple[str, ...] = MISSING

    track_air_time: bool = False

    # isaaclab specific
    history_length: int = 1

    # mjlab specific
    num_slots: int = 1
    fields: Tuple[str, ...] = ("found", "force")
    reduce: Literal["none", "mindist", "netforce"] = "maxforce"

    def isaaclab(self):
        return IsaaclabContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/" + f"Robot/{self.primary}",
            track_air_time=self.track_air_time,
            filter_prim_paths_expr=self.secondary,
            history_length=self.history_length,
        )

    def mjlab(self):
        return CfrcContactSensorCfg(
            name=self.name,
            entity="robot",
            track_air_time=self.track_air_time,
        )
        return MjlabContactSensorCfg(
            name=self.name,
            primary=ContactMatch(mode="subtree", pattern=self.primary, entity="robot"),
            secondary=ContactMatch(**self.secondary),
            fields=self.fields,
            reduce=self.reduce,
            num_slots=self.num_slots,
            track_air_time=self.track_air_time,
        )


@dataclass(kw_only=True, frozen=True)
class AssetCfg:
    """Configuration for a complete asset (robot, object, etc.).
    
    Defines all properties needed to spawn and configure an asset in the simulation,
    including model paths, initial state, actuators, and collision settings.
    
    Attributes:
        mjcf_path: Path to the MuJoCo XML/MJCF model file. Required field.
        usd_path: Path to the USD (Universal Scene Description) model file. Required field.
        init_state: Initial state configuration for the asset. Required field.
        actuators: Dictionary mapping actuator names to their configurations. Required field.
        self_collisions: Whether to enable self-collisions for the asset. Defaults to True.
        joint_symmetry_mapping: Optional dictionary mapping joint names to symmetry information.
            Format: {joint_name: (symmetry_group_id, symmetric_joint_name)}. Defaults to None.
        spatial_symmetry_mapping: Optional dictionary mapping spatial elements for symmetry.
            Format: {element_name: symmetric_element_name}. Defaults to None.
    """
    
    mjcf_path: str | Path = MISSING
    usd_path: str | Path = MISSING
    init_state: InitialStateCfg = MISSING
    key_frames: Optional[Dict[str, InitialStateCfg]] = None
    actuators: Dict[str, ActuatorCfg] = MISSING

    sensors_isaaclab: List[ContactSensorCfg] = field(default_factory=list)
    sensors_mjlab: List[ContactSensorCfg] = field(default_factory=list)
    
    joint_names_isaac: Optional[List[str]] = None
    joint_names_mjlab: Optional[List[str]] = None
    body_names_isaac: Optional[List[str]] = None
    body_names_mjlab: Optional[List[str]] = None

    self_collisions: bool = True

    joint_symmetry_mapping: Optional[Dict[str, Tuple[int, str]]] = None
    spatial_symmetry_mapping: Optional[Dict[str, str]] = None

    # def __post_init__(self):
    #     if self.mjcf_path is not MISSING:
    #         mjcf_path = Path(self.mjcf_path)
    #         assert mjcf_path.exists(), f"MJCF file not found: {mjcf_path}"
    #     if self.usd_path is not MISSING:
    #         usd_path = Path(self.usd_path)
    #         assert usd_path.exists(), f"USD file not found: {usd_path}"

    def isaaclab(self):
        """Convert to Isaac Sim asset configuration.
        
        Creates an Isaac Sim ArticulationCfg with appropriate physics properties,
        collision settings, and actuator configurations. Uses the USD file path
        for spawning the asset.
        
        Returns:
            ArticulationCfg: Isaac Sim compatible asset configuration with:
                - USD file spawning configuration
                - Rigid body properties (damping, velocity limits)
                - Articulation properties (self-collisions, solver settings)
                - Collision properties (contact/rest offsets)
                - Initial state and actuator configurations
        """
        if len(self.actuators) > 1:
            joint_names_expr = ""
            effort_limit = {}
            velocity_limit = {}
            stiffness = {}
            damping = {}
            friction = {}
            armature = {}
            
            def parse_cfg(expr, cfg):
                if isinstance(cfg, float):
                    return {expr: cfg}
                else:
                    return cfg
            
            # merge all actuator configurations into a single implicit actuator configuration
            for _, actuator in self.actuators.items():
                expr = actuator.joint_names_expr
                if not isinstance(expr, str):
                    expr = "|".join(expr)
                joint_names_expr += f"({expr})|"
                effort_limit.update(parse_cfg(expr, actuator.effort_limit))
                velocity_limit.update(parse_cfg(expr, actuator.velocity_limit))
                stiffness.update(parse_cfg(expr, actuator.stiffness))
                damping.update(parse_cfg(expr, actuator.damping))
                friction.update(parse_cfg(expr, actuator.friction))
                armature.update(parse_cfg(expr, actuator.armature))
            actuators = {
                "all": ImplicitActuatorCfg(
                    joint_names_expr=joint_names_expr,
                    effort_limit_sim=effort_limit,
                    velocity_limit_sim=velocity_limit,
                    stiffness=stiffness,
                    damping=damping,
                    friction=friction,
                    armature=armature,
                )
            }
        else:
            actuators = {
                name: actuator.isaaclab() for name, actuator in self.actuators.items()
            }
        
        return ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(self.usd_path),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=self.self_collisions,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.02,
                    rest_offset=0.0,
                ),
            ),
            init_state=self.init_state.isaaclab(),
            actuators=actuators,
            soft_joint_pos_limit_factor=0.9,
            joint_symmetry_mapping=self.joint_symmetry_mapping,
            spatial_symmetry_mapping=self.spatial_symmetry_mapping,
            joint_names_isaac=self.joint_names_isaac,
            joint_names_mjlab=self.joint_names_mjlab,
            body_names_isaac=self.body_names_isaac,
            body_names_mjlab=self.body_names_mjlab,
        )
    
    def mujoco(self):
        joint_names_expr = ""
        effort_limit = {}
        velocity_limit = {}
        stiffness = {}
        damping = {}
        friction = {}
        armature = {}
        
        def parse_cfg(expr, cfg):
            if isinstance(cfg, float):
                return {expr: cfg}
            else:
                return cfg
        
        # merge all actuator configurations into a single implicit actuator configuration
        for _, actuator in self.actuators.items():
            expr = actuator.joint_names_expr
            if not isinstance(expr, str):
                expr = "|".join(expr)
            joint_names_expr += f"({expr})|"
            effort_limit.update(parse_cfg(expr, actuator.effort_limit))
            velocity_limit.update(parse_cfg(expr, actuator.velocity_limit))
            stiffness.update(parse_cfg(expr, actuator.stiffness))
            damping.update(parse_cfg(expr, actuator.damping))
            friction.update(parse_cfg(expr, actuator.friction))
            armature.update(parse_cfg(expr, actuator.armature))
        
        return MJArticulationCfg(
            mjcf_path=str(self.mjcf_path),
            init_state={
                "pos": self.init_state.pos,
                "rot": self.init_state.rot,
                "joint_pos": self.init_state.joint_pos,
                "joint_vel": self.init_state.joint_vel,
            },
            actuators={
                "all": {
                    "joint_names_expr": joint_names_expr,
                    # "effort_limit_sim": effort_limit, # TODO: add effort limit
                    # "velocity_limit_sim": velocity_limit, # TODO: add velocity limit
                    "stiffness": stiffness,
                    "damping": damping,
                    "friction": friction,
                    "armature": armature,
                }
            },
            body_names_isaac=self.body_names_isaac,
            joint_names_isaac=self.joint_names_isaac,
            joint_symmetry_mapping=self.joint_symmetry_mapping,
            spatial_symmetry_mapping=self.spatial_symmetry_mapping,
        )

    def mjlab(self):
        """Convert to MuJoCo Lab asset configuration.
        
        Creates a MuJoCo Lab EntityCfg with initial state and actuator configurations.
        Uses the MJCF file path (specified via mjcf_path) for loading the model.
        
        Returns:
            EntityCfg: MuJoCo Lab compatible asset configuration with:
                - Initial state configuration
                - Articulation info with actuator configurations
                - Empty collisions tuple (collisions handled by MJCF)
        """
        if self.self_collisions:
            collision_cfg = CollisionCfg(
                geom_names_expr=(".*_collision",),
            )
        else:
            collision_cfg = CollisionCfg(
                geom_names_expr=(".*_collision",),
                contype=0,
                conaffinity=1,
            )
        
        spec = mujoco.MjSpec.from_file(str(self.mjcf_path))

        return EntityCfg(
            init_state=self.init_state.mjlab(),
            spec_fn=lambda: spec,
            articulation=EntityArticulationInfoCfg(
                actuators=(
                    actuator.mjlab()
                    for actuator in self.actuators.values()
                ),
                soft_joint_pos_limit_factor=0.9,
            ),
            collisions=(collision_cfg,),
            joint_symmetry_mapping=self.joint_symmetry_mapping,
            spatial_symmetry_mapping=self.spatial_symmetry_mapping,
            joint_names_isaac=self.joint_names_isaac,
            joint_names_mjlab=self.joint_names_mjlab,
            body_names_isaac=self.body_names_isaac,
            body_names_mjlab=self.body_names_mjlab,
        )


@dataclass(kw_only=True, frozen=True)
class RigidObjectCfg:
    
    usd_path: str | Path = MISSING
    activate_contact_sensors: bool = True
    disable_gravity: bool = False

    def isaaclab(self):
        return IsaaclabRigidObjectCfg(
            spawn=sim_utils.UsdFileCfg(
                scale=(1.0, 1.0, 1.0),
                usd_path=str(self.usd_path),
                activate_contact_sensors=self.activate_contact_sensors,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=self.disable_gravity,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=10.0,
                    # enable_gyroscopic_forces=True,
                ),
            )
        )
    
    def mujoco(self):
        raise NotImplementedError("MuJoCo backend does not support rigid objects")

    def mjlab(self):
        raise NotImplementedError("MuJoCo Lab backend does not support rigid objects")


def get_input_joint_indexing(
    input_order: Literal["isaac", "mujoco", "mjlab"],
    asset_cfg: AssetCfg,
    target_joint_names: List[str],
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[str]]:
    if input_order == aa.get_backend() or input_order == "mujoco":
        # aa's mujoco backend uses the same joint order as isaaclab
        return slice(None), target_joint_names
    if input_order == "isaac":
        source_joint_names = [name for name in asset_cfg.joint_names_isaac if name in target_joint_names]
    elif input_order == "mjlab":
        source_joint_names = [name for name in asset_cfg.joint_names_mjlab if name in target_joint_names]
    else:
        raise ValueError(f"Invalid input_order: {input_order}")
    if not len(source_joint_names) == len(target_joint_names):
        raise ValueError(f"Source joint names {source_joint_names} do not match target joint names {target_joint_names}")
    indexing = [source_joint_names.index(name) for name in target_joint_names]
    return torch.tensor(indexing, device=device), source_joint_names


def get_output_joint_indexing(
    output_order: Literal["isaac", "mujoco", "mjlab"],
    asset_cfg: AssetCfg,
    source_joint_names: List[str],
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[str]]:
    if output_order == aa.get_backend() or output_order == "mujoco":
        return slice(None), source_joint_names
    if output_order == "isaac":
        target_joint_names = [name for name in asset_cfg.joint_names_isaac if name in source_joint_names]
    elif output_order == "mjlab":
        target_joint_names = [name for name in asset_cfg.joint_names_mjlab if name in source_joint_names]
    else:
        raise ValueError(f"Invalid output_order: {output_order}")
    if not len(target_joint_names) == len(source_joint_names):
        raise ValueError(f"Target joint names {target_joint_names} do not match source joint names {source_joint_names}")
    indexing = [source_joint_names.index(name) for name in target_joint_names]
    return torch.tensor(indexing, device=device), target_joint_names


def get_output_body_indexing(
    output_order: Literal["isaac", "mujoco", "mjlab"],
    asset_cfg: AssetCfg,
    source_body_names: List[str],
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[str]]:
    if output_order == aa.get_backend() or output_order == "mujoco":
        return slice(None), source_body_names
    if output_order == "isaac":
        target_body_names = [name for name in asset_cfg.body_names_isaac if name in source_body_names]
    elif output_order == "mjlab":
        target_body_names = [name for name in asset_cfg.body_names_mjlab if name in source_body_names]
    else:
        raise ValueError(f"Invalid output_order: {output_order}")
    if not len(target_body_names) == len(source_body_names):
        raise ValueError(f"Target body names {target_body_names} do not match source body names {source_body_names}")
    indexing = [source_body_names.index(name) for name in target_body_names]
    return torch.tensor(indexing, device=device), target_body_names


if __name__ == "__main__":
    from active_adaptation.assets import UNITREE_GO2_CFG
    from mjlab.entity.entity import Entity

    import mujoco.viewer as viewer
    cfg = UNITREE_GO2_CFG
    # print(cfg.isaaclab())
    print(cfg.mjlab())

    entity = Entity(cfg.mjlab())
    viewer.launch(entity.spec.compile())
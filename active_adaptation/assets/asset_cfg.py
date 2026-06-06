"""Configuration classes for assets in active adaptation framework.

This module provides backend-agnostic configuration classes for defining
assets (robots, objects) that can be used across different simulation
backends (Isaac Sim, MuJoCo Lab, MuJoCo).
"""
from __future__ import annotations

from dataclasses import dataclass, field, MISSING, asdict
from typing import Dict, Tuple, List, Optional, Literal, Sequence
from pathlib import Path

import torch
import active_adaptation as aa

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
        joint_names_simulation: Optional[List[str]] = None
        body_names_simulation: Optional[List[str]] = None

elif aa.get_backend() in ("mjlab", "motrix"):
    import mujoco
    from typing import Callable
    from mjlab.entity import EntityCfg as _EntityCfg, EntityArticulationInfoCfg
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.utils.spec_config import CollisionCfg
    from mjlab.sensor import ContactSensorCfg as MjlabContactSensorCfg, ContactMatch

    @dataclass
    class EntityCfg(_EntityCfg):
        joint_symmetry_mapping: Optional[Dict[str, Tuple[int, str]]] = None
        spatial_symmetry_mapping: Optional[Dict[str, str]] = None
        joint_names_simulation: Optional[List[str]] = None
        body_names_simulation: Optional[List[str]] = None
        motrix_mjcf_path_fn: Callable[["EntityCfg"], str] | None = None

elif aa.get_backend() == "mujoco":
    import mujoco
    from active_adaptation.envs.backends.mujoco.mujoco import MJArticulationCfg


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
        joint_names_expr: Regex pattern used to match joints. Defaults to ".*" (all joints).
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
    effort_limit: float | Dict[str, float] = MISSING
    velocity_limit: float | Dict[str, float] = MISSING
    stiffness: float | Dict[str, float] = MISSING
    damping: float | Dict[str, float] = MISSING
    friction: float | Dict[str, float] = MISSING
    armature: float | Dict[str, float] = MISSING

    def isaaclab(self):
        """Convert to Isaac Sim actuator configuration.
        
        Returns:
            ImplicitActuatorCfg: Isaac Sim compatible actuator configuration.
        """
        joint_expr = (
            "|".join(self.joint_names_expr)
            if isinstance(self.joint_names_expr, list)
            else self.joint_names_expr
        )
        return ImplicitActuatorCfg(
            joint_names_expr=joint_expr,
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
        def _assert_scalar(name: str, value):
            if not isinstance(value, (float, int)):
                raise AssertionError(
                    f"ActuatorCfg.{name} must be a scalar float/int for mjlab, got {type(value).__name__}"
                )

        _assert_scalar("effort_limit", self.effort_limit)
        _assert_scalar("velocity_limit", self.velocity_limit)
        _assert_scalar("stiffness", self.stiffness)
        _assert_scalar("damping", self.damping)
        _assert_scalar("friction", self.friction)
        _assert_scalar("armature", self.armature)

        target_names_expr = (
            tuple(self.joint_names_expr)
            if isinstance(self.joint_names_expr, list)
            else (self.joint_names_expr,)
        )
        return BuiltinPositionActuatorCfg(
            target_names_expr=target_names_expr,
            effort_limit=float(self.effort_limit),
            stiffness=float(self.stiffness),
            damping=float(self.damping),
            frictionloss=float(self.friction),
            armature=float(self.armature),
        )


@dataclass(kw_only=True, frozen=True)
class ContactSensorCfg:
    """
    Configuration for ContactSensors.
    """
    name: str = MISSING
    track_air_time: bool = False
    history_length: int = 1

    # for isaaclab, secondary is a list of strings
    primary: str = None
    secondary: str | Sequence[str] | None = None

    # for mjlab, contact match is defined by mode/pattern/entity
    # ("body" = per-body geoms; "subtree" = MuJoCo xbody, includes descendant contacts)
    primary_contact_match_mode: Literal["geom", "subtree", "body"] = None
    primary_contact_match_pattern: str = None
    primary_contact_match_entity: str | None = None
    secondary_contact_match_mode: Optional[Literal["geom", "subtree", "body"]] = None
    secondary_contact_match_pattern: Optional[str] = None
    secondary_contact_match_entity: str | None = None
    num_slots: int = 1
    fields: Tuple[str, ...] = ("found", "force")
    reduce: Literal["none", "mindist", "maxforce", "netforce"] = "maxforce"

    def isaaclab(self):
        assert self.primary is not None, "ContactSensorCfg.primary is required for isaaclab backend"
        kwargs = {
            "prim_path": "{ENV_REGEX_NS}/" + f"Robot/{self.primary}",
            "track_air_time": self.track_air_time,
            "history_length": self.history_length,
        }
        if isinstance(self.secondary, str):
            kwargs["filter_prim_paths_expr"] = [self.secondary]
        elif isinstance(self.secondary, Sequence) and len(self.secondary) > 0:
            kwargs["filter_prim_paths_expr"] = list(self.secondary)
        return IsaaclabContactSensorCfg(**kwargs)

    def mjlab(self):
        assert self.primary_contact_match_mode is not None, "ContactSensorCfg.primary_contact_match_mode is required for mjlab backend"
        assert self.primary_contact_match_pattern is not None, "ContactSensorCfg.primary_contact_match_pattern is required for mjlab backend"
        assert self.secondary_contact_match_mode is not None, "ContactSensorCfg.secondary_contact_match_mode is required for mjlab backend"
        assert self.secondary_contact_match_pattern is not None, "ContactSensorCfg.secondary_contact_match_pattern is required for mjlab backend"
        primary = ContactMatch(
            mode=self.primary_contact_match_mode,
            pattern=self.primary_contact_match_pattern,
            entity=self.primary_contact_match_entity,
        )
        secondary = ContactMatch(
            mode=self.secondary_contact_match_mode,
            pattern=self.secondary_contact_match_pattern,
            entity=self.secondary_contact_match_entity,
        )
        return MjlabContactSensorCfg(
            name=self.name,
            primary=primary,
            secondary=secondary,
            fields=self.fields,
            reduce=self.reduce,
            num_slots=self.num_slots,
            track_air_time=self.track_air_time,
            history_length=self.history_length,
        )


@dataclass(kw_only=True, frozen=True)
class AssetCfg:
    """Configuration for a complete asset (robot, object, etc.).
    
    Defines all properties needed to spawn and configure an asset in the simulation,
    including model paths, initial state, actuators, and collision settings.
    
    Attributes:
        mjcf_path: Path to the MuJoCo XML/MJCF model file. Required field.
        usd_path: Model path for Isaac Sim: ``.usd`` / ``.usda`` / ``.usdc`` (or any
            non-``.urdf`` suffix) uses USD spawn; ``.urdf`` uses URDF spawn. Required field.
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
    actuators: Dict[str, ActuatorCfg] = MISSING

    sensors_isaaclab: List[ContactSensorCfg] = field(default_factory=list)
    sensors_mjlab: List[ContactSensorCfg] = field(default_factory=list)
    
    # Isaac Sim uses breadth-first traversal to find the joints and bodies
    joint_names_simulation: Optional[List[str]] = None
    body_names_simulation: Optional[List[str]] = None

    self_collisions: bool = True
    mjlab_collisions: List[MjlabCollisionCfg] = field(default_factory=list)

    joint_symmetry_mapping: Optional[Dict[str, Tuple[int, str]]] = None
    spatial_symmetry_mapping: Optional[Dict[str, str]] = None

    # def __post_init__(self):
    #     if self.mjcf_path and self.mjcf_path is not MISSING:
    #         mjcf_path = Path(self.mjcf_path)
    #         assert mjcf_path.exists(), f"MJCF file not found: {mjcf_path}"
    #     if self.usd_path and self.usd_path is not MISSING:
    #         usd_path = Path(self.usd_path)
    #         assert usd_path.exists(), f"USD file not found: {usd_path}"

    @staticmethod
    def _as_pattern_dict(
        expr: str,
        value: float | Dict[str, float],
    ) -> Dict[str, float]:
        if isinstance(value, (float, int)):
            return {expr: float(value)}
        return value

    def _merge_actuator_dicts(self):
        joint_names_exprs = []
        effort_limit = {}
        velocity_limit = {}
        stiffness = {}
        damping = {}
        friction = {}
        armature = {}

        def _checked_update(dst: Dict[str, float], src: Dict[str, float], field: str, actuator_name: str):
            overlap = set(dst).intersection(src)
            if overlap:
                overlap_str = ", ".join(sorted(overlap))
                raise ValueError(
                    f"Duplicate actuator pattern(s) for '{field}': {overlap_str}. "
                    f"Actuator '{actuator_name}' would overwrite existing values."
                )
            dst.update(src)

        for actuator_name, actuator in self.actuators.items():
            expr = actuator.joint_names_expr
            if isinstance(expr, list):
                expr = "|".join(expr)
            joint_names_exprs.append(f"({expr})")
            _checked_update(
                effort_limit,
                self._as_pattern_dict(expr, actuator.effort_limit),
                "effort_limit",
                actuator_name,
            )
            _checked_update(
                velocity_limit,
                self._as_pattern_dict(expr, actuator.velocity_limit),
                "velocity_limit",
                actuator_name,
            )
            _checked_update(
                stiffness,
                self._as_pattern_dict(expr, actuator.stiffness),
                "stiffness",
                actuator_name,
            )
            _checked_update(
                damping,
                self._as_pattern_dict(expr, actuator.damping),
                "damping",
                actuator_name,
            )
            _checked_update(
                friction,
                self._as_pattern_dict(expr, actuator.friction),
                "friction",
                actuator_name,
            )
            _checked_update(
                armature,
                self._as_pattern_dict(expr, actuator.armature),
                "armature",
                actuator_name,
            )

        return {
            "joint_names_expr": "|".join(joint_names_exprs),
            "effort_limit": effort_limit,
            "velocity_limit": velocity_limit,
            "stiffness": stiffness,
            "damping": damping,
            "friction": friction,
            "armature": armature,
        }

    def isaaclab(self):
        """Convert to Isaac Sim asset configuration.
        
        Creates an Isaac Sim ArticulationCfg with appropriate physics properties,
        collision settings, and actuator configurations. Spawns from ``usd_path``:
        URDF if the path ends in ``.urdf``, otherwise USD (``usd_path`` may be
        ``.usd`` / ``.usda`` / ``.usdc`` or other non-URDF extensions).
        
        Returns:
            ArticulationCfg: Isaac Sim compatible asset configuration with:
                - USD or URDF file spawning configuration
                - Rigid body properties (damping, velocity limits)
                - Articulation properties (self-collisions, solver settings)
                - Collision properties (contact/rest offsets)
                - Initial state and actuator configurations
        """
        merged = self._merge_actuator_dicts()
        actuators = {
            "all": ImplicitActuatorCfg(
                joint_names_expr=merged["joint_names_expr"],
                effort_limit_sim=merged["effort_limit"],
                velocity_limit_sim=merged["velocity_limit"],
                stiffness=merged["stiffness"],
                damping=merged["damping"],
                friction=merged["friction"],
                armature=merged["armature"],
            )
        }

        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.002,
            angular_damping=0.002,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        )
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=self.self_collisions,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        )
        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        )

        asset_path = Path(self.usd_path)
        if asset_path.suffix.lower() == ".urdf":
            spawn_cfg = sim_utils.UrdfFileCfg(
                asset_path=str(self.usd_path),
                fix_base=False,
                make_instanceable=False,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
                ),
                activate_contact_sensors=True,
                rigid_props=rigid_props,
                articulation_props=articulation_props,
                collision_props=collision_props,
                replace_cylinders_with_capsules=True,
            )
        else:
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=str(self.usd_path),
                activate_contact_sensors=True,
                rigid_props=rigid_props,
                articulation_props=articulation_props,
                collision_props=collision_props,
            )

        return ArticulationCfg(
            spawn=spawn_cfg,
            init_state=self.init_state.isaaclab(),
            actuators=actuators,
            soft_joint_pos_limit_factor=0.9,
            joint_symmetry_mapping=self.joint_symmetry_mapping,
            spatial_symmetry_mapping=self.spatial_symmetry_mapping,
            joint_names_simulation=self.joint_names_simulation,
            body_names_simulation=self.body_names_simulation,
        )
    
    def mujoco(self):
        merged = self._merge_actuator_dicts()
        
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
                    "joint_names_expr": merged["joint_names_expr"],
                    # "effort_limit_sim": effort_limit, # TODO: add effort limit
                    # "velocity_limit_sim": velocity_limit, # TODO: add velocity limit
                    "stiffness": merged["stiffness"],
                    "damping": merged["damping"],
                    "friction": merged["friction"],
                    "armature": merged["armature"],
                }
            },
            body_names_simulation=self.body_names_simulation,
            joint_names_simulation=self.joint_names_simulation,
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
        collisions = tuple(
            CollisionCfg(**asdict(collision_cfg))
            for collision_cfg in self.mjlab_collisions
        )
        
        spec = mujoco.MjSpec.from_file(str(self.mjcf_path))

        return EntityCfg(
            init_state=self.init_state.mjlab(),
            spec_fn=lambda: spec,
            articulation=EntityArticulationInfoCfg(
                actuators=tuple(
                    actuator.mjlab()
                    for actuator in self.actuators.values()
                ),
                soft_joint_pos_limit_factor=0.9,
            ),
            collisions=collisions,
            joint_symmetry_mapping=self.joint_symmetry_mapping,
            spatial_symmetry_mapping=self.spatial_symmetry_mapping,
            joint_names_simulation=self.joint_names_simulation,
            body_names_simulation=self.body_names_simulation,
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


@dataclass
class MjlabCollisionCfg:
    """Configuration to modify collision properties of geoms in the MuJoCo spec.

    Supports regex pattern matching for geom names and dict-based field resolution
    for fine-grained control over collision properties.
    """

    geom_names_expr: tuple[str, ...]
    """Tuple of regex patterns to match geom names."""
    contype: int | dict[str, int] = 1
    """Collision type (int or dict mapping patterns to values). Must be non-negative."""
    conaffinity: int | dict[str, int] = 1
    """Collision affinity (int or dict mapping patterns to values). Must be
    non-negative."""
    condim: int | dict[str, int] = 3
    """Contact dimension (int or dict mapping patterns to values). Must be one
    of {1, 3, 4, 6}."""
    priority: int | dict[str, int] = 0
    """Collision priority (int or dict mapping patterns to values). Must be
    non-negative."""
    friction: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
    """Friction coefficients as tuple or dict mapping patterns to tuples."""
    solref: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
    """Solver reference parameters as tuple or dict mapping patterns to tuples."""
    solimp: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
    """Solver impedance parameters as tuple or dict mapping patterns to tuples."""
    margin: float | dict[str, float] | None = None
    """Detection margin. Contacts are generated when geom distance < margin."""
    gap: float | dict[str, float] | None = None
    """Gap for solver inclusion. Contact included when dist < margin - gap."""
    solmix: float | dict[str, float] | None = None
    """Mixing weight for blending solver parameters between geom pairs."""
    disable_other_geoms: bool = True
    """Whether to disable collision for non-matching geoms."""


def sort_names_by_preferred_order(
    matched_names: Sequence[str],
    preferred_names: Sequence[str],
) -> List[str]:
    """Return ``matched_names`` reordered to follow ``preferred_names``.

    This is used when task code resolves a subset of joints or bodies through
    regex matching but still needs the final tensor layout to respect the
    asset's canonical simulation order.
    """
    matched_names = list(matched_names)
    preferred_names = list(preferred_names)
    ordered_names = [name for name in preferred_names if name in matched_names]
    if len(ordered_names) != len(matched_names):
        missing_names = [name for name in matched_names if name not in preferred_names]
        raise ValueError(
            f"Failed to resolve names {missing_names} in preferred order."
        )
    return ordered_names


def to_simulation_joint_order(
    joint_names: Sequence[str],
    asset_cfg: AssetCfg,
) -> List[str]:
    preferred_joint_names = asset_cfg.joint_names_simulation
    if preferred_joint_names is None:
        return list(joint_names)
    return sort_names_by_preferred_order(joint_names, preferred_joint_names)


def to_simulation_body_order(
    body_names: Sequence[str],
    asset_cfg: AssetCfg,
) -> List[str]:
    preferred_body_names = asset_cfg.body_names_simulation
    if preferred_body_names is None:
        return list(body_names)
    return sort_names_by_preferred_order(body_names, preferred_body_names)


"""Configuration classes for assets in active adaptation framework."""
from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import torch

import active_adaptation as aa

if aa.get_backend() == "isaaclab":
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
    from isaaclab.assets import (
        ArticulationCfg as _ArticulationCfg,
        RigidObjectCfg as IsaaclabRigidObjectCfg,
    )
    from isaaclab.sensors import ContactSensorCfg as IsaaclabContactSensorCfg
    from isaaclab.utils import configclass

    @configclass
    class ArticulationCfg(_ArticulationCfg):
        joint_symmetry_mapping: Optional[Dict[str, Tuple[int, str]]] = None
        spatial_symmetry_mapping: Optional[Dict[str, str]] = None
        joint_names_simulation: Optional[List[str]] = None
        body_names_simulation: Optional[List[str]] = None

elif aa.get_backend() in ("mjlab", "motrix"):
    import mujoco
    from mjlab.actuator import BuiltinPdActuatorCfg, BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg as _EntityCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg as MjlabContactSensorCfg
    from mjlab.utils.spec_config import CollisionCfg

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
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    joint_pos: Dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    joint_vel: Dict[str, float] = field(default_factory=lambda: {".*": 0.0})

    def isaaclab(self):
        return ArticulationCfg.InitialStateCfg(
            pos=self.pos,
            rot=self.rot,
            joint_pos=self.joint_pos,
            joint_vel=self.joint_vel,
        )

    def mjlab(self):
        return EntityCfg.InitialStateCfg(
            pos=self.pos,
            rot=self.rot,
            joint_pos=self.joint_pos,
            joint_vel=self.joint_vel,
        )


@dataclass(kw_only=True, frozen=True)
class ActuatorCfg:
    joint_names_expr: str | List[str] = ".*"
    mode: Literal["implicit", "explicit"] = "implicit"
    effort_limit: float | Dict[str, float] = MISSING
    velocity_limit: float | Dict[str, float] = MISSING
    stiffness: float | Dict[str, float] = MISSING
    damping: float | Dict[str, float] = MISSING
    friction: float | Dict[str, float] = MISSING
    armature: float | Dict[str, float] = MISSING

    def __post_init__(self):
        if self.mode not in ("implicit", "explicit"):
            raise ValueError(
                f"ActuatorCfg.mode must be 'implicit' or 'explicit', got {self.mode!r}"
            )

    def isaaclab(self):
        joint_expr = (
            "|".join(self.joint_names_expr)
            if isinstance(self.joint_names_expr, list)
            else self.joint_names_expr
        )
        actuator_cls = (
            ImplicitActuatorCfg if self.mode == "implicit" else IdealPDActuatorCfg
        )
        kwargs = {
            "joint_names_expr": joint_expr,
            "effort_limit_sim": self.effort_limit,
            "velocity_limit_sim": self.velocity_limit,
            "stiffness": self.stiffness,
            "damping": self.damping,
            "friction": self.friction,
            "armature": self.armature,
        }
        if self.mode == "explicit":
            kwargs["effort_limit"] = self.effort_limit
            kwargs["velocity_limit"] = self.velocity_limit
        return actuator_cls(**kwargs)

    def mjlab(self):
        def _assert_scalar(name: str, value: Any) -> float:
            if not isinstance(value, (float, int)):
                raise AssertionError(
                    f"ActuatorCfg.{name} must be scalar for mjlab, got {type(value).__name__}"
                )
            return float(value)

        target_names_expr = (
            tuple(self.joint_names_expr)
            if isinstance(self.joint_names_expr, list)
            else (self.joint_names_expr,)
        )
        kwargs = {
            "target_names_expr": target_names_expr,
            "effort_limit": _assert_scalar("effort_limit", self.effort_limit),
            "stiffness": _assert_scalar("stiffness", self.stiffness),
            "damping": _assert_scalar("damping", self.damping),
            "frictionloss": _assert_scalar("friction", self.friction),
            "armature": _assert_scalar("armature", self.armature),
        }
        actuator_cls = (
            BuiltinPositionActuatorCfg
            if self.mode == "implicit"
            else BuiltinPdActuatorCfg
        )
        return actuator_cls(**kwargs)


@dataclass(kw_only=True, frozen=True)
class ContactSensorCfg:
    name: str = MISSING
    track_air_time: bool = False
    history_length: int = 1
    primary: str | None = None
    secondary: str | Sequence[str] | None = None
    primary_contact_match_mode: Literal["geom", "subtree", "body"] | None = None
    primary_contact_match_pattern: str | None = None
    primary_contact_match_entity: str | None = None
    secondary_contact_match_mode: Literal["geom", "subtree", "body"] | None = None
    secondary_contact_match_pattern: str | None = None
    secondary_contact_match_entity: str | None = None
    num_slots: int = 1
    fields: Tuple[str, ...] = ("found", "force")
    reduce: Literal["none", "mindist", "maxforce", "netforce"] = "maxforce"

    def isaaclab(self):
        assert self.primary is not None
        kwargs = {
            "prim_path": "{ENV_REGEX_NS}/Robot/" + self.primary,
            "track_air_time": self.track_air_time,
            "history_length": self.history_length,
        }
        if isinstance(self.secondary, str):
            kwargs["filter_prim_paths_expr"] = [self.secondary]
        elif isinstance(self.secondary, Sequence) and len(self.secondary) > 0:
            kwargs["filter_prim_paths_expr"] = list(self.secondary)
        return IsaaclabContactSensorCfg(**kwargs)

    def mjlab(self):
        assert self.primary_contact_match_mode is not None
        assert self.primary_contact_match_pattern is not None
        assert self.secondary_contact_match_mode is not None
        assert self.secondary_contact_match_pattern is not None
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


@dataclass
class MjlabCollisionCfg:
    geom_names_expr: tuple[str, ...]
    priority: int | dict[str, int] = 0
    friction: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
    solref: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
    solimp: tuple[float, ...] | dict[str, tuple[float, ...]] | None = None
    margin: float | dict[str, float] | None = None
    gap: float | dict[str, float] | None = None
    solmix: float | dict[str, float] | None = None
    disable_other_geoms: bool = True


@dataclass(kw_only=True, frozen=True)
class AssetCfg:
    mjcf_path: str | Path = MISSING
    usd_path: str | Path = MISSING
    init_state: InitialStateCfg = MISSING
    actuators: Dict[str, ActuatorCfg] = MISSING
    sensors_isaaclab: List[ContactSensorCfg] = field(default_factory=list)
    sensors_mjlab: List[ContactSensorCfg] = field(default_factory=list)
    joint_names_simulation: Optional[List[str]] = None
    body_names_simulation: Optional[List[str]] = None
    self_collisions: bool = True
    mjlab_collisions: List[MjlabCollisionCfg] = field(default_factory=list)
    joint_symmetry_mapping: Optional[Dict[str, Tuple[int, str]]] = None
    spatial_symmetry_mapping: Optional[Dict[str, str]] = None

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

        def _checked_update(
            dst: Dict[str, float],
            src: Dict[str, float],
            field_name: str,
            actuator_name: str,
        ):
            overlap = set(dst).intersection(src)
            if overlap:
                overlap_str = ", ".join(sorted(overlap))
                raise ValueError(
                    f"Duplicate actuator pattern(s) for '{field_name}': {overlap_str}. "
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
        actuators = {
            actuator_name: actuator.isaaclab()
            for actuator_name, actuator in self.actuators.items()
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
                replace_cylinders_with_capsules=True,
                self_collision=self.self_collisions,
                make_instanceable=False,
                force_usd_conversion=True,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0,
                        damping=0,
                    )
                ),
                activate_contact_sensors=True,
                rigid_props=rigid_props,
                articulation_props=articulation_props,
                collision_props=collision_props,
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
        collisions = []
        for collision_cfg in self.mjlab_collisions:
            fields = asdict(collision_cfg)
            fields["contype"] = 1 if self.self_collisions else 0
            fields["conaffinity"] = 1
            collisions.append(CollisionCfg(**fields))
        spec = mujoco.MjSpec.from_file(str(self.mjcf_path))
        return EntityCfg(
            init_state=self.init_state.mjlab(),
            spec_fn=lambda: spec,
            articulation=EntityArticulationInfoCfg(
                actuators=tuple(actuator.mjlab() for actuator in self.actuators.values()),
                soft_joint_pos_limit_factor=0.9,
            ),
            collisions=tuple(collisions),
            joint_symmetry_mapping=self.joint_symmetry_mapping,
            spatial_symmetry_mapping=self.spatial_symmetry_mapping,
            joint_names_simulation=self.joint_names_simulation,
            body_names_simulation=self.body_names_simulation,
        )

    def to_asset_spec(self, backend: str) -> "AssetSpec":
        if backend == "isaaclab":
            sensors = {
                sensor.name: sensor.isaaclab() for sensor in self.sensors_isaaclab
            }
            return AssetSpec(config=self.isaaclab(), sensors=sensors)
        if backend == "mjlab":
            sensors = tuple(sensor.mjlab() for sensor in self.sensors_mjlab)
            return AssetSpec(config=self.mjlab(), sensors=sensors)
        if backend == "mujoco":
            return AssetSpec(config=self.mujoco(), sensors=())
        raise ValueError(f"Unsupported backend for AssetCfg: {backend}")


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
                ),
            )
        )


@dataclass
class AssetSpec:
    config: Any
    sensors: Any = ()
    adaptations: Sequence[Any] = ()
    """Composable :class:`~active_adaptation.envs.robots.adaptation.RobotAdaptation` instances."""
    wrapper: Optional[Any] = None
    """Deprecated alias for a single adaptation; prefer :attr:`adaptations`."""

    def with_wrapper(self, wrapper: Any) -> "AssetSpec":
        """Deprecated: append ``wrapper`` to :attr:`adaptations`."""
        self.wrapper = wrapper
        return self

    def with_adaptations(self, *adaptations: Any) -> "AssetSpec":
        self.adaptations = tuple(self.adaptations) + tuple(adaptations)
        return self

    def iter_adaptations(self) -> list[Any]:
        items = list(self.adaptations or ())
        if self.wrapper is not None:
            items.append(self.wrapper)
        return items


def coerce_asset_spec(asset_entry: Any, *, backend: str, **kwargs: Any) -> AssetSpec:
    result = asset_entry(backend=backend, **kwargs) if callable(asset_entry) else asset_entry
    if isinstance(result, AssetSpec):
        return result
    if isinstance(result, AssetCfg):
        return result.to_asset_spec(backend)
    return AssetSpec(config=result)


def get_input_joint_indexing(
    input_order: Literal["isaaclab", "mujoco", "mjlab", "simulation"],
    asset_cfg: AssetCfg,
    target_joint_names: List[str],
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[str]]:
    if input_order == aa.get_backend() or input_order == "mujoco":
        return slice(None), target_joint_names
    if input_order not in {"isaaclab", "mjlab", "simulation"}:
        raise ValueError(f"Invalid input_order: {input_order}")
    if asset_cfg.joint_names_simulation is None:
        raise ValueError("asset_cfg.joint_names_simulation is required")
    source_joint_names = [
        name for name in asset_cfg.joint_names_simulation if name in target_joint_names
    ]
    if len(source_joint_names) != len(target_joint_names):
        raise ValueError(
            f"Source joint names {source_joint_names} do not match target joint names {target_joint_names}"
        )
    indexing = [source_joint_names.index(name) for name in target_joint_names]
    return torch.tensor(indexing, device=device), source_joint_names


def get_output_joint_indexing(
    output_order: Literal["isaaclab", "mujoco", "mjlab", "simulation"],
    asset_cfg: AssetCfg,
    source_joint_names: List[str],
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[str]]:
    if output_order == aa.get_backend() or output_order == "mujoco":
        return slice(None), source_joint_names
    if output_order not in {"isaaclab", "mjlab", "simulation"}:
        raise ValueError(f"Invalid output_order: {output_order}")
    if asset_cfg.joint_names_simulation is None:
        raise ValueError("asset_cfg.joint_names_simulation is required")
    target_joint_names = [
        name for name in asset_cfg.joint_names_simulation if name in source_joint_names
    ]
    if len(target_joint_names) != len(source_joint_names):
        raise ValueError(
            f"Target joint names {target_joint_names} do not match source joint names {source_joint_names}"
        )
    indexing = [source_joint_names.index(name) for name in target_joint_names]
    return torch.tensor(indexing, device=device), target_joint_names


def get_output_body_indexing(
    output_order: Literal["isaaclab", "mujoco", "mjlab", "simulation"],
    asset_cfg: AssetCfg,
    source_body_names: List[str],
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[str]]:
    if output_order == aa.get_backend() or output_order == "mujoco":
        return slice(None), source_body_names
    if output_order not in {"isaaclab", "mjlab", "simulation"}:
        raise ValueError(f"Invalid output_order: {output_order}")
    if asset_cfg.body_names_simulation is None:
        raise ValueError("asset_cfg.body_names_simulation is required")
    target_body_names = [
        name for name in asset_cfg.body_names_simulation if name in source_body_names
    ]
    if len(target_body_names) != len(source_body_names):
        raise ValueError(
            f"Target body names {target_body_names} do not match source body names {source_body_names}"
        )
    indexing = [source_body_names.index(name) for name in target_body_names]
    return torch.tensor(indexing, device=device), target_body_names


def sort_names_by_preferred_order(
    matched_names: Sequence[str],
    preferred_names: Sequence[str],
) -> List[str]:
    matched_names = list(matched_names)
    preferred_names = list(preferred_names)
    ordered_names = [name for name in preferred_names if name in matched_names]
    if len(ordered_names) != len(matched_names):
        missing_names = [name for name in matched_names if name not in preferred_names]
        raise ValueError(f"Failed to resolve names {missing_names} in preferred order.")
    return ordered_names


def to_simulation_joint_order(
    joint_names: Sequence[str],
    asset_cfg: Any,
) -> List[str]:
    preferred_joint_names = asset_cfg.joint_names_simulation
    if preferred_joint_names is None:
        return list(joint_names)
    return sort_names_by_preferred_order(joint_names, preferred_joint_names)


def to_simulation_body_order(
    body_names: Sequence[str],
    asset_cfg: Any,
) -> List[str]:
    preferred_body_names = asset_cfg.body_names_simulation
    if preferred_body_names is None:
        return list(body_names)
    return sort_names_by_preferred_order(body_names, preferred_body_names)

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
    asset_cfg: "AssetCfg",
) -> List[str]:
    preferred_joint_names = asset_cfg.joint_names_simulation
    if preferred_joint_names is None:
        return list(joint_names)
    return sort_names_by_preferred_order(joint_names, preferred_joint_names)


def to_simulation_body_order(
    body_names: Sequence[str],
    asset_cfg: "AssetCfg",
) -> List[str]:
    preferred_body_names = asset_cfg.body_names_simulation
    if preferred_body_names is None:
        return list(body_names)
    return sort_names_by_preferred_order(body_names, preferred_body_names)


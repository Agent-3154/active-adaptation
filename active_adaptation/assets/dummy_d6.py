"""Fixed dummy base + 6-DoF handle with PD drives and a bar handle.

Isaac default uses a native USD ``PhysicsJoint`` D6 for **rotation**. PhysX
articulation joints allow only one linear DoF *or* up to three angular DoFs, so
xyz is still a prismatic chain and the D6 is spherical (DoFs ``d6:0/1/2``)::

    dummy ─d6_tx→ link_tx ─d6_ty→ link_ty ─d6_tz→ link_tz ─d6→ handle

``joint_model="serial"`` keeps the historical 6 one-DoF chain::

    dummy ─d6_tx→ link_tx ─d6_ty→ link_ty ─d6_tz→ link_tz
          ─d6_rx→ link_rx ─d6_ry→ link_ry ─d6_rz→ handle

mjlab has no USD D6, so it always uses the serial MjSpec.

``dummy`` is a fixed / mocap root (Isaac ``fix_root_link``, mjlab no freejoint).
The handle is a capsule along ``handle_axis`` (default **+X**); GraspPose is
the bar midpoint with approach perpendicular to that axis.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence

Axis = Literal["X", "Y", "Z"]
JointModel = Literal["d6", "serial"]

from active_adaptation.assets._procedural import (
    Backend,
    _DEFAULT_POS,
    _DEFAULT_ROT,
    _add_capsule_leg,
    _as_float_tuple,
    _rgba,
    _usd_add_default_transform_,
    _usd_apply_geom_rgba,
    _usd_create_capsule,
    _usd_create_prismatic_joint,
    _usd_from_mjspec_articulated,
    registry,
)

D6_JOINT_NAMES: tuple[str, ...] = (
    "d6_tx",
    "d6_ty",
    "d6_tz",
    "d6_rx",
    "d6_ry",
    "d6_rz",
)
D6_NATIVE_JOINT_NAMES: tuple[str, ...] = (
    "d6_tx",
    "d6_ty",
    "d6_tz",
    "d6:0",
    "d6:1",
    "d6:2",
)
D6_BODY_NAMES: tuple[str, ...] = (
    "dummy",
    "link_tx",
    "link_ty",
    "link_tz",
    "link_rx",
    "link_ry",
    "handle",
)
D6_NATIVE_BODY_NAMES: tuple[str, ...] = (
    "dummy",
    "link_tx",
    "link_ty",
    "link_tz",
    "handle",
)
D6_INIT_JOINT_POS: dict[str, float] = {name: 0.0 for name in D6_JOINT_NAMES}
D6_NATIVE_INIT_JOINT_POS: dict[str, float] = {name: 0.0 for name in D6_NATIVE_JOINT_NAMES}

_SLIDE_AXES = (
    ("d6_tx", "link_tx", (1.0, 0.0, 0.0)),
    ("d6_ty", "link_ty", (0.0, 1.0, 0.0)),
    ("d6_tz", "link_tz", (0.0, 0.0, 1.0)),
)
_HINGE_AXES = (
    ("d6_rx", "link_rx", (1.0, 0.0, 0.0)),
    ("d6_ry", "link_ry", (0.0, 1.0, 0.0)),
    ("d6_rz", "handle", (0.0, 0.0, 1.0)),
)
_D6_USD_AXES: tuple[str, ...] = (
    "transX",
    "transY",
    "transZ",
    "rotX",
    "rotY",
    "rotZ",
)
_HANDLE_AXIS_VEC: dict[str, tuple[float, float, float]] = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}
_HANDLE_APPROACH_DIRS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "X": ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
    "Y": ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    "Z": ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
}

_NATIVE_SLIDE_ALIASES: tuple[tuple[str, ...], ...] = (
    D6_JOINT_NAMES[:3],
    D6_NATIVE_JOINT_NAMES[:3],
)
_NATIVE_HINGE_ALIASES: tuple[tuple[str, ...], ...] = (
    D6_JOINT_NAMES[3:],
    D6_NATIVE_JOINT_NAMES[3:],
)


def _parse_handle_axis(axis: str | Axis) -> str:
    key = str(axis).upper()
    if key not in _HANDLE_AXIS_VEC:
        raise ValueError(f"handle_axis must be 'X', 'Y', or 'Z', got {axis!r}")
    return key


def _parse_joint_model(joint_model: str | JointModel) -> str:
    key = str(joint_model).lower()
    if key not in ("d6", "serial"):
        raise ValueError(f"joint_model must be 'd6' or 'serial', got {joint_model!r}")
    return key


def _handle_fromto(axis: str, half: float) -> list[float]:
    vec = _HANDLE_AXIS_VEC[axis]
    return [-half * vec[0], -half * vec[1], -half * vec[2], half * vec[0], half * vec[1], half * vec[2]]


def resolve_d6_slide_hinge_names(
    joint_names: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Map an articulation's joint names to (slide xyz, hinge rpy) D6 axes."""
    present = set(joint_names)
    for slide, hinge in zip(_NATIVE_SLIDE_ALIASES, _NATIVE_HINGE_ALIASES):
        if set(slide) <= present and set(hinge) <= present:
            return slide, hinge
    raise ValueError(
        "Could not resolve dummy_d6 slide/hinge joints from "
        f"{list(joint_names)}; expected serial {D6_JOINT_NAMES} or native "
        f"{D6_NATIVE_JOINT_NAMES}"
    )


def d6_hinge_pos_from_quat(
    quat: torch.Tensor,
    hinge_names: Sequence[str],
) -> torch.Tensor:
    """Map a world/parent quaternion to dummy_d6 rotational joint coordinates.

    Native PhysX spherical D6 (``d6:0/1/2``) uses the exponential map
    ``PxExp(twist, swing1, swing2)``. Serial ``d6_rx/ry/rz`` is intrinsic XYZ
    (``Rx Ry Rz``).
    """
    from active_adaptation.utils.math import axis_angle_from_quat, matrix_from_quat

    names = tuple(hinge_names)
    if names == D6_NATIVE_JOINT_NAMES[3:]:
        return axis_angle_from_quat(quat)

    if names != D6_JOINT_NAMES[3:]:
        raise ValueError(f"Unknown dummy_d6 hinge names {names}")

    mat = matrix_from_quat(quat)
    pitch = torch.asin(mat[..., 0, 2].clamp(-1.0, 1.0))
    roll = torch.atan2(-mat[..., 1, 2], mat[..., 2, 2])
    yaw = torch.atan2(-mat[..., 0, 1], mat[..., 0, 0])
    return torch.stack([roll, pitch, yaw], dim=-1)


def build_dummy_d6_spec(
    *,
    handle_length: float = 0.16,
    handle_radius: float = 0.022,
    handle_mass: float = 0.15,
    handle_axis: str | Axis = "X",
    slide_range: Sequence[float] = (-0.12, 0.12),
    hinge_range: Sequence[float] = (-3.1416, 3.1416),
    rgba: Sequence[float] = (0.42, 0.55, 0.48, 1.0),
    disable_gravity: bool = False,
):
    """Fixed ``dummy`` + serial D6 + capsule ``handle`` at the origin.

    ``handle_axis`` is the capsule long axis in the handle body frame
    (``X`` / ``Y`` / ``Z``). ``disable_gravity`` sets MuJoCo ``gravcomp=1``
    on every link (Isaac uses rigid-body ``disable_gravity`` instead).
    """
    import mujoco

    slide_lo, slide_hi = _as_float_tuple(slide_range, 2)
    hinge_lo, hinge_hi = _as_float_tuple(hinge_range, 2)
    rgba_t = _rgba(rgba)
    handle_key = _parse_handle_axis(handle_axis)
    half = 0.5 * float(handle_length)
    gravcomp = 1.0 if disable_gravity else 0.0

    spec = mujoco.MjSpec()
    dummy = spec.worldbody.add_body(name="dummy")
    dummy.mass = 5.0
    dummy.inertia = [0.05, 0.05, 0.05]
    dummy.gravcomp = gravcomp

    parent = dummy
    for joint_name, body_name, joint_axis in _SLIDE_AXES:
        child = parent.add_body(name=body_name, pos=(0.0, 0.0, 0.0))
        child.mass = 0.02
        child.inertia = [1e-4, 1e-4, 1e-4]
        child.gravcomp = gravcomp
        joint = child.add_joint(
            name=joint_name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=list(joint_axis),
        )
        joint.range = [slide_lo, slide_hi]
        parent = child

    for joint_name, body_name, joint_axis in _HINGE_AXES:
        child = parent.add_body(name=body_name, pos=(0.0, 0.0, 0.0))
        if body_name == "handle":
            child.mass = float(handle_mass)
            child.inertia = [1e-3, 1e-3, 1e-3]
        else:
            child.mass = 0.02
            child.inertia = [1e-4, 1e-4, 1e-4]
        child.gravcomp = gravcomp
        joint = child.add_joint(
            name=joint_name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=list(joint_axis),
        )
        joint.range = [hinge_lo, hinge_hi]
        parent = child

    _add_capsule_leg(
        parent,
        name="handle_collision",
        radius=float(handle_radius),
        fromto=_handle_fromto(handle_key, half),
        rgba=rgba_t,
    )
    return spec


def _usd_set_body_mass(prim, mass: float, inertia: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdPhysics

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))


_SPHERICAL_LIMIT_RAD = math.pi * (1.0 - 1e-6)


def _clamp_spherical_angle_limit(value: float) -> float:
    """PhysX spherical / D6 rot limits must lie in ``[-π, π]`` (radians)."""
    return float(min(_SPHERICAL_LIMIT_RAD, max(-_SPHERICAL_LIMIT_RAD, value)))


def _usd_create_d6_joint(
    stage,
    path: str,
    body_0,
    body_1,
    *,
    hinge_range: tuple[float, float],
):
    """UsdPhysics D6 used as an articulation spherical joint (rotations only).

    Translational axes are locked (PhysX will not expose them). USD angular
    LimitAPI is in **degrees**; Isaac Lab converts those to radians for
    ``joint_pos_limits``. Values are clamped so PhysX spherical limits stay
    inside ``[-π, π]``.
    """
    from pxr import Gf, UsdPhysics

    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None

    hinge_lo = _clamp_spherical_angle_limit(float(hinge_range[0]))
    hinge_hi = _clamp_spherical_angle_limit(float(hinge_range[1]))
    if hinge_lo >= hinge_hi:
        hinge_lo, hinge_hi = -_SPHERICAL_LIMIT_RAD, _SPHERICAL_LIMIT_RAD
    hinge_lo_deg = hinge_lo * 180.0 / math.pi
    hinge_hi_deg = hinge_hi * 180.0 / math.pi

    joint = UsdPhysics.Joint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body_0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body_1.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr(False)

    prim = joint.GetPrim()
    if PhysxSchema is not None and not PhysxSchema.PhysxJointAPI(prim):
        PhysxSchema.PhysxJointAPI.Apply(prim)

    for axis in _D6_USD_AXES:
        limit_api = UsdPhysics.LimitAPI.Apply(prim, axis)
        if axis.startswith("trans"):
            # Locked: low == high. Do not drive locked axes.
            limit_api.CreateLowAttr(0.0)
            limit_api.CreateHighAttr(0.0)
            continue
        limit_api.CreateLowAttr(hinge_lo_deg)
        limit_api.CreateHighAttr(hinge_hi_deg)
        if not UsdPhysics.DriveAPI(prim, axis):
            drive = UsdPhysics.DriveAPI.Apply(prim, axis)
            drive.CreateTypeAttr("force")
    return joint


def _usd_spawn_native_d6(
    stage,
    prim_path: str,
    *,
    handle_length: float,
    handle_radius: float,
    handle_mass: float,
    handle_axis: str,
    slide_range: tuple[float, float],
    hinge_range: tuple[float, float],
    rgba: tuple[float, float, float, float],
):
    from pxr import UsdGeom, UsdPhysics

    slide_lo, slide_hi = (float(slide_range[0]), float(slide_range[1]))
    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    _usd_add_default_transform_(root)

    dummy = UsdGeom.Xform.Define(stage, f"{prim_path}/dummy").GetPrim()
    _usd_add_default_transform_(dummy)
    UsdPhysics.RigidBodyAPI.Apply(dummy)
    _usd_set_body_mass(dummy, 5.0, (0.05, 0.05, 0.05))

    parent = dummy
    for joint_name, body_name, _axis in _SLIDE_AXES:
        axis = {"d6_tx": "X", "d6_ty": "Y", "d6_tz": "Z"}[joint_name]
        child = UsdGeom.Xform.Define(stage, f"{prim_path}/{body_name}").GetPrim()
        _usd_add_default_transform_(child)
        UsdPhysics.RigidBodyAPI.Apply(child)
        _usd_set_body_mass(child, 0.02, (1e-4, 1e-4, 1e-4))
        usd_joint = _usd_create_prismatic_joint(
            stage, f"{parent.GetPath()}/{joint_name}", parent, child, axis
        )
        usd_joint.CreateLowerLimitAttr(slide_lo)
        usd_joint.CreateUpperLimitAttr(slide_hi)
        parent = child

    handle = UsdGeom.Xform.Define(stage, f"{prim_path}/handle").GetPrim()
    _usd_add_default_transform_(handle)
    UsdPhysics.RigidBodyAPI.Apply(handle)
    _usd_set_body_mass(handle, float(handle_mass), (1e-3, 1e-3, 1e-3))

    half = 0.5 * float(handle_length)
    geom_path = f"{prim_path}/handle/handle_collision"
    _usd_create_capsule(stage, geom_path, float(handle_radius), _handle_fromto(handle_axis, half))
    geom_prim = stage.GetPrimAtPath(geom_path)
    UsdPhysics.CollisionAPI.Apply(geom_prim)
    _usd_apply_geom_rgba(stage, geom_path, rgba)

    _usd_create_d6_joint(
        stage,
        f"{prim_path}/handle/d6",
        parent,
        handle,
        hinge_range=hinge_range,
    )
    return root


_D6_SPAWNER_CLS = None


def _get_dummy_d6_spawner_cls():
    global _D6_SPAWNER_CLS
    if _D6_SPAWNER_CLS is not None:
        return _D6_SPAWNER_CLS

    from collections.abc import Callable

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab.utils import configclass
    from pxr import Usd

    def _apply_dummy_d6_physics(stage, prim_path: str, cfg, body_names: tuple[str, ...]) -> None:
        from isaaclab.sim.utils import bind_physics_material
        from pxr import UsdPhysics

        if cfg.rigid_props is not None:
            schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)

        for body_name in body_names:
            body_prim = stage.GetPrimAtPath(f"{prim_path}/{body_name}")
            if not body_prim.IsValid():
                continue
            for child in body_prim.GetChildren():
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if cfg.collision_props is not None:
                    schemas.define_collision_properties(
                        str(child.GetPath()), cfg.collision_props, stage=stage
                    )

        if cfg.physics_material is not None:
            if not cfg.physics_material_path.startswith("/"):
                material_path = f"{prim_path}/{cfg.physics_material_path}"
            else:
                material_path = cfg.physics_material_path
            cfg.physics_material.func(material_path, cfg.physics_material)
            for body_name in body_names:
                body_prim = stage.GetPrimAtPath(f"{prim_path}/{body_name}")
                if not body_prim.IsValid():
                    continue
                for child in body_prim.GetChildren():
                    if not child.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    bind_physics_material(
                        str(child.GetPath()), material_path, stage=stage
                    )

    def _finalize_dummy_root(stage, prim_path: str, cfg, root) -> None:
        from pxr import UsdPhysics

        try:
            from pxr import PhysxSchema
        except ImportError:
            PhysxSchema = None

        dummy_path = f"{prim_path}/dummy"
        dummy_prim = stage.GetPrimAtPath(dummy_path)
        if not dummy_prim.IsValid():
            raise RuntimeError(f"Expected dummy body at {dummy_path}")
        if root.HasAPI(UsdPhysics.ArticulationRootAPI):
            root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if PhysxSchema is not None and root.HasAPI(PhysxSchema.PhysxArticulationAPI):
            root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        UsdPhysics.ArticulationRootAPI.Apply(dummy_prim)

        if cfg.articulation_props is not None:
            schemas.modify_articulation_root_properties(
                dummy_path, cfg.articulation_props
            )
        if cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_path, stage=stage)

    @clone
    def spawn_dummy_d6(
        prim_path: str,
        cfg: "ProceduralDummyD6Cfg",
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> Usd.Prim:
        del kwargs
        stage = get_current_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            raise ValueError(f"A prim already exists at path: '{prim_path}'.")

        from pxr import Gf

        use_serial = _parse_joint_model(cfg.joint_model) == "serial"
        if use_serial:
            spec = build_dummy_d6_spec(
                handle_length=cfg.handle_length,
                handle_radius=cfg.handle_radius,
                handle_mass=cfg.handle_mass,
                handle_axis=cfg.handle_axis,
                slide_range=cfg.slide_range,
                hinge_range=cfg.hinge_range,
                rgba=cfg.rgba,
            )
            root = _usd_from_mjspec_articulated(stage, prim_path, spec)
            body_names = D6_BODY_NAMES
        else:
            root = _usd_spawn_native_d6(
                stage,
                prim_path,
                handle_length=cfg.handle_length,
                handle_radius=cfg.handle_radius,
                handle_mass=cfg.handle_mass,
                handle_axis=cfg.handle_axis,
                slide_range=tuple(cfg.slide_range),
                hinge_range=tuple(cfg.hinge_range),
                rgba=cfg.rgba,
            )
            body_names = D6_NATIVE_BODY_NAMES

        if translation is not None:
            root.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*translation))
        if orientation is not None:
            root.GetAttribute("xformOp:orient").Set(Gf.Quatf(*orientation))

        _apply_dummy_d6_physics(stage, prim_path, cfg, body_names)
        _finalize_dummy_root(stage, prim_path, cfg, root)
        return root

    @configclass
    class ProceduralDummyD6Cfg(SpawnerCfg):
        func: Callable = spawn_dummy_d6
        handle_length: float = 0.16
        handle_radius: float = 0.022
        handle_mass: float = 0.15
        handle_axis: str = "X"
        joint_model: str = "d6"
        slide_range: tuple[float, float] = (-0.12, 0.12)
        hinge_range: tuple[float, float] = (-3.1416, 3.1416)
        rgba: tuple[float, float, float, float] = (0.42, 0.55, 0.48, 1.0)
        collision_props: Any = None
        physics_material_path: str = "material"
        physics_material: Any = None
        articulation_props: Any = None
        rigid_props: Any = None
        activate_contact_sensors: bool = True
        copy_from_source: bool = False

    _D6_SPAWNER_CLS = ProceduralDummyD6Cfg
    return _D6_SPAWNER_CLS


def make_dummy_d6(
    backend: Backend,
    handle_length: float = 0.16,
    handle_radius: float = 0.022,
    handle_mass: float = 0.15,
    handle_axis: str | Axis = "X",
    joint_model: str | JointModel = "d6",
    slide_range: Sequence[float] = (-0.12, 0.12),
    hinge_range: Sequence[float] = (-3.1416, 3.1416),
    stiffness: float = 40.0,
    damping: float = 4.0,
    rgba: Sequence[float] = (0.42, 0.55, 0.48, 1.0),
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    disable_gravity: bool = False,
    activate_contact_sensors: bool = True,
    attach_grasp: bool = True,
):
    """Compliant D6 handle on a dummy base. Returns ``AssetSpec``.

    ``joint_model`` is ``d6`` (Isaac native ``PhysicsJoint``) or ``serial``
    (6 one-DoF joints). mjlab always uses serial. ``handle_axis`` (``X`` /
    ``Y`` / ``Z``) is the capsule long axis in the handle body frame.
    ``disable_gravity`` turns gravity off on every link (Isaac rigid-body
    flag; mjlab ``gravcomp``).
    """
    from active_adaptation.assets.asset_cfg import AssetSpec
    from active_adaptation.envs.behaviors.grasp_pose import GraspPose

    pos_t = _as_float_tuple(pos, 3)
    rot_t = _as_float_tuple(rot, 4)
    rgba_t = _rgba(rgba)
    slide_t = _as_float_tuple(slide_range, 2)
    hinge_t = _as_float_tuple(hinge_range, 2)
    axis = _parse_handle_axis(handle_axis)
    model = _parse_joint_model(joint_model)
    use_serial = model == "serial" or backend != "isaaclab"
    joint_names = list(D6_JOINT_NAMES if use_serial else D6_NATIVE_JOINT_NAMES)
    body_names = list(D6_BODY_NAMES if use_serial else D6_NATIVE_BODY_NAMES)
    init_joint = dict(D6_INIT_JOINT_POS if use_serial else D6_NATIVE_INIT_JOINT_POS)

    if backend == "isaaclab":
        import isaaclab.sim as sim_utils
        from active_adaptation.assets.asset_cfg import (
            ArticulationCfg,
            ImplicitActuatorCfg,
        )

        ProceduralDummyD6Cfg = _get_dummy_d6_spawner_cls()
        spawn = ProceduralDummyD6Cfg(
            handle_length=float(handle_length),
            handle_radius=float(handle_radius),
            handle_mass=float(handle_mass),
            handle_axis=axis,
            joint_model="serial" if use_serial else "d6",
            slide_range=slide_t,
            hinge_range=hinge_t,
            rgba=rgba_t,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=bool(disable_gravity),
            ),
            activate_contact_sensors=activate_contact_sensors,
            copy_from_source=False,
        )
        cfg = ArticulationCfg(
            spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos_t,
                rot=rot_t,
                joint_pos=init_joint,
                joint_vel={".*": 0.0},
            ),
            actuators={
                "d6": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=50.0,
                    stiffness=float(stiffness),
                    damping=float(damping),
                    armature=0.01,
                    friction=0.0,
                ),
            },
            joint_names_simulation=joint_names,
            body_names_simulation=body_names,
        )
    elif backend == "mjlab":
        from active_adaptation.assets.asset_cfg import EntityCfg
        from mjlab.actuator import BuiltinPdActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg
        from mjlab.utils.spec_config import CollisionCfg

        def spec_fn():
            return build_dummy_d6_spec(
                handle_length=handle_length,
                handle_radius=handle_radius,
                handle_mass=handle_mass,
                handle_axis=axis,
                slide_range=slide_t,
                hinge_range=hinge_t,
                rgba=rgba_t,
                disable_gravity=bool(disable_gravity),
            )

        cfg = EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=pos_t,
                rot=rot_t,
                joint_pos=init_joint,
                joint_vel={".*": 0.0},
            ),
            spec_fn=spec_fn,
            articulation=EntityArticulationInfoCfg(
                actuators=(
                    BuiltinPdActuatorCfg(
                        target_names_expr=("d6_.*",),
                        effort_limit=50.0,
                        stiffness=float(stiffness),
                        damping=float(damping),
                        armature=0.01,
                        frictionloss=0.0,
                    ),
                ),
            ),
            collisions=(
                CollisionCfg(
                    geom_names_expr=(".*_collision",),
                    contype=1,
                    conaffinity=1,
                    condim=3,
                    priority=0,
                    solref=(0.02, 1),
                    friction=(1.0, 5e-3, 5e-4),
                ),
            ),
            joint_names_simulation=joint_names,
            body_names_simulation=body_names,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    behaviors = ()
    if attach_grasp:
        behaviors = (
            GraspPose.for_side_axis(
                axis=_HANDLE_AXIS_VEC[axis],
                approach_dirs=_HANDLE_APPROACH_DIRS[axis],
                pos=(0.0, 0.0, 0.0),
            ),
        )
    return AssetSpec(config=cfg, behaviors=behaviors)


registry.register("asset", "dummy_d6", make_dummy_d6)

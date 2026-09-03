"""Procedural primitive shapes (box, sphere, cylinder, capsule).

Factories return a bare Isaac ``RigidObjectCfg`` / ``AssetBaseCfg`` or mjlab
``EntityCfg`` (not ``AssetSpec``), matching ``dummy_objects`` / ``hoi_object``.
Size parameters use Isaac / USD conventions (full extents, full cylinder/capsule
height) and are converted to MuJoCo half-sizes internally.

By default shapes are floating rigid bodies. Pass ``collision_only=True`` for a
static collider (Isaac: CollisionAPI only via ``AssetBaseCfg``; mjlab: fixed-base
body, no freejoint — auto-wrapped as mocap).

YAML example::

    objects:
      object:
        _target_: box
        size: [0.05, 0.05, 0.05]
        mass: 0.1
      pedestal:
        _target_: box
        size: [0.1, 0.1, 0.1]
        pos: [0.0, 0.0, 0.05]
        collision_only: true
"""

from __future__ import annotations

from typing import Literal, Sequence

from active_adaptation.registry import Registry

registry = Registry.instance()

Backend = Literal["isaaclab", "mjlab"]
Axis = Literal["X", "Y", "Z"]

_DEFAULT_MASS = 0.1
_DEFAULT_RGBA = (0.72, 0.45, 0.28, 1.0)
_DEFAULT_POS = (0.0, 0.0, 0.0)
_DEFAULT_ROT = (1.0, 0.0, 0.0, 0.0)
_SQRT2_2 = 0.7071067811865476
# Rotate MuJoCo's default +Z geom axis onto +X / +Y.
_AXIS_QUAT = {
    "Z": None,
    "X": (_SQRT2_2, 0.0, _SQRT2_2, 0.0),
    "Y": (_SQRT2_2, _SQRT2_2, 0.0, 0.0),
}


def _as_float_tuple(value: Sequence[float] | float, expected: int | None = None) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        if expected is None:
            return (float(value),)
        return (float(value),) * expected
    out = tuple(float(x) for x in value)
    if expected is not None and len(out) != expected:
        raise ValueError(f"Expected {expected} values, got {len(out)}: {value}")
    return out


def _rgba(rgba: Sequence[float]) -> tuple[float, float, float, float]:
    values = _as_float_tuple(rgba)
    if len(values) == 3:
        r, g, b = values
        return (r, g, b, 1.0)
    if len(values) == 4:
        r, g, b, a = values
        return (r, g, b, a)
    raise ValueError(f"rgba must have 3 or 4 components, got {rgba}")


def _mj_geom_size(kind: str, *, size, radius: float | None, height: float | None) -> tuple[float, float, float]:
    """Isaac full-extents → MuJoCo geom ``size`` (half-extents / radius)."""
    if kind == "box":
        hx, hy, hz = _as_float_tuple(size, 3)
        return (hx * 0.5, hy * 0.5, hz * 0.5)
    if kind == "sphere":
        assert radius is not None
        return (float(radius), 0.0, 0.0)
    assert radius is not None and height is not None
    return (float(radius), float(height) * 0.5, 0.0)


def _isaac_spawn_kwargs(
    *,
    mass: float,
    rgba: tuple[float, float, float, float],
    disable_gravity: bool,
    activate_contact_sensors: bool,
    collision_only: bool,
) -> dict:
    import isaaclab.sim as sim_utils

    kw: dict = dict(
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=rgba[:3],
            opacity=rgba[3],
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    if collision_only:
        # CollisionAPI only — no RigidBodyAPI / mass / contact reporter.
        return kw
    kw.update(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=disable_gravity,
            retain_accelerations=False,
            linear_damping=0.001,
            angular_damping=0.001,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        activate_contact_sensors=activate_contact_sensors,
    )
    return kw


def _isaac_shape_cfg(kind: str, axis: str, size, radius: float | None, height: float | None, spawn_kw: dict):
    import isaaclab.sim as sim_utils

    axis = axis.upper()
    if kind == "box":
        return sim_utils.CuboidCfg(size=_as_float_tuple(size, 3), **spawn_kw)
    if kind == "sphere":
        return sim_utils.SphereCfg(radius=float(radius), **spawn_kw)
    if kind == "cylinder":
        return sim_utils.CylinderCfg(
            radius=float(radius), height=float(height), axis=axis, **spawn_kw
        )
    if kind == "capsule":
        return sim_utils.CapsuleCfg(
            radius=float(radius), height=float(height), axis=axis, **spawn_kw
        )
    raise ValueError(f"Unknown primitive kind: {kind}")


def _make_isaaclab_cfg(
    kind: str,
    *,
    size,
    radius: float | None,
    height: float | None,
    axis: str,
    mass: float,
    rgba: tuple[float, float, float, float],
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    disable_gravity: bool,
    activate_contact_sensors: bool,
    collision_only: bool,
):
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    spawn_kw = _isaac_spawn_kwargs(
        mass=mass,
        rgba=rgba,
        disable_gravity=disable_gravity,
        activate_contact_sensors=activate_contact_sensors,
        collision_only=collision_only,
    )
    spawn = _isaac_shape_cfg(kind, axis, size, radius, height, spawn_kw)

    if collision_only:
        return AssetBaseCfg(
            spawn=spawn,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        )
    return RigidObjectCfg(
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def _make_mjlab_cfg(
    kind: str,
    *,
    body_name: str | None,
    size,
    radius: float | None,
    height: float | None,
    axis: str,
    mass: float,
    rgba: tuple[float, float, float, float],
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    disable_gravity: bool,
    collision_only: bool,
):
    import mujoco
    from active_adaptation.assets.asset_cfg import EntityCfg
    from mjlab.utils.spec_config import CollisionCfg

    geom_types = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    }
    geom_type = geom_types[kind]
    geom_size = _mj_geom_size(kind, size=size, radius=radius, height=height)
    geom_quat = _AXIS_QUAT[axis.upper()]
    body_name = body_name or kind

    def spec_fn():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name=body_name)
        if collision_only:
            # Fixed-base collider; mjlab auto-wraps as mocap for per-env pose.
            geom_kw = dict(
                name=f"{body_name}_collision",
                type=geom_type,
                size=geom_size,
                rgba=rgba,
            )
        else:
            if disable_gravity:
                body.gravcomp = 1.0
            body.add_freejoint(name=f"{body_name}_joint")
            geom_kw = dict(
                name=f"{body_name}_collision",
                type=geom_type,
                size=geom_size,
                mass=mass,
                rgba=rgba,
            )
        if geom_quat is not None:
            geom_kw["quat"] = geom_quat
        body.add_geom(**geom_kw)
        return spec

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=pos, rot=rot),
        spec_fn=spec_fn,
        articulation=None,
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
    )


def _make_primitive(
    backend: Backend,
    kind: str,
    *,
    body_name: str | None = None,
    size=None,
    radius: float | None = None,
    height: float | None = None,
    axis: Axis = "Z",
    mass: float = _DEFAULT_MASS,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    disable_gravity: bool = False,
    activate_contact_sensors: bool = True,
    collision_only: bool = False,
):
    rgba_t = _rgba(rgba)
    pos_t = _as_float_tuple(pos, 3)
    rot_t = _as_float_tuple(rot, 4)
    if backend == "isaaclab":
        return _make_isaaclab_cfg(
            kind,
            size=size,
            radius=radius,
            height=height,
            axis=axis,
            mass=mass,
            rgba=rgba_t,
            pos=pos_t,
            rot=rot_t,
            disable_gravity=disable_gravity,
            activate_contact_sensors=activate_contact_sensors,
            collision_only=collision_only,
        )
    if backend == "mjlab":
        del activate_contact_sensors
        return _make_mjlab_cfg(
            kind,
            body_name=body_name,
            size=size,
            radius=radius,
            height=height,
            axis=axis,
            mass=mass,
            rgba=rgba_t,
            pos=pos_t,
            rot=rot_t,
            disable_gravity=disable_gravity,
            collision_only=collision_only,
        )
    raise ValueError(f"Invalid backend: {backend}")


def make_box(
    backend: Backend,
    name: str = "box",
    size: Sequence[float] | float = (0.05, 0.05, 0.05),
    mass: float = _DEFAULT_MASS,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    disable_gravity: bool = False,
    activate_contact_sensors: bool = True,
    collision_only: bool = False,
):
    """Cuboid. ``size`` is full (x, y, z) extents in meters.

    ``collision_only=True`` → static collider (no rigid body / no freejoint).
    """
    return _make_primitive(
        backend,
        "box",
        body_name=name,
        size=size,
        mass=mass,
        rgba=rgba,
        pos=pos,
        rot=rot,
        disable_gravity=disable_gravity,
        activate_contact_sensors=activate_contact_sensors,
        collision_only=collision_only,
    )


def make_sphere(
    backend: Backend,
    radius: float = 0.025,
    mass: float = _DEFAULT_MASS,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    disable_gravity: bool = False,
    activate_contact_sensors: bool = True,
    collision_only: bool = False,
):
    """Sphere. ``collision_only=True`` → static collider."""
    return _make_primitive(
        backend,
        "sphere",
        radius=radius,
        mass=mass,
        rgba=rgba,
        pos=pos,
        rot=rot,
        disable_gravity=disable_gravity,
        activate_contact_sensors=activate_contact_sensors,
        collision_only=collision_only,
    )


def make_cylinder(
    backend: Backend,
    radius: float = 0.025,
    height: float = 0.08,
    axis: Axis = "Z",
    mass: float = _DEFAULT_MASS,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    disable_gravity: bool = False,
    activate_contact_sensors: bool = True,
    collision_only: bool = False,
):
    """Cylinder. ``height`` is the full length along ``axis``.

    ``collision_only=True`` → static collider.
    """
    return _make_primitive(
        backend,
        "cylinder",
        radius=radius,
        height=height,
        axis=axis,
        mass=mass,
        rgba=rgba,
        pos=pos,
        rot=rot,
        disable_gravity=disable_gravity,
        activate_contact_sensors=activate_contact_sensors,
        collision_only=collision_only,
    )


def make_capsule(
    backend: Backend,
    radius: float = 0.02,
    height: float = 0.06,
    axis: Axis = "Z",
    mass: float = _DEFAULT_MASS,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    disable_gravity: bool = False,
    activate_contact_sensors: bool = True,
    collision_only: bool = False,
):
    """Capsule. ``height`` is the cylindrical section (USD convention).

    ``collision_only=True`` → static collider.
    """
    return _make_primitive(
        backend,
        "capsule",
        radius=radius,
        height=height,
        axis=axis,
        mass=mass,
        rgba=rgba,
        pos=pos,
        rot=rot,
        disable_gravity=disable_gravity,
        activate_contact_sensors=activate_contact_sensors,
        collision_only=collision_only,
    )


registry.register("asset", "capsule", make_capsule)
registry.register("asset", "cylinder", make_cylinder)
registry.register("asset", "box", make_box)
registry.register("asset", "sphere", make_sphere)

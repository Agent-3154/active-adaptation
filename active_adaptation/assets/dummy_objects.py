"""Dummy / procedural scene objects (stands, baskets, furniture).

Furniture (``table``, ``chair``) is **not articulated**: a single rigid body with
multiple collision geoms (box top / seat / back + capsule legs). Isaac builds USD
from an MjSpec the same way as ``metamorphosis.utils.usd_utils.from_mjspec``, but
applies ``RigidBodyAPI`` on the root only (no ``ArticulationRootAPI``). mjlab uses
the same MjSpec via ``EntityCfg.spec_fn``.

YAML example::

    objects:
      table:
        _target_: dummy_table
        leg_length: 0.75
        top_size: [1.0, 0.6, 0.04]
      chair:
        _target_: dummy_chair
        leg_length: 0.45

Factories return ``AssetSpec`` with a ``GraspPose`` adaptation of prescribed
poses at each leg midpoint looking at the center axis
(``env.require_adaptation("table.grasp")``).
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.registry import Registry

registry = Registry.instance()

Backend = Literal["isaaclab", "mjlab"]

_DEFAULT_RGBA = (0.55, 0.42, 0.28, 1.0)
_DEFAULT_POS = (0.0, 0.0, 0.0)
_DEFAULT_ROT = (1.0, 0.0, 0.0, 0.0)
_DEFAULT_MASS = 5.0


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


# ---------------------------------------------------------------------------
# Shared MjSpec builders (Isaac USD + mjlab EntityCfg)
# ---------------------------------------------------------------------------


def _add_capsule_leg(body, *, name: str, radius: float, fromto: list[float], rgba) -> None:
    import mujoco

    geom = body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        rgba=rgba,
    )
    geom.size = [radius, 0.0, 0.0]
    geom.fromto = fromto


def _leg_corners(half_x: float, half_y: float, inset: float) -> list[tuple[float, float]]:
    hx = max(half_x - inset, 0.0)
    hy = max(half_y - inset, 0.0)
    return [(hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy)]


def build_table_spec(
    *,
    leg_length: float = 0.75,
    top_size: Sequence[float] = (1.0, 0.6, 0.04),
    leg_radius: float = 0.025,
    leg_inset: float = 0.06,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    mass: float | None = _DEFAULT_MASS,
    body_name: str = "table",
):
    """Table: box surface + four capsule legs. Origin at floor center."""
    import mujoco

    lx, ly, lz = _as_float_tuple(top_size, 3)
    rgba_t = _rgba(rgba)
    half_x, half_y = lx * 0.5, ly * 0.5

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=body_name)
    if mass is not None:
        body.mass = float(mass)

    top_z = leg_length + lz * 0.5
    body.add_geom(
        name=f"{body_name}_top_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_x, half_y, lz * 0.5),
        pos=(0.0, 0.0, top_z),
        rgba=rgba_t,
    )

    for i, (x, y) in enumerate(_leg_corners(half_x, half_y, leg_inset)):
        _add_capsule_leg(
            body,
            name=f"{body_name}_leg{i}_collision",
            radius=leg_radius,
            fromto=[x, y, 0.0, x, y, float(leg_length)],
            rgba=rgba_t,
        )
    return spec


def build_chair_spec(
    *,
    leg_length: float = 0.45,
    seat_size: Sequence[float] = (0.42, 0.42, 0.04),
    back_height: float = 0.42,
    back_thickness: float = 0.04,
    leg_radius: float = 0.02,
    leg_inset: float = 0.04,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    mass: float | None = 8.0,
    body_name: str = "chair",
):
    """Chair: box seat + box back + four capsule legs. Origin at floor center."""
    import mujoco

    sx, sy, sz = _as_float_tuple(seat_size, 3)
    rgba_t = _rgba(rgba)
    half_x, half_y = sx * 0.5, sy * 0.5

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=body_name)
    if mass is not None:
        body.mass = float(mass)

    seat_z = leg_length + sz * 0.5
    body.add_geom(
        name=f"{body_name}_seat_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_x, half_y, sz * 0.5),
        pos=(0.0, 0.0, seat_z),
        rgba=rgba_t,
    )

    back_hz = back_height * 0.5
    back_z = leg_length + sz + back_hz
    body.add_geom(
        name=f"{body_name}_back_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_x, back_thickness * 0.5, back_hz),
        pos=(0.0, -(half_y - back_thickness * 0.5), back_z),
        rgba=rgba_t,
    )

    for i, (x, y) in enumerate(_leg_corners(half_x, half_y, leg_inset)):
        _add_capsule_leg(
            body,
            name=f"{body_name}_leg{i}_collision",
            radius=leg_radius,
            fromto=[x, y, 0.0, x, y, float(leg_length)],
            rgba=rgba_t,
        )
    return spec


# ---------------------------------------------------------------------------
# Isaac: MjSpec → single rigid USD (metamorphosis-style, no articulation)
# ---------------------------------------------------------------------------


def _usd_add_default_transform_(prim) -> None:
    from pxr import Sdf, Gf

    order = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3, False).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatf, False).Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Float3, False).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, False).Set(order)


def _usd_create_capsule(stage, path: str, radius: float, fromto):
    """Mirror ``metamorphosis.utils.usd_utils.create_capsule``."""
    import numpy as np
    from pxr import UsdGeom, Gf
    from scipy.spatial.transform import Rotation as R

    fromto = np.asarray(fromto, dtype=float)
    capsule = UsdGeom.Capsule.Define(stage, path)
    add_prim = capsule.GetPrim()
    _usd_add_default_transform_(add_prim)
    direction = fromto[3:] - fromto[:3]
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        raise ValueError(f"Degenerate capsule fromto: {fromto}")
    direction = direction / length
    axis = np.cross(direction, [0.0, 0.0, 1.0])
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        # Parallel to +Z (or -Z)
        orient = np.array([1.0, 0.0, 0.0, 0.0]) if direction[2] >= 0 else np.array([0.0, 1.0, 0.0, 0.0])
    else:
        angle = float(np.arccos(np.clip(np.dot(direction, [0.0, 0.0, 1.0]), -1.0, 1.0)))
        orient = R.from_rotvec(angle * (axis / axis_norm)).as_quat(scalar_first=True)
    translation = (fromto[:3] + fromto[3:]) * 0.5
    capsule.CreateAxisAttr("Z")
    capsule.CreateRadiusAttr(float(radius))
    capsule.CreateHeightAttr(length)
    add_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*translation))
    add_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(*orient))
    return capsule


def _usd_from_mjspec_rigid(stage, prim_path: str, spec) -> object:
    """Build a non-articulated rigid USD prim from an MjSpec (single body + geoms).

    Unlike ``metamorphosis.utils.usd_utils.from_mjspec``, this does **not** apply
    ``ArticulationRootAPI`` / joints. Rigid / mass / collision schemas are applied
    by the spawner from ``RigidObjectSpawnerCfg`` fields.
    """
    import mujoco
    import numpy as np
    from pxr import UsdGeom, Gf, UsdPhysics

    mjmodel = spec.compile()
    mjdata = mujoco.MjData(mjmodel)
    mujoco.mj_forward(mjmodel, mjdata)

    bodies = list(spec.worldbody.find_all("body"))
    if len(bodies) != 1:
        raise ValueError(
            f"Furniture MjSpec must have exactly one body (got {len(bodies)}); "
            "table/chair are non-articulated rigid objects."
        )
    mjbody = bodies[0]

    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    geom_root_path = f"{prim_path}/geometry"
    UsdGeom.Xform.Define(stage, geom_root_path)

    for i, geom in enumerate(mjbody.geoms):
        name = geom.name or f"collision_{i}"
        geom_path = f"{geom_root_path}/{name}"
        match geom.type:
            case mujoco.mjtGeom.mjGEOM_BOX:
                cube = UsdGeom.Cube.Define(stage, geom_path)
                cube.CreateSizeAttr(2.0)
                _usd_add_default_transform_(cube.GetPrim())
                cube.GetPrim().GetAttribute("xformOp:scale").Set(
                    Gf.Vec3f(float(geom.size[0]), float(geom.size[1]), float(geom.size[2]))
                )
                cube.GetPrim().GetAttribute("xformOp:translate").Set(
                    Gf.Vec3f(float(geom.pos[0]), float(geom.pos[1]), float(geom.pos[2]))
                )
            case mujoco.mjtGeom.mjGEOM_CAPSULE:
                fromto = np.array(geom.fromto, dtype=float)
                if np.allclose(fromto, 0.0):
                    # size = [radius, half-length]; axis +Z at geom.pos
                    half = float(geom.size[1])
                    pos = np.array(geom.pos, dtype=float)
                    fromto = np.array(
                        [pos[0], pos[1], pos[2] - half, pos[0], pos[1], pos[2] + half],
                        dtype=float,
                    )
                _usd_create_capsule(stage, geom_path, float(geom.size[0]), fromto)
            case _:
                raise ValueError(f"Unsupported furniture geom type: {geom.type}")
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(geom_path))

    # Root pose matches compiled body pose (usually identity at origin).
    _usd_add_default_transform_(root)
    root.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*mjdata.xpos[mjbody.id]))
    root.GetAttribute("xformOp:orient").Set(Gf.Quatf(*mjdata.xquat[mjbody.id]))
    return root


_FURNITURE_SPAWNER_CLS = None


def _get_furniture_spawner_cls():
    """Lazy Isaac ``RigidObjectSpawnerCfg`` for procedural furniture."""
    global _FURNITURE_SPAWNER_CLS
    if _FURNITURE_SPAWNER_CLS is not None:
        return _FURNITURE_SPAWNER_CLS

    from collections.abc import Callable

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab.utils import configclass
    from pxr import Usd

    @clone
    def spawn_furniture(
        prim_path: str,
        cfg: "ProceduralFurnitureCfg",
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> Usd.Prim:
        del kwargs
        stage = get_current_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            raise ValueError(f"A prim already exists at path: '{prim_path}'.")

        if cfg.kind == "table":
            spec = build_table_spec(
                leg_length=cfg.leg_length,
                top_size=cfg.top_size,
                leg_radius=cfg.leg_radius,
                leg_inset=cfg.leg_inset,
                rgba=cfg.rgba,
                mass=None,
                body_name=cfg.body_name,
            )
        elif cfg.kind == "chair":
            spec = build_chair_spec(
                leg_length=cfg.leg_length,
                seat_size=cfg.seat_size,
                back_height=cfg.back_height,
                back_thickness=cfg.back_thickness,
                leg_radius=cfg.leg_radius,
                leg_inset=cfg.leg_inset,
                rgba=cfg.rgba,
                mass=None,
                body_name=cfg.body_name,
            )
        else:
            raise ValueError(f"Unknown furniture kind: {cfg.kind}")

        root = _usd_from_mjspec_rigid(stage, prim_path, spec)

        from isaaclab.sim.utils import bind_physics_material
        from pxr import Gf, UsdPhysics

        # Apply spawn-time pose (RigidObject init_state also sets this; keep consistent).
        if translation is not None:
            root.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*translation))
        if orientation is not None:
            root.GetAttribute("xformOp:orient").Set(Gf.Quatf(*orientation))
        if cfg.collision_props is not None:
            geom_root = stage.GetPrimAtPath(f"{prim_path}/geometry")
            for child in geom_root.GetChildren():
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                schemas.define_collision_properties(str(child.GetPath()), cfg.collision_props, stage=stage)
        if cfg.physics_material is not None:
            geom_root_path = f"{prim_path}/geometry"
            # Keep the material beside geometry (not under it) so iterating
            # collision children never includes the material prim itself.
            # ``bind_physics_material`` is nested and would warn on the material.
            if not cfg.physics_material_path.startswith("/"):
                material_path = f"{prim_path}/{cfg.physics_material_path}"
            else:
                material_path = cfg.physics_material_path
            cfg.physics_material.func(material_path, cfg.physics_material)
            geom_root = stage.GetPrimAtPath(geom_root_path)
            for child in geom_root.GetChildren():
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                bind_physics_material(str(child.GetPath()), material_path, stage=stage)
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
        if cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_path, stage=stage)
        return root

    @configclass
    class ProceduralFurnitureCfg(RigidObjectSpawnerCfg):
        func: Callable = spawn_furniture
        kind: str = "table"
        body_name: str = "table"
        leg_length: float = 0.75
        leg_radius: float = 0.025
        leg_inset: float = 0.06
        top_size: tuple[float, float, float] = (1.0, 0.6, 0.04)
        seat_size: tuple[float, float, float] = (0.42, 0.42, 0.04)
        back_height: float = 0.42
        back_thickness: float = 0.04
        rgba: tuple[float, float, float, float] = _DEFAULT_RGBA
        physics_material_path: str = "material"
        physics_material: Any = None
        copy_from_source: bool = True

    _FURNITURE_SPAWNER_CLS = ProceduralFurnitureCfg
    return _FURNITURE_SPAWNER_CLS


def _isaac_spawn_kwargs(*, mass: float, collision_only: bool, activate_contact_sensors: bool) -> dict:
    import isaaclab.sim as sim_utils

    kw: dict = dict(
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        activate_contact_sensors=activate_contact_sensors and not collision_only,
    )
    if collision_only:
        return kw
    kw.update(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
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
    )
    return kw

def _make_furniture_isaaclab(
    kind: str,
    *,
    body_name: str,
    leg_length: float,
    leg_radius: float,
    leg_inset: float,
    top_size,
    seat_size,
    back_height: float,
    back_thickness: float,
    mass: float,
    rgba,
    pos,
    rot,
    collision_only: bool,
    activate_contact_sensors: bool,
):
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    ProceduralFurnitureCfg = _get_furniture_spawner_cls()
    spawn = ProceduralFurnitureCfg(
        kind=kind,
        body_name=body_name,
        leg_length=leg_length,
        leg_radius=leg_radius,
        leg_inset=leg_inset,
        top_size=_as_float_tuple(top_size, 3),
        seat_size=_as_float_tuple(seat_size, 3),
        back_height=back_height,
        back_thickness=back_thickness,
        rgba=_rgba(rgba),
        **_isaac_spawn_kwargs(
            mass=mass,
            collision_only=collision_only,
            activate_contact_sensors=activate_contact_sensors,
        ),
    )
    if collision_only:
        return AssetBaseCfg(
            spawn=spawn,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        )
    return RigidObjectCfg(
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


# ---------------------------------------------------------------------------
# mjlab
# ---------------------------------------------------------------------------


def _make_furniture_mjlab(
    kind: str,
    *,
    body_name: str,
    leg_length: float,
    leg_radius: float,
    leg_inset: float,
    top_size,
    seat_size,
    back_height: float,
    back_thickness: float,
    mass: float,
    rgba,
    pos,
    rot,
    collision_only: bool,
):
    from active_adaptation.assets.asset_cfg import EntityCfg
    from mjlab.utils.spec_config import CollisionCfg

    rgba_t = _rgba(rgba)

    def spec_fn():
        if kind == "table":
            spec = build_table_spec(
                leg_length=leg_length,
                top_size=top_size,
                leg_radius=leg_radius,
                leg_inset=leg_inset,
                rgba=rgba_t,
                mass=None if collision_only else mass,
                body_name=body_name,
            )
        else:
            spec = build_chair_spec(
                leg_length=leg_length,
                seat_size=seat_size,
                back_height=back_height,
                back_thickness=back_thickness,
                leg_radius=leg_radius,
                leg_inset=leg_inset,
                rgba=rgba_t,
                mass=None if collision_only else mass,
                body_name=body_name,
            )
        if not collision_only:
            # One freejoint on the single furniture body.
            body = next(iter(spec.worldbody.find_all("body")))
            body.add_freejoint(name=f"{body_name}_joint")
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


def _make_furniture(backend: Backend, kind: str, **kwargs):
    from active_adaptation.assets.asset_cfg import AssetSpec
    from active_adaptation.envs.robots.grasp_pose import GraspPose

    pos = _as_float_tuple(kwargs.pop("pos", _DEFAULT_POS), 3)
    rot = _as_float_tuple(kwargs.pop("rot", _DEFAULT_ROT), 4)
    collision_only = bool(kwargs.pop("collision_only", False))
    activate_contact_sensors = bool(kwargs.pop("activate_contact_sensors", True))
    attach_grasp = bool(kwargs.pop("attach_grasp", True))

    leg_length = float(kwargs["leg_length"])
    leg_radius = float(kwargs["leg_radius"])
    leg_inset = float(kwargs["leg_inset"])
    if kind == "table":
        lx, ly, _ = _as_float_tuple(kwargs["top_size"], 3)
        half_x, half_y = lx * 0.5, ly * 0.5
    else:
        sx, sy, _ = _as_float_tuple(kwargs["seat_size"], 3)
        half_x, half_y = sx * 0.5, sy * 0.5
    corners = _leg_corners(half_x, half_y, leg_inset)

    if backend == "isaaclab":
        cfg = _make_furniture_isaaclab(
            kind,
            pos=pos,
            rot=rot,
            collision_only=collision_only,
            activate_contact_sensors=activate_contact_sensors,
            **kwargs,
        )
    elif backend == "mjlab":
        del activate_contact_sensors
        cfg = _make_furniture_mjlab(
            kind,
            pos=pos,
            rot=rot,
            collision_only=collision_only,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")

    adaptations = ()
    if attach_grasp:
        adaptations = (
            GraspPose.for_legs(corners, leg_length=leg_length, leg_radius=leg_radius),
        )
    return AssetSpec(config=cfg, adaptations=adaptations)


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def make_table(
    backend: Backend,
    leg_length: float = 0.75,
    top_size: Sequence[float] = (1.0, 0.6, 0.04),
    leg_radius: float = 0.025,
    leg_inset: float = 0.06,
    mass: float = _DEFAULT_MASS,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    collision_only: bool = False,
    activate_contact_sensors: bool = True,
    attach_grasp: bool = True,
    name: str = "table",
):
    """Procedural table (box top + capsule legs). Non-articulated rigid body.

    Returns ``AssetSpec`` with prescribed leg-midpoint ``GraspPose`` adaptations
    (disable with ``attach_grasp=False``).
    """
    return _make_furniture(
        backend,
        "table",
        body_name=name,
        leg_length=leg_length,
        top_size=top_size,
        seat_size=(0.42, 0.42, 0.04),
        back_height=0.42,
        back_thickness=0.04,
        leg_radius=leg_radius,
        leg_inset=leg_inset,
        mass=mass,
        rgba=rgba,
        pos=pos,
        rot=rot,
        collision_only=collision_only,
        activate_contact_sensors=activate_contact_sensors,
        attach_grasp=attach_grasp,
    )


def make_chair(
    backend: Backend,
    leg_length: float = 0.45,
    seat_size: Sequence[float] = (0.42, 0.42, 0.04),
    back_height: float = 0.42,
    back_thickness: float = 0.04,
    leg_radius: float = 0.02,
    leg_inset: float = 0.04,
    mass: float = 8.0,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    collision_only: bool = False,
    activate_contact_sensors: bool = True,
    attach_grasp: bool = True,
    name: str = "chair",
):
    """Procedural chair (box seat/back + capsule legs). Non-articulated rigid body.

    Returns ``AssetSpec`` with prescribed leg-midpoint ``GraspPose`` adaptations
    (disable with ``attach_grasp=False``).
    """
    return _make_furniture(
        backend,
        "chair",
        body_name=name,
        leg_length=leg_length,
        top_size=(1.0, 0.6, 0.04),
        seat_size=seat_size,
        back_height=back_height,
        back_thickness=back_thickness,
        leg_radius=leg_radius,
        leg_inset=leg_inset,
        mass=mass,
        rgba=rgba,
        pos=pos,
        rot=rot,
        collision_only=collision_only,
        activate_contact_sensors=activate_contact_sensors,
        attach_grasp=attach_grasp,
    )


# ---------------------------------------------------------------------------
# Legacy USD-file dummy props (Isaac-only for now)
# ---------------------------------------------------------------------------


def _make_rigid(name: str):
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    path = ROBOT_MODEL_DIR / "dummy_objects" / f"{name}.usda"

    return RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.001,
                angular_damping=0.001,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def _make_static_platform(size: tuple[float, float, float] = (0.2, 0.2, 0.2)):
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    return RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.32, 0.32, 0.32),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, size[2] * 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_dummy_stand(backend: Backend):
    if backend != "isaaclab":
        raise NotImplementedError
    return _make_rigid("dummy_stand")


def make_dummy_basket(backend: Backend):
    if backend != "isaaclab":
        raise NotImplementedError
    return _make_rigid("dummy_basket")


def make_dummy_basket_platform(backend: Backend):
    if backend != "isaaclab":
        raise NotImplementedError
    return _make_static_platform()


registry.register("asset", "dummy_table", make_table)
registry.register("asset", "dummy_chair", make_chair)
registry.register("asset", "dummy_stand", make_dummy_stand)
registry.register("asset", "dummy_basket", make_dummy_basket)
registry.register("asset", "dummy_basket_platform", make_dummy_basket_platform)

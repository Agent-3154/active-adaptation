"""Shared helpers for procedural furniture / grasp-board assets."""
from __future__ import annotations

from typing import Any, Literal, Sequence

from active_adaptation.registry import Registry

registry = Registry.instance()

Backend = Literal["isaaclab", "mjlab"]

_DEFAULT_RGBA = (0.55, 0.42, 0.28, 1.0)
# Cooler slate for door jambs / lintel vs warm panel wood.
_DEFAULT_DOOR_FRAME_RGBA = (0.28, 0.30, 0.34, 1.0)
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


HandleShape = Literal["capsule", "box"]


def _parse_handle_shape(shape: str | HandleShape) -> HandleShape:
    key = str(shape).lower()
    if key not in ("capsule", "box"):
        raise ValueError(f"handle_shape must be 'capsule' or 'box', got {shape!r}")
    return key  # type: ignore[return-value]


def _handle_box_half_extents(
    *,
    handle_length: float,
    handle_radius: float,
    handle_box_size: Sequence[float] | None,
) -> tuple[float, float, float]:
    """Half-sizes ``(hx, hy, hz)`` for a thin bar handle along **+X**.

    ``handle_box_size`` is full ``(depth_y, height_z)``; defaults to a squat
    rectangle based on ``handle_radius`` (more graspable than a thin plate).
    """
    half_x = 0.5 * float(handle_length)
    if handle_box_size is None:
        # Depth (approach) a bit larger than height for finger wrap.
        half_y = max(float(handle_radius), 0.018)
        half_z = max(0.75 * float(handle_radius), 0.012)
    else:
        depth, height = _as_float_tuple(handle_box_size, 2)
        if depth <= 0 or height <= 0:
            raise ValueError(f"handle_box_size must be positive, got {handle_box_size}")
        half_y, half_z = 0.5 * depth, 0.5 * height
    return half_x, half_y, half_z


def _add_bar_handle(
    body,
    *,
    name: str,
    shape: HandleShape,
    handle_length: float,
    handle_radius: float,
    handle_box_size: Sequence[float] | None,
    y: float,
    rgba,
) -> float:
    """Add a horizontal bar handle centered at ``(0, y, 0)``, long axis **+X**.

    Returns the **Y half-extent** of the geom (capsule radius or box half-depth)
    so callers can place grasp frames.
    """
    import mujoco

    shape = _parse_handle_shape(shape)
    half_hl = 0.5 * float(handle_length)
    if shape == "capsule":
        r = float(handle_radius)
        _add_capsule_leg(
            body,
            name=name,
            radius=r,
            fromto=[-half_hl, y, 0.0, half_hl, y, 0.0],
            rgba=rgba,
        )
        return r

    hx, hy, hz = _handle_box_half_extents(
        handle_length=handle_length,
        handle_radius=handle_radius,
        handle_box_size=handle_box_size,
    )
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(hx, hy, hz),
        pos=(0.0, y, 0.0),
        rgba=rgba,
    )
    return hy


def _leg_corners(half_x: float, half_y: float, inset: float) -> list[tuple[float, float]]:
    hx = max(half_x - inset, 0.0)
    hy = max(half_y - inset, 0.0)
    return [(hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy)]


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


def _usd_apply_geom_rgba(stage, geom_path: str, rgba) -> None:
    """Color a USD gprim from MuJoCo ``geom.rgba`` (Isaac needs a visual material).

    Sets ``displayColor`` / ``displayOpacity`` and binds a per-geom
    ``PreviewSurface`` so RTX / interactive viewports pick up the color.
    MjSpec ``rgba`` alone is ignored by the USD path.
    """
    from pxr import UsdGeom, Gf
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_visual_material

    r, g, b, a = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
    prim = stage.GetPrimAtPath(geom_path)
    if not prim.IsValid():
        raise ValueError(f"Cannot color missing prim: {geom_path}")
    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr([(r, g, b)])
    gprim.CreateDisplayOpacityAttr([a])

    mat_path = f"{geom_path}/Looks/material"
    if not stage.GetPrimAtPath(mat_path).IsValid():
        mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(r, g, b), opacity=a)
        mat_cfg.func(mat_path, mat_cfg)
    bind_visual_material(geom_path, mat_path, stage=stage)


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
    # Local +Z → ``direction``: rotvec axis is ``Z × direction`` (not the reverse).
    # The reverse mapped 45° board bars onto the other diagonal; ±X/±Z fromto
    # still looked right because a 180° flip along the long axis is the same geom.
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        # Parallel to +Z (or -Z)
        orient = np.array([1.0, 0.0, 0.0, 0.0]) if direction[2] >= 0 else np.array([0.0, 1.0, 0.0, 0.0])
    else:
        angle = float(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
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
                # MuJoCo geom quat is wxyz (identity = 1,0,0,0).
                qw, qx, qy, qz = (float(v) for v in geom.quat)
                cube.GetPrim().GetAttribute("xformOp:orient").Set(
                    Gf.Quatf(qw, qx, qy, qz)
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
        _usd_apply_geom_rgba(stage, geom_path, geom.rgba)

    # Root pose matches compiled body pose (usually identity at origin).
    _usd_add_default_transform_(root)
    root.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*mjdata.xpos[mjbody.id]))
    root.GetAttribute("xformOp:orient").Set(Gf.Quatf(*mjdata.xquat[mjbody.id]))
    return root


def _usd_create_fixed_joint(stage, path: str, body_0, body_1):
    from pxr import UsdGeom, UsdPhysics, Gf

    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body_0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body_1.GetPath()])
    xf_cache = UsdGeom.XformCache()
    body_0_pose = xf_cache.GetLocalToWorldTransform(body_0)
    body_1_pose = xf_cache.GetLocalToWorldTransform(body_1)
    rel_pose = body_1_pose * body_0_pose.GetInverse()
    rel_pose = rel_pose.RemoveScaleShear()
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel_pose.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(rel_pose.ExtractRotationQuat()))
    return joint


def _usd_create_revolute_joint(stage, path: str, body_0, body_1, axis: str = "Z"):
    from pxr import UsdGeom, UsdPhysics, Gf

    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None

    assert axis in ("X", "Y", "Z"), f"Invalid axis: {axis}"
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body_0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body_1.GetPath()])
    joint.CreateAxisAttr(axis)
    xf_cache = UsdGeom.XformCache()
    body_0_pose = xf_cache.GetLocalToWorldTransform(body_0)
    body_1_pose = xf_cache.GetLocalToWorldTransform(body_1)
    rel_pose = body_1_pose * body_0_pose.GetInverse()
    rel_pose = rel_pose.RemoveScaleShear()
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel_pose.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(rel_pose.ExtractRotationQuat()))
    prim = joint.GetPrim()
    if not UsdPhysics.DriveAPI(prim, "angular"):
        UsdPhysics.DriveAPI.Apply(prim, "angular")
    if PhysxSchema is not None and not PhysxSchema.PhysxJointAPI(prim):
        PhysxSchema.PhysxJointAPI.Apply(prim)
    return joint


def _usd_create_prismatic_joint(stage, path: str, body_0, body_1, axis: str = "Y"):
    """Linear slide joint (MuJoCo ``slide`` / USD ``PrismaticJoint``)."""
    from pxr import UsdGeom, UsdPhysics, Gf

    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None

    assert axis in ("X", "Y", "Z"), f"Invalid axis: {axis}"
    joint = UsdPhysics.PrismaticJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body_0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body_1.GetPath()])
    joint.CreateAxisAttr(axis)
    xf_cache = UsdGeom.XformCache()
    body_0_pose = xf_cache.GetLocalToWorldTransform(body_0)
    body_1_pose = xf_cache.GetLocalToWorldTransform(body_1)
    rel_pose = body_1_pose * body_0_pose.GetInverse()
    rel_pose = rel_pose.RemoveScaleShear()
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel_pose.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(rel_pose.ExtractRotationQuat()))
    prim = joint.GetPrim()
    if not UsdPhysics.DriveAPI(prim, "linear"):
        UsdPhysics.DriveAPI.Apply(prim, "linear")
    if PhysxSchema is not None and not PhysxSchema.PhysxJointAPI(prim):
        PhysxSchema.PhysxJointAPI.Apply(prim)
    return joint


def _usd_add_body_geoms(stage, xform, mjbody) -> None:
    """Attach box/capsule collision geoms under a body xform (named from MJCF)."""
    import mujoco
    import numpy as np
    from pxr import UsdGeom, Gf, UsdPhysics

    for i, geom in enumerate(mjbody.geoms):
        name = geom.name or f"collision_{i}"
        geom_path = f"{xform.GetPath()}/{name}"
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
                qw, qx, qy, qz = (float(v) for v in geom.quat)
                cube.GetPrim().GetAttribute("xformOp:orient").Set(
                    Gf.Quatf(qw, qx, qy, qz)
                )
            case mujoco.mjtGeom.mjGEOM_CAPSULE:
                fromto = np.array(geom.fromto, dtype=float)
                if np.allclose(fromto, 0.0):
                    half = float(geom.size[1])
                    pos = np.array(geom.pos, dtype=float)
                    fromto = np.array(
                        [pos[0], pos[1], pos[2] - half, pos[0], pos[1], pos[2] + half],
                        dtype=float,
                    )
                _usd_create_capsule(stage, geom_path, float(geom.size[0]), fromto)
            case _:
                raise ValueError(f"Unsupported door geom type: {geom.type}")
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(geom_path))
        _usd_apply_geom_rgba(stage, geom_path, geom.rgba)


def _usd_from_mjspec_articulated(stage, prim_path: str, spec) -> object:
    """Build an articulated USD prim from an MjSpec (metamorphosis-style).

    Applies ``ArticulationRootAPI`` on the root and revolute / fixed joints
    between bodies. Collision geoms are children of each body xform.
    """
    import mujoco
    import numpy as np
    from pxr import UsdGeom, Gf, UsdPhysics

    mjmodel = spec.compile()
    mjdata = mujoco.MjData(mjmodel)
    mujoco.mj_forward(mjmodel, mjdata)

    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    _usd_add_default_transform_(root)

    prim_dict: dict[int, object] = {}
    for mjbody in spec.worldbody.find_all("body"):
        xform = UsdGeom.Xform.Define(stage, f"{prim_path}/{mjbody.name}")
        xform_prim = xform.GetPrim()
        _usd_add_body_geoms(stage, xform, mjbody)
        _usd_add_default_transform_(xform_prim)
        xform_prim.GetAttribute("xformOp:translate").Set(
            Gf.Vec3f(*mjdata.xpos[mjbody.id])
        )
        xform_prim.GetAttribute("xformOp:orient").Set(
            Gf.Quatf(*mjdata.xquat[mjbody.id])
        )
        UsdPhysics.CollisionAPI.Apply(xform_prim)
        UsdPhysics.RigidBodyAPI.Apply(xform_prim)
        prim_dict[mjbody.id] = xform_prim

        if mjbody.parent.id <= 0:
            continue
        parent_prim = prim_dict[mjbody.parent.id]
        joints = mjbody.joints
        if len(joints):
            if len(joints) != 1:
                raise ValueError(
                    "Only one joint per body is supported for articulated USD"
                )
            joint = joints[0]
            joint_path = f"{parent_prim.GetPath()}/{joint.name}"
            axis = ["X", "Y", "Z"][int(np.argmax(np.abs(joint.axis)))]
            jrange = np.asarray(joint.range, dtype=float)
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                # USD revolute limits are degrees.
                lim = jrange / np.pi * 180.0
                usd_joint = _usd_create_revolute_joint(
                    stage, joint_path, parent_prim, xform_prim, axis
                )
            elif joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                # USD prismatic limits are meters (same as MuJoCo).
                lim = jrange
                usd_joint = _usd_create_prismatic_joint(
                    stage, joint_path, parent_prim, xform_prim, axis
                )
            else:
                raise ValueError(f"Unsupported joint type: {joint.type}")
            usd_joint.CreateLowerLimitAttr(float(lim[0]))
            usd_joint.CreateUpperLimitAttr(float(lim[1]))
        else:
            joint_path = f"{parent_prim.GetPath()}/{mjbody.name}_fixed"
            _usd_create_fixed_joint(stage, joint_path, parent_prim, xform_prim)
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
        elif cfg.kind == "grasp_board":
            spec = build_grasp_board_spec(
                panel_size=cfg.panel_size,
                handle_length=cfg.handle_length,
                handle_radius=cfg.handle_radius,
                handle_box_size=cfg.handle_box_size,
                bar_standoffs=cfg.bar_standoffs,
                panel_rgba=cfg.panel_rgba,
                box_rgba=cfg.box_rgba,
                capsule_rgba=cfg.capsule_rgba,
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
        # grasp_board fields
        panel_size: tuple[float, float, float] = (1.1, 0.04, 1.0)
        handle_length: float = 0.16
        handle_radius: float = 0.022
        handle_box_size: tuple[float, float] | None = (0.04, 0.03)
        bar_standoffs: tuple[float, ...] | None = None
        panel_rgba: tuple[float, float, float, float] = _DEFAULT_GRASP_BOARD_PANEL_RGBA
        box_rgba: tuple[float, float, float, float] = _DEFAULT_GRASP_BOARD_BOX_RGBA
        capsule_rgba: tuple[float, float, float, float] = _DEFAULT_GRASP_BOARD_CAPSULE_RGBA
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


"""Procedural furniture assets (table, chair, door, drawer).

Non-articulated (``table``, ``chair``): single rigid body with multiple collision
geoms. Articulated (``door``, ``drawer``): fixed-base multi-body MJCF → Isaac
articulated USD / mjlab ``EntityCfg.spec_fn``.

Shared USD / MjSpec helpers live in ``_procedural``.
"""
from __future__ import annotations

from typing import Any, Sequence

from active_adaptation.assets._procedural import (
    Backend,
    HandleShape,
    _DEFAULT_DOOR_FRAME_RGBA,
    _DEFAULT_MASS,
    _DEFAULT_POS,
    _DEFAULT_RGBA,
    _DEFAULT_ROT,
    _add_bar_handle,
    _add_capsule_leg,
    _as_float_tuple,
    _handle_box_half_extents,
    _isaac_spawn_kwargs,
    _leg_corners,
    _parse_handle_shape,
    _rgba,
    _usd_from_mjspec_articulated,
    _usd_from_mjspec_rigid,
    registry,
)

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


def build_door_spec(
    *,
    door_dimensions: Sequence[float] = (0.9, 0.04, 2.0),
    handle_position: Sequence[float] = (0.35, 1.0),
    frame_thickness: float = 0.08,
    handle_radius: float = 0.022,
    handle_length: float = 0.16,
    handle_shape: HandleShape = "capsule",
    handle_box_size: Sequence[float] | None = None,
    door_joint_range: tuple[float, float] = (-1.8, 1.8),
    handle_joint_range: tuple[float, float] = (-1.2, 1.2),
    rgba: Sequence[float] = _DEFAULT_RGBA,
    frame_rgba: Sequence[float] = _DEFAULT_DOOR_FRAME_RGBA,
    body_name: str = "door",
):
    """Articulated door: fixed ``frame`` → hinged ``panel`` → hinged ``handle``.

    Joints
    ------
    - ``door_joint``: revolute about **+Z** at the left (-X) edge of the panel.
    - ``handle_joint``: revolute about **+Y** (through the door) at the handle.

    Object / frame frame: origin at floor under the opening center; **+Z** up,
    **+Y** through the door, **+X** along the width. ``door_dimensions`` is full
    ``(width, thickness, height)``. ``handle_position`` is ``(x, z)`` in the
    **frame** frame (same as the closed panel face). No freejoint — fixed-base
    fixture (mjlab auto-mocap / Isaac ``fix_root_link`` on ``frame``).

    ``handle_shape`` is ``"capsule"`` (default) or ``"box"`` (thin bar). For
    boxes, ``handle_box_size`` is full ``(depth_y, height_z)``; omitted → derived
    from ``handle_radius``.
    """
    import mujoco

    del body_name
    width, thickness, height = _as_float_tuple(door_dimensions, 3)
    hx, hz = _as_float_tuple(handle_position, 2)
    ft = float(frame_thickness)
    hr = float(handle_radius)
    hl = float(handle_length)
    shape = _parse_handle_shape(handle_shape)
    rgba_t = _rgba(rgba)
    frame_rgba_t = _rgba(frame_rgba)

    if width <= 0 or thickness <= 0 or height <= 0:
        raise ValueError(f"door_dimensions must be positive, got {door_dimensions}")
    if ft <= 0:
        raise ValueError(f"frame_thickness must be positive, got {frame_thickness}")
    if abs(hx) > 0.5 * width:
        raise ValueError(
            f"handle_position x={hx} is outside door half-width {0.5 * width}"
        )
    if not (0.0 < hz < height):
        raise ValueError(f"handle_position z={hz} must be in (0, height={height})")

    half_w, half_d, half_h = width * 0.5, thickness * 0.5, height * 0.5
    half_ft = ft * 0.5
    hinge_x = -half_w

    # Protrusion of handle geom along ±Y from the panel face.
    if shape == "capsule":
        y_half = hr
    else:
        _, y_half, _ = _handle_box_half_extents(
            handle_length=hl, handle_radius=hr, handle_box_size=handle_box_size
        )

    spec = mujoco.MjSpec()
    frame = spec.worldbody.add_body(name="frame")
    frame.mass = 20.0
    frame.inertia = [1.0, 1.0, 1.0]

    # U-frame (left/right jambs + top lintel), same depth as the panel.
    jamb_x = half_w + half_ft
    for side, x in (("left", -jamb_x), ("right", jamb_x)):
        frame.add_geom(
            name=f"frame_{side}_collision",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(half_ft, half_d, half_h),
            pos=(x, 0.0, half_h),
            rgba=frame_rgba_t,
        )
    frame.add_geom(
        name="frame_top_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_w + ft, half_d, half_ft),
        pos=(0.0, 0.0, height + half_ft),
        rgba=frame_rgba_t,
    )

    # Panel hinged at the left edge; geom centered to fill the opening when q=0.
    panel = frame.add_body(name="panel", pos=(hinge_x, 0.0, 0.0))
    panel.mass = 15.0
    panel.inertia = [1.0, 1.0, 1.0]
    door_joint = panel.add_joint(
        name="door_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 0.0, 1.0],
    )
    door_joint.range = list(door_joint_range)
    panel.add_geom(
        name="panel_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_w, half_d, half_h),
        pos=(half_w, 0.0, half_h),
        rgba=rgba_t,
    )

    # Handle body at the latch; joint about +Y (through the door).
    # Panel-frame handle position: frame (hx, 0, hz) → panel (hx - hinge_x, 0, hz).
    handle_pos_panel = (hx - hinge_x, 0.0, hz)
    handle = panel.add_body(name="handle", pos=handle_pos_panel)
    handle.mass = 0.4
    handle.inertia = [0.01, 0.01, 0.01]
    handle_joint = handle.add_joint(
        name="handle_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 1.0, 0.0],
    )
    handle_joint.range = list(handle_joint_range)

    for side, y_sign in (("front", +1.0), ("back", -1.0)):
        y = y_sign * (half_d + y_half)
        _add_bar_handle(
            handle,
            name=f"handle_{side}_collision",
            shape=shape,
            handle_length=hl,
            handle_radius=hr,
            handle_box_size=handle_box_size,
            y=y,
            rgba=rgba_t,
        )
    return spec


DOOR_JOINT_NAMES_SIMULATION = ["door_joint", "handle_joint"]
DOOR_BODY_NAMES_SIMULATION = ["frame", "panel", "handle"]
DOOR_INIT_JOINT_POS = {"door_joint": 0.0, "handle_joint": 0.0}


def _drawer_joint_names(num_drawers: int) -> list[str]:
    return [f"drawer_{i}_joint" for i in range(num_drawers)]


def _drawer_body_names(num_drawers: int) -> list[str]:
    names: list[str] = ["frame"]
    for i in range(num_drawers):
        names.append(f"drawer_{i}")
        names.append(f"handle_{i}")
    return names


def _drawer_init_joint_pos(num_drawers: int) -> dict[str, float]:
    return {name: 0.0 for name in _drawer_joint_names(num_drawers)}


def build_drawer_spec(
    *,
    cabinet_dimensions: Sequence[float] = (0.5, 0.4, 0.7),
    num_drawers: int = 2,
    wall_thickness: float = 0.02,
    drawer_gap: float = 0.006,
    drawer_travel: float | None = None,
    handle_radius: float = 0.022,
    handle_length: float = 0.18,
    handle_shape: HandleShape = "capsule",
    handle_box_size: Sequence[float] | None = None,
    handle_standoff: float = 0.008,
    drawer_joint_range: tuple[float, float] | None = None,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    frame_rgba: Sequence[float] = _DEFAULT_DOOR_FRAME_RGBA,
    body_name: str = "drawer",
):
    """Articulated multi-drawer cabinet: fixed ``frame`` + stacked drawers.

    Per drawer ``i`` (bottom → top)::

        frame ──drawer_{i}_joint (slide +Y)──▶ drawer_{i}
                                             └── (welded) handle_{i}

    No handle joints / locks — handles are fixed bars on the drawer front.
    ``handle_shape`` is ``"capsule"`` or ``"box"`` (see ``build_door_spec``).
    ``handle_standoff`` is the gap along **+Y** between the drawer front
    face and the inner face of the handle bar.
    """
    import mujoco

    del body_name
    n_drawers = int(num_drawers)
    if n_drawers < 1:
        raise ValueError(f"num_drawers must be >= 1, got {num_drawers}")

    width, depth, height = _as_float_tuple(cabinet_dimensions, 3)
    wt = float(wall_thickness)
    gap = float(drawer_gap)
    hr = float(handle_radius)
    hl = float(handle_length)
    shape = _parse_handle_shape(handle_shape)
    rgba_t = _rgba(rgba)
    frame_rgba_t = _rgba(frame_rgba)

    if width <= 0 or depth <= 0 or height <= 0:
        raise ValueError(
            f"cabinet_dimensions must be positive, got {cabinet_dimensions}"
        )
    if wt <= 0:
        raise ValueError(f"wall_thickness must be positive, got {wall_thickness}")
    if gap < 0:
        raise ValueError(f"drawer_gap must be >= 0, got {drawer_gap}")
    standoff = float(handle_standoff)
    if standoff < 0.0:
        raise ValueError(f"handle_standoff must be >= 0, got {handle_standoff}")

    half_w, half_d, half_h = width * 0.5, depth * 0.5, height * 0.5
    half_wt = wt * 0.5

    # Interior cavity (front open on +Y). Thin shelves between drawers.
    inner_w = width - 2.0 * wt
    inner_h = height - 2.0 * wt
    n_shelves = n_drawers - 1
    shelf_t = min(wt, 0.012)
    usable_h = inner_h - n_shelves * shelf_t - (n_drawers + 1) * gap
    drawer_h = usable_h / float(n_drawers)
    drawer_depth = depth - wt - gap
    drawer_w = inner_w - 2.0 * gap
    if drawer_w <= 0 or drawer_h <= 0 or drawer_depth <= 0:
        raise ValueError(
            "Cabinet too small for walls/gap/drawers: "
            f"drawer size ({drawer_w}, {drawer_depth}, {drawer_h})"
        )

    travel = (
        float(drawer_travel)
        if drawer_travel is not None
        else 0.75 * drawer_depth
    )
    if travel <= 0:
        raise ValueError(f"drawer_travel must be positive, got {travel}")
    if drawer_joint_range is None:
        joint_lo, joint_hi = 0.0, travel
    else:
        joint_lo, joint_hi = _as_float_tuple(drawer_joint_range, 2)
        if joint_hi <= joint_lo:
            raise ValueError(
                f"drawer_joint_range must be increasing, got {drawer_joint_range}"
            )

    if shape == "capsule":
        y_half = hr
    else:
        _, y_half, _ = _handle_box_half_extents(
            handle_length=hl, handle_radius=hr, handle_box_size=handle_box_size
        )

    half_dw, half_dd, half_dh = drawer_w * 0.5, drawer_depth * 0.5, drawer_h * 0.5
    # Front outer face at +half_d when closed.
    drawer_y0 = half_d - half_dd

    spec = mujoco.MjSpec()
    frame = spec.worldbody.add_body(name="frame")
    frame.mass = 25.0
    frame.inertia = [2.0, 2.0, 2.0]

    # Cabinet shell: bottom, top, left, right, back (front open).
    frame.add_geom(
        name="frame_bottom_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_w, half_d, half_wt),
        pos=(0.0, 0.0, half_wt),
        rgba=frame_rgba_t,
    )
    frame.add_geom(
        name="frame_top_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_w, half_d, half_wt),
        pos=(0.0, 0.0, height - half_wt),
        rgba=frame_rgba_t,
    )
    mid_z = half_h
    wall_hz = half_h - wt
    for side, x in (("left", -(half_w - half_wt)), ("right", half_w - half_wt)):
        frame.add_geom(
            name=f"frame_{side}_collision",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(half_wt, half_d, wall_hz),
            pos=(x, 0.0, mid_z),
            rgba=frame_rgba_t,
        )
    frame.add_geom(
        name="frame_back_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_w - wt, half_wt, wall_hz),
        pos=(0.0, -(half_d - half_wt), mid_z),
        rgba=frame_rgba_t,
    )

    # Horizontal shelves between drawer slots.
    half_shelf = shelf_t * 0.5
    z_cursor = wt + gap
    for i in range(n_drawers):
        drawer_z = z_cursor + half_dh
        drawer = frame.add_body(
            name=f"drawer_{i}", pos=(0.0, drawer_y0, drawer_z)
        )
        drawer.mass = 4.0
        drawer.inertia = [0.2, 0.2, 0.2]
        slide = drawer.add_joint(
            name=f"drawer_{i}_joint",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 1.0, 0.0],
        )
        slide.range = [joint_lo, joint_hi]

        panel_t = min(0.015, 0.5 * half_dd)
        half_pt = panel_t * 0.5
        drawer.add_geom(
            name=f"drawer_{i}_bottom_collision",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(half_dw, half_dd, half_pt),
            pos=(0.0, 0.0, -(half_dh - half_pt)),
            rgba=rgba_t,
        )
        drawer.add_geom(
            name=f"drawer_{i}_front_collision",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(half_dw, half_pt, half_dh),
            pos=(0.0, half_dd - half_pt, 0.0),
            rgba=rgba_t,
        )
        drawer.add_geom(
            name=f"drawer_{i}_back_collision",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(half_dw, half_pt, half_dh),
            pos=(0.0, -(half_dd - half_pt), 0.0),
            rgba=rgba_t,
        )
        side_hy = half_dd - panel_t
        for side, x in (("left", -(half_dw - half_pt)), ("right", half_dw - half_pt)):
            drawer.add_geom(
                name=f"drawer_{i}_{side}_collision",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(half_pt, side_hy, half_dh),
                pos=(x, 0.0, 0.0),
                rgba=rgba_t,
            )

        # Fixed (welded) pull-bar on the front face — no lock / no handle joint.
        handle_y = half_dd + standoff + y_half
        handle = drawer.add_body(name=f"handle_{i}", pos=(0.0, handle_y, 0.0))
        handle.mass = 0.25
        handle.inertia = [0.01, 0.01, 0.01]
        _add_bar_handle(
            handle,
            name=f"handle_{i}_front_collision",
            shape=shape,
            handle_length=hl,
            handle_radius=hr,
            handle_box_size=handle_box_size,
            y=0.0,
            rgba=rgba_t,
        )

        z_cursor += drawer_h + gap
        if i < n_shelves:
            shelf_z = z_cursor + half_shelf
            frame.add_geom(
                name=f"frame_shelf_{i}_collision",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(half_w - wt, half_d - half_wt, half_shelf),
                pos=(0.0, half_wt * 0.5, shelf_z),
                rgba=frame_rgba_t,
            )
            z_cursor += shelf_t + gap

    return spec


# Default simulation order for the default ``num_drawers=2`` cabinet.
DRAWER_JOINT_NAMES_SIMULATION = _drawer_joint_names(2)
DRAWER_BODY_NAMES_SIMULATION = _drawer_body_names(2)
DRAWER_INIT_JOINT_POS = _drawer_init_joint_pos(2)

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
    from active_adaptation.envs.behaviors.grasp_pose import GraspPose

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

    behaviors = ()
    if attach_grasp:
        behaviors = (
            GraspPose.for_legs(corners, leg_length=leg_length, leg_radius=leg_radius),
        )
    return AssetSpec(config=cfg, behaviors=behaviors)


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

    Returns ``AssetSpec`` with prescribed leg-midpoint ``GraspPose`` behaviors
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

    Returns ``AssetSpec`` with prescribed leg-midpoint ``GraspPose`` behaviors
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


_DOOR_SPAWNER_CLS = None


def _get_door_spawner_cls():
    """Lazy Isaac spawner for the articulated procedural door."""
    global _DOOR_SPAWNER_CLS
    if _DOOR_SPAWNER_CLS is not None:
        return _DOOR_SPAWNER_CLS

    from collections.abc import Callable

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab.utils import configclass
    from pxr import Usd

    @clone
    def spawn_door(
        prim_path: str,
        cfg: "ProceduralDoorCfg",
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> Usd.Prim:
        del kwargs
        stage = get_current_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            raise ValueError(f"A prim already exists at path: '{prim_path}'.")

        spec = build_door_spec(
            door_dimensions=cfg.door_dimensions,
            handle_position=cfg.handle_position,
            frame_thickness=cfg.frame_thickness,
            handle_radius=cfg.handle_radius,
            handle_length=cfg.handle_length,
            handle_shape=cfg.handle_shape,
            handle_box_size=cfg.handle_box_size,
            door_joint_range=cfg.door_joint_range,
            handle_joint_range=cfg.handle_joint_range,
            rgba=cfg.rgba,
            frame_rgba=cfg.frame_rgba,
        )
        root = _usd_from_mjspec_articulated(stage, prim_path, spec)

        from pxr import Gf, UsdPhysics
        from isaaclab.sim.utils import bind_physics_material

        try:
            from pxr import PhysxSchema
        except ImportError:
            PhysxSchema = None

        if translation is not None:
            root.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*translation))
        if orientation is not None:
            root.GetAttribute("xformOp:orient").Set(Gf.Quatf(*orientation))

        # Collision / material on every collision geom under each link.
        for body_name in DOOR_BODY_NAMES_SIMULATION:
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
            for body_name in DOOR_BODY_NAMES_SIMULATION:
                body_prim = stage.GetPrimAtPath(f"{prim_path}/{body_name}")
                if not body_prim.IsValid():
                    continue
                for child in body_prim.GetChildren():
                    if not child.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    bind_physics_material(str(child.GetPath()), material_path, stage=stage)

        # Isaac ``fix_root_link`` needs ArticulationRootAPI on a RigidBody.
        # ``_usd_from_mjspec_articulated`` puts it on the outer Xform; move it
        # onto ``frame`` so the jamb is welded to world.
        frame_path = f"{prim_path}/frame"
        frame_prim = stage.GetPrimAtPath(frame_path)
        if not frame_prim.IsValid():
            raise RuntimeError(f"Expected frame body at {frame_path}")
        if root.HasAPI(UsdPhysics.ArticulationRootAPI):
            root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if PhysxSchema is not None and root.HasAPI(PhysxSchema.PhysxArticulationAPI):
            root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        UsdPhysics.ArticulationRootAPI.Apply(frame_prim)

        if cfg.articulation_props is not None:
            schemas.modify_articulation_root_properties(
                frame_path, cfg.articulation_props
            )
        if cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_path, stage=stage)
        return root

    @configclass
    class ProceduralDoorCfg(SpawnerCfg):
        func: Callable = spawn_door
        door_dimensions: tuple[float, float, float] = (0.9, 0.04, 2.0)
        handle_position: tuple[float, float] = (0.35, 1.0)
        frame_thickness: float = 0.08
        handle_radius: float = 0.022
        handle_length: float = 0.16
        handle_shape: str = "capsule"
        handle_box_size: tuple[float, float] | None = None
        door_joint_range: tuple[float, float] = (-1.8, 1.8)
        handle_joint_range: tuple[float, float] = (-1.2, 1.2)
        rgba: tuple[float, float, float, float] = _DEFAULT_RGBA
        frame_rgba: tuple[float, float, float, float] = _DEFAULT_DOOR_FRAME_RGBA
        collision_props: Any = None
        physics_material_path: str = "material"
        physics_material: Any = None
        articulation_props: Any = None
        activate_contact_sensors: bool = True
        copy_from_source: bool = False

    _DOOR_SPAWNER_CLS = ProceduralDoorCfg
    return _DOOR_SPAWNER_CLS



def make_door(
    backend: Backend,
    door_dimensions: Sequence[float] = (0.9, 0.04, 2.0),
    handle_position: Sequence[float] = (0.35, 1.0),
    frame_thickness: float = 0.1,
    handle_radius: float = 0.022,
    handle_length: float = 0.16,
    handle_shape: HandleShape = "capsule",
    handle_box_size: Sequence[float] | None = None,
    door_joint_range: Sequence[float] = (-1.8, 1.8),
    handle_joint_range: Sequence[float] = (-1.2, 1.2),
    rgba: Sequence[float] = _DEFAULT_RGBA,
    frame_rgba: Sequence[float] = _DEFAULT_DOOR_FRAME_RGBA,
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    activate_contact_sensors: bool = True,
    attach_grasp: bool = True,
    attach_door: bool = True,
    open_direction: str = "pull",
    handle_unlock_threshold_deg: float = 30.0,
    initially_locked: bool = True,
    name: str = "door",
):
    """Articulated door: ``frame`` —``door_joint``→ ``panel`` —``handle_joint``→ ``handle``.

    Fixed-base fixture (no freejoint). Isaac welds ``frame`` with
    ``fix_root_link=True``; mjlab auto-wraps a mocap root. ``door_dimensions``
    is full ``(width, thickness, height)``. ``handle_position`` is ``(x, z)``
    in the frame frame. ``rgba`` is panel/handle; ``frame_rgba`` is jambs/lintel.
    ``door_joint`` defaults to **zero stiffness** (free hinge + light damping).

    Behaviors (disable with flags):
    - ``DoorBehavior`` (``door.door``): lock / push-pull / handle unlock
    - ``GraspPose`` (``door.grasp``): prescribed handle-face grasps
    """
    from active_adaptation.assets.asset_cfg import AssetSpec
    from active_adaptation.envs.behaviors.door import DoorBehavior
    from active_adaptation.envs.behaviors.grasp_pose import GraspPose

    door_dimensions_t = _as_float_tuple(door_dimensions, 3)
    handle_position_t = _as_float_tuple(handle_position, 2)
    door_range_t = _as_float_tuple(door_joint_range, 2)
    handle_range_t = _as_float_tuple(handle_joint_range, 2)
    pos_t = _as_float_tuple(pos, 3)
    rot_t = _as_float_tuple(rot, 4)
    rgba_t = _rgba(rgba)
    frame_rgba_t = _rgba(frame_rgba)
    shape = _parse_handle_shape(handle_shape)
    box_size_t = (
        None if handle_box_size is None else _as_float_tuple(handle_box_size, 2)
    )
    del name  # link names are fixed: frame / panel / handle

    if backend == "isaaclab":
        import isaaclab.sim as sim_utils
        from active_adaptation.assets.asset_cfg import ArticulationCfg, ImplicitActuatorCfg

        ProceduralDoorCfg = _get_door_spawner_cls()
        spawn = ProceduralDoorCfg(
            door_dimensions=door_dimensions_t,
            handle_position=handle_position_t,
            frame_thickness=float(frame_thickness),
            handle_radius=float(handle_radius),
            handle_length=float(handle_length),
            handle_shape=shape,
            handle_box_size=box_size_t,
            door_joint_range=door_range_t,
            handle_joint_range=handle_range_t,
            rgba=rgba_t,
            frame_rgba=frame_rgba_t,
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
            activate_contact_sensors=activate_contact_sensors,
            copy_from_source=False,
        )
        cfg = ArticulationCfg(
            spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos_t,
                rot=rot_t,
                joint_pos=dict(DOOR_INIT_JOINT_POS),
                joint_vel={".*": 0.0},
            ),
            actuators={
                "door": ImplicitActuatorCfg(
                    joint_names_expr=["door_joint"],
                    effort_limit_sim=80.0,
                    stiffness=0.0,
                    damping=4.0,
                    armature=0.01,
                    friction=0.01,
                ),
                "handle": ImplicitActuatorCfg(
                    joint_names_expr=["handle_joint"],
                    effort_limit_sim=20.0,
                    stiffness=20.0,
                    damping=2.0,
                    armature=0.005,
                    friction=0.01,
                ),
            },
            joint_names_simulation=list(DOOR_JOINT_NAMES_SIMULATION),
            body_names_simulation=list(DOOR_BODY_NAMES_SIMULATION),
        )
    elif backend == "mjlab":
        from active_adaptation.assets.asset_cfg import EntityCfg
        from mjlab.actuator import BuiltinPdActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg
        from mjlab.utils.spec_config import CollisionCfg

        def spec_fn():
            # No freejoint ⇒ fixed base. mjlab auto-wraps a mocap root so
            # per-env placement via init_state still works.
            return build_door_spec(
                door_dimensions=door_dimensions_t,
                handle_position=handle_position_t,
                frame_thickness=frame_thickness,
                handle_radius=handle_radius,
                handle_length=handle_length,
                handle_shape=shape,
                handle_box_size=box_size_t,
                door_joint_range=door_range_t,
                handle_joint_range=handle_range_t,
                rgba=rgba_t,
                frame_rgba=frame_rgba_t,
            )

        cfg = EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=pos_t,
                rot=rot_t,
                joint_pos=dict(DOOR_INIT_JOINT_POS),
                joint_vel={".*": 0.0},
            ),
            spec_fn=spec_fn,
            articulation=EntityArticulationInfoCfg(
                actuators=(
                    BuiltinPdActuatorCfg(
                        target_names_expr=("door_joint",),
                        effort_limit=80.0,
                        stiffness=0.0,
                        damping=4.0,
                        armature=0.01,
                        frictionloss=0.01,
                    ),
                    BuiltinPdActuatorCfg(
                        target_names_expr=("handle_joint",),
                        effort_limit=20.0,
                        stiffness=20.0,
                        damping=2.0,
                        armature=0.005,
                        frictionloss=0.01,
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
            joint_names_simulation=list(DOOR_JOINT_NAMES_SIMULATION),
            body_names_simulation=list(DOOR_BODY_NAMES_SIMULATION),
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")

    behaviors: list = []
    if attach_door:
        behaviors.append(
            DoorBehavior(
                open_direction=open_direction,  # type: ignore[arg-type]
                handle_unlock_threshold_deg=handle_unlock_threshold_deg,
                initially_locked=initially_locked,
            )
        )
    if attach_grasp:
        behaviors.append(
            GraspPose.for_door_handles(
                door_thickness=door_dimensions_t[1],
                handle_radius=float(handle_radius),
                handle_shape=shape,
                handle_box_size=box_size_t,
            )
        )
    return AssetSpec(config=cfg, behaviors=tuple(behaviors))


_DRAWER_SPAWNER_CLS = None


def _get_drawer_spawner_cls():
    """Lazy Isaac spawner for the articulated procedural drawer."""
    global _DRAWER_SPAWNER_CLS
    if _DRAWER_SPAWNER_CLS is not None:
        return _DRAWER_SPAWNER_CLS

    from collections.abc import Callable

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab.utils import configclass
    from pxr import Usd

    @clone
    def spawn_drawer(
        prim_path: str,
        cfg: "ProceduralDrawerCfg",
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> Usd.Prim:
        del kwargs
        stage = get_current_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            raise ValueError(f"A prim already exists at path: '{prim_path}'.")

        spec = build_drawer_spec(
            cabinet_dimensions=cfg.cabinet_dimensions,
            num_drawers=cfg.num_drawers,
            wall_thickness=cfg.wall_thickness,
            drawer_gap=cfg.drawer_gap,
            drawer_travel=cfg.drawer_travel,
            handle_radius=cfg.handle_radius,
            handle_length=cfg.handle_length,
            handle_shape=cfg.handle_shape,
            handle_box_size=cfg.handle_box_size,
            handle_standoff=cfg.handle_standoff,
            drawer_joint_range=cfg.drawer_joint_range,
            rgba=cfg.rgba,
            frame_rgba=cfg.frame_rgba,
        )
        root = _usd_from_mjspec_articulated(stage, prim_path, spec)

        from pxr import Gf, UsdPhysics
        from isaaclab.sim.utils import bind_physics_material

        if translation is not None:
            root.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*translation))
        if orientation is not None:
            root.GetAttribute("xformOp:orient").Set(Gf.Quatf(*orientation))

        body_names = _drawer_body_names(int(cfg.num_drawers))
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

        if cfg.articulation_props is not None:
            schemas.modify_articulation_root_properties(
                prim_path, cfg.articulation_props
            )
        if cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_path, stage=stage)
        return root

    @configclass
    class ProceduralDrawerCfg(SpawnerCfg):
        func: Callable = spawn_drawer
        cabinet_dimensions: tuple[float, float, float] = (0.5, 0.4, 0.7)
        num_drawers: int = 2
        wall_thickness: float = 0.02
        drawer_gap: float = 0.006
        drawer_travel: float | None = None
        handle_radius: float = 0.022
        handle_length: float = 0.18
        handle_shape: str = "capsule"
        handle_box_size: tuple[float, float] | None = None
        handle_standoff: float = 0.008
        drawer_joint_range: tuple[float, float] | None = None
        rgba: tuple[float, float, float, float] = _DEFAULT_RGBA
        frame_rgba: tuple[float, float, float, float] = _DEFAULT_DOOR_FRAME_RGBA
        collision_props: Any = None
        physics_material_path: str = "material"
        physics_material: Any = None
        articulation_props: Any = None
        activate_contact_sensors: bool = True
        copy_from_source: bool = False

    _DRAWER_SPAWNER_CLS = ProceduralDrawerCfg
    return _DRAWER_SPAWNER_CLS


def make_drawer(
    backend: Backend,
    cabinet_dimensions: Sequence[float] = (0.5, 0.4, 0.7),
    num_drawers: int = 2,
    wall_thickness: float = 0.02,
    drawer_gap: float = 0.006,
    drawer_travel: float | None = None,
    handle_radius: float = 0.022,
    handle_length: float = 0.18,
    handle_shape: HandleShape = "capsule",
    handle_box_size: Sequence[float] | None = None,
    handle_standoff: float = 0.008,
    drawer_joint_range: Sequence[float] | None = None,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    frame_rgba: Sequence[float] = _DEFAULT_DOOR_FRAME_RGBA,
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    activate_contact_sensors: bool = True,
    attach_grasp: bool = True,
    attach_drawer: bool = True,
    name: str = "drawer",
):
    """Articulated multi-drawer cabinet (``num_drawers``, default 2).

    Per drawer ``i``: ``frame`` —``drawer_{i}_joint``→ ``drawer_{i}`` with a
    **welded** ``handle_{i}`` (no lock / no handle joint). Pull-out along **+Y**;
    slide joints default to **zero stiffness**.

    ``handle_shape``: ``"capsule"`` or ``"box"``; optional ``handle_box_size``
    as full ``(depth_y, height_z)``. ``handle_standoff`` is the gap between
    the drawer front and the inner face of the bar (default 8 mm).

    Behaviors (disable with flags):
    - ``DrawerBehavior`` (``drawer.drawer``): slide joint accessors
    - ``GraspPose`` (``drawer.grasp``): prescribed front-handle grasp (local)
    """
    from active_adaptation.assets.asset_cfg import AssetSpec
    from active_adaptation.envs.behaviors.drawer import DrawerBehavior
    from active_adaptation.envs.behaviors.grasp_pose import GraspPose

    n_drawers = int(num_drawers)
    if n_drawers < 1:
        raise ValueError(f"num_drawers must be >= 1, got {num_drawers}")

    cabinet_t = _as_float_tuple(cabinet_dimensions, 3)
    drawer_range_t = (
        None
        if drawer_joint_range is None
        else _as_float_tuple(drawer_joint_range, 2)
    )
    pos_t = _as_float_tuple(pos, 3)
    rot_t = _as_float_tuple(rot, 4)
    rgba_t = _rgba(rgba)
    frame_rgba_t = _rgba(frame_rgba)
    shape = _parse_handle_shape(handle_shape)
    box_size_t = (
        None if handle_box_size is None else _as_float_tuple(handle_box_size, 2)
    )
    travel = None if drawer_travel is None else float(drawer_travel)
    standoff = float(handle_standoff)
    if standoff < 0.0:
        raise ValueError(f"handle_standoff must be >= 0, got {handle_standoff}")
    joint_names = _drawer_joint_names(n_drawers)
    body_names = _drawer_body_names(n_drawers)
    init_joint_pos = _drawer_init_joint_pos(n_drawers)
    del name

    if backend == "isaaclab":
        import isaaclab.sim as sim_utils
        from active_adaptation.assets.asset_cfg import (
            ArticulationCfg,
            ImplicitActuatorCfg,
        )

        ProceduralDrawerCfg = _get_drawer_spawner_cls()
        spawn = ProceduralDrawerCfg(
            cabinet_dimensions=cabinet_t,
            num_drawers=n_drawers,
            wall_thickness=float(wall_thickness),
            drawer_gap=float(drawer_gap),
            drawer_travel=travel,
            handle_radius=float(handle_radius),
            handle_length=float(handle_length),
            handle_shape=shape,
            handle_box_size=box_size_t,
            handle_standoff=standoff,
            drawer_joint_range=drawer_range_t,
            rgba=rgba_t,
            frame_rgba=frame_rgba_t,
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
            ),
            activate_contact_sensors=activate_contact_sensors,
            copy_from_source=False,
        )
        cfg = ArticulationCfg(
            spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos_t,
                rot=rot_t,
                joint_pos=dict(init_joint_pos),
                joint_vel={".*": 0.0},
            ),
            actuators={
                "drawers": ImplicitActuatorCfg(
                    joint_names_expr=["drawer_.*_joint"],
                    effort_limit_sim=80.0,
                    stiffness=0.0,
                    damping=4.0,
                    armature=0.01,
                    friction=0.01,
                ),
            },
            joint_names_simulation=list(joint_names),
            body_names_simulation=list(body_names),
        )
    elif backend == "mjlab":
        from active_adaptation.assets.asset_cfg import EntityCfg
        from mjlab.actuator import BuiltinPdActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg
        from mjlab.utils.spec_config import CollisionCfg

        def spec_fn():
            return build_drawer_spec(
                cabinet_dimensions=cabinet_t,
                num_drawers=n_drawers,
                wall_thickness=wall_thickness,
                drawer_gap=drawer_gap,
                drawer_travel=travel,
                handle_radius=handle_radius,
                handle_length=handle_length,
                handle_shape=shape,
                handle_box_size=box_size_t,
                handle_standoff=standoff,
                drawer_joint_range=drawer_range_t,
                rgba=rgba_t,
                frame_rgba=frame_rgba_t,
            )

        cfg = EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=pos_t,
                rot=rot_t,
                joint_pos=dict(init_joint_pos),
                joint_vel={".*": 0.0},
            ),
            spec_fn=spec_fn,
            articulation=EntityArticulationInfoCfg(
                actuators=(
                    BuiltinPdActuatorCfg(
                        target_names_expr=("drawer_.*_joint",),
                        effort_limit=80.0,
                        stiffness=0.0,
                        damping=4.0,
                        armature=0.01,
                        frictionloss=0.01,
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
            joint_names_simulation=list(joint_names),
            body_names_simulation=list(body_names),
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")

    behaviors: list = []
    if attach_drawer:
        behaviors.append(DrawerBehavior())
    if attach_grasp:
        behaviors.append(
            GraspPose.for_drawer_handle(
                handle_radius=float(handle_radius),
                handle_shape=shape,
                handle_box_size=box_size_t,
                handle_standoff=standoff,
            )
        )
    return AssetSpec(config=cfg, behaviors=tuple(behaviors))


# ---------------------------------------------------------------------------
# Legacy USD-file dummy props (Isaac-only for now)
# ---------------------------------------------------------------------------



registry.register("asset", "dummy_table", make_table)
registry.register("asset", "dummy_chair", make_chair)
registry.register("asset", "dummy_door", make_door)
registry.register("asset", "dummy_drawer", make_drawer)

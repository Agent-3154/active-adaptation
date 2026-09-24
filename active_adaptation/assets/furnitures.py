"""Procedural furniture assets (table, chair, stand, door, drawer).

Non-articulated (``table``, ``chair``, ``stand``): single rigid body with
multiple collision geoms. Articulated (``door``, ``drawer``): fixed-base
multi-body MJCF → Isaac articulated USD / mjlab ``EntityCfg.spec_fn``.

Shared USD / MjSpec helpers live in ``_procedural``.
"""
from __future__ import annotations

import math
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


def _chair_inertial(
    *,
    mass: float,
    leg_length: float,
    seat_size: Sequence[float],
    back_height: float,
    back_thickness: float,
    leg_inset: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Hollow-frame chair inertial: one rigid body, CoM not from solid slabs.

    Mass split is legs / seat / back ``(0.70, 0.22, 0.08)`` so most of the
    weight is in the posts. Returns ``(com_xyz, diag_inertia)`` in the body
    frame (origin on the floor at the seat center).
    """
    m = float(mass)
    if m <= 0.0:
        raise ValueError(f"chair mass must be positive, got {mass}")
    sx, sy, sz = _as_float_tuple(seat_size, 3)
    L = float(leg_length)
    bh = float(back_height)
    bt = float(back_thickness)
    hx, hy = 0.5 * sx, 0.5 * sy
    m_legs, m_seat, m_back = 0.70 * m, 0.22 * m, 0.08 * m
    z_legs = 0.5 * L
    z_seat = L + 0.5 * sz
    z_back = L + sz + 0.5 * bh
    y_back = -(hy - 0.5 * bt)
    com_y = (m_back * y_back) / m
    com_z = (m_legs * z_legs + m_seat * z_seat + m_back * z_back) / m
    com = (0.0, com_y, com_z)

    ixx = iyy = izz = 0.0
    cy, cz = com[1], com[2]
    lx = max(hx - float(leg_inset), 0.0)
    ly = max(hy - float(leg_inset), 0.0)
    for x, y in ((lx, ly), (lx, -ly), (-lx, ly), (-lx, -ly)):
        mm = m_legs / 4.0
        dx, dy, dz = x, y - cy, z_legs - cz
        ixx += mm * (dy * dy + dz * dz)
        iyy += mm * (dx * dx + dz * dz)
        izz += mm * (dx * dx + dy * dy)
    dx, dy, dz = 0.0, -cy, z_seat - cz
    ixx += m_seat * ((sy * sy + sz * sz) / 12.0 + dy * dy + dz * dz)
    iyy += m_seat * ((sx * sx + sz * sz) / 12.0 + dx * dx + dz * dz)
    izz += m_seat * ((sx * sx + sy * sy) / 12.0 + dx * dx + dy * dy)
    dx, dy, dz = 0.0, y_back - cy, z_back - cz
    ixx += m_back * ((bt * bt + bh * bh) / 12.0 + dy * dy + dz * dz)
    iyy += m_back * ((sx * sx + bh * bh) / 12.0 + dx * dx + dz * dz)
    izz += m_back * ((sx * sx + bt * bt) / 12.0 + dx * dx + dy * dy)
    return com, (ixx, iyy, izz)


def _apply_chair_body_inertial(body, *, mass: float, com, inertia) -> None:
    body.mass = float(mass)
    body.ipos = [float(com[0]), float(com[1]), float(com[2])]
    body.inertia = [float(inertia[0]), float(inertia[1]), float(inertia[2])]


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
        com, inertia = _chair_inertial(
            mass=mass,
            leg_length=leg_length,
            seat_size=seat_size,
            back_height=back_height,
            back_thickness=back_thickness,
            leg_inset=leg_inset,
        )
        _apply_chair_body_inertial(body, mass=mass, com=com, inertia=inertia)

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


def _stand_dims(height: float, radius: float) -> tuple[float, float, float, float, float]:
    """Pole height/radius plus derived tripod hub, leg radius, and foot span."""
    h = float(height)
    r = float(radius)
    if h <= 0.0 or r <= 0.0:
        raise ValueError(
            f"stand height and radius must be positive, got {height}, {radius}"
        )
    leg_r = max(0.65 * r, min(r, 0.012))
    hub_z = min(max(8.0 * r, 0.06), 0.22 * h)
    if h - r <= hub_z + r:
        raise ValueError(
            f"stand height ({h}) is too small relative to radius ({r}); "
            "need room for the pole above the tripod hub"
        )
    span = max(6.0 * r, 0.20 * h, 0.08)
    return h, r, leg_r, hub_z, span


def build_stand_spec(
    *,
    height: float = 1.0,
    radius: float = 0.015,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    mass: float | None = 0.5,
    body_name: str = "stand",
):
    """Tripod stand: vertical capsule pole + three splayed capsule legs.

    Origin at floor center. ``height`` is the top of the pole (including the
    upper hemisphere). ``radius`` is the pole radius. Legs are slightly thinner
    and rest their lower hemispheres on ``z=0``. Capsules only — Isaac cylinders
    are too expensive.
    """
    import mujoco

    h, r, leg_r, hub_z, span = _stand_dims(height, radius)
    rgba_t = _rgba(rgba)

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=body_name)
    if mass is not None:
        body.mass = float(mass)

    _add_capsule_leg(
        body,
        name=f"{body_name}_body_collision",
        radius=r,
        fromto=[0.0, 0.0, hub_z, 0.0, 0.0, h - r],
        rgba=rgba_t,
    )
    for i in range(3):
        ang = math.radians(90.0 + 120.0 * i)
        c, s = math.cos(ang), math.sin(ang)
        _add_capsule_leg(
            body,
            name=f"{body_name}_leg{i}_collision",
            radius=leg_r,
            fromto=[r * c, r * s, hub_z, span * c, span * s, leg_r],
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
    handle_standoff: float = 0.0,
    door_joint_range: tuple[float, float] = (-1.8, 1.8),
    door_slide_range: tuple[float, float] | None = None,
    handle_joint_range: tuple[float, float] = (-1.6, 1.6),
    rgba: Sequence[float] = _DEFAULT_RGBA,
    frame_rgba: Sequence[float] = _DEFAULT_DOOR_FRAME_RGBA,
    body_name: str = "door",
):
    """Articulated door: fixed ``frame`` → hinged ``panel`` → hinged ``handle``.

    Joints
    ------
    - ``door_slide_joint``: prismatic along **+X** (panel translates in its plane).
      Positive travel opens toward the latch, away from the left hinge.
    - ``door_joint``: revolute about **+Z** at the left (-X) edge of the panel.
    - ``handle_joint``: revolute about **+Y** (through the door) at the handle.
      ``q = 0`` is a horizontal bar; ``q = +π/2`` stands the bar vertical.

    Object / frame frame: origin at floor under the opening center; **+Z** up,
    **+Y** through the door, **+X** along the width. ``door_dimensions`` is full
    ``(width, thickness, height)``. ``handle_position`` is ``(x, z)`` in the
    **frame** frame (same as the closed panel face). No freejoint — fixed-base
    fixture (mjlab auto-mocap / Isaac ``fix_root_link`` on ``frame``).

    ``handle_shape`` is ``"capsule"`` (default) or ``"box"`` (thin bar). For
    boxes, ``handle_box_size`` is full ``(depth_y, height_z)``; omitted → derived
    from ``handle_radius``. ``handle_standoff`` is the gap along **±Y** between
    each panel face and the inner face of that side's handle bar.
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
    standoff = float(handle_standoff)
    if standoff < 0.0:
        raise ValueError(f"handle_standoff must be >= 0, got {handle_standoff}")

    half_w, half_d, half_h = width * 0.5, thickness * 0.5, height * 0.5
    half_ft = ft * 0.5
    hinge_x = -half_w
    if door_slide_range is None:
        slide_lo, slide_hi = 0.0, width
    else:
        slide_lo, slide_hi = (float(door_slide_range[0]), float(door_slide_range[1]))
    if slide_hi <= slide_lo:
        raise ValueError(
            f"door_slide_range must have hi > lo, got {(slide_lo, slide_hi)}"
        )

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

    # Carriage slides in the door plane; the panel hinges on that carriage.
    # Pull/push lock the slide at 0. Slide mode locks the hinge at 0.
    carriage = frame.add_body(name="carriage", pos=(hinge_x, 0.0, 0.0))
    carriage.mass = 0.2
    carriage.inertia = [0.01, 0.01, 0.01]
    slide_joint = carriage.add_joint(
        name="door_slide_joint",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1.0, 0.0, 0.0],
    )
    slide_joint.range = [slide_lo, slide_hi]

    panel = carriage.add_body(name="panel", pos=(0.0, 0.0, 0.0))
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
        y = y_sign * (half_d + standoff + y_half)
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


DOOR_JOINT_NAMES_SIMULATION = ["door_slide_joint", "door_joint", "handle_joint"]
DOOR_BODY_NAMES_SIMULATION = ["frame", "carriage", "panel", "handle"]
DOOR_INIT_JOINT_POS = {
    "door_slide_joint": 0.0,
    "door_joint": 0.0,
    "handle_joint": 0.0,
}


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
        elif cfg.kind == "stand":
            spec = build_stand_spec(
                height=cfg.height,
                radius=cfg.radius,
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
        if (
            cfg.kind == "chair"
            and cfg.mass_props is not None
            and cfg.mass_props.mass is not None
        ):
            from pxr import Gf, UsdPhysics

            com, inertia = _chair_inertial(
                mass=cfg.mass_props.mass,
                leg_length=cfg.leg_length,
                seat_size=cfg.seat_size,
                back_height=cfg.back_height,
                back_thickness=cfg.back_thickness,
                leg_inset=cfg.leg_inset,
            )
            mass_api = UsdPhysics.MassAPI.Apply(root)
            mass_api.CreateMassAttr(float(cfg.mass_props.mass))
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
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
        height: float = 1.0
        radius: float = 0.015
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


def make_dummy_stand(
    backend: Backend,
    height: float = 1.0,
    radius: float = 0.015,
    mass: float = 0.5,
    rgba: Sequence[float] = _DEFAULT_RGBA,
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    collision_only: bool = False,
    activate_contact_sensors: bool = True,
    attach_grasp: bool = True,
    name: str = "stand",
):
    """Procedural tripod stand (capsule pole + three splayed capsule legs).

    Replaces the USD ``dummy_stand`` (capsule on a cube base). ``height`` is the
    top of the pole; ``radius`` is the pole radius. Returns ``AssetSpec`` with a
    side-grasp on the pole (disable with ``attach_grasp=False``).
    """
    from active_adaptation.assets.asset_cfg import AssetSpec
    from active_adaptation.envs.behaviors.grasp_pose import GraspPose

    pos_t = _as_float_tuple(pos, 3)
    rot_t = _as_float_tuple(rot, 4)
    rgba_t = _rgba(rgba)
    h, r, _, hub_z, _ = _stand_dims(height, radius)
    grasp_z = 0.5 * (hub_z + (h - r))

    if backend == "isaaclab":
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        ProceduralFurnitureCfg = _get_furniture_spawner_cls()
        spawn = ProceduralFurnitureCfg(
            kind="stand",
            body_name=name,
            height=h,
            radius=r,
            rgba=rgba_t,
            **_isaac_spawn_kwargs(
                mass=mass,
                collision_only=collision_only,
                activate_contact_sensors=activate_contact_sensors,
            ),
        )
        if collision_only:
            cfg = AssetBaseCfg(
                spawn=spawn,
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos_t, rot=rot_t),
            )
        else:
            cfg = RigidObjectCfg(
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos_t, rot=rot_t),
            )
    elif backend == "mjlab":
        from active_adaptation.assets.asset_cfg import EntityCfg
        from mjlab.utils.spec_config import CollisionCfg

        def spec_fn():
            spec = build_stand_spec(
                height=h,
                radius=r,
                rgba=rgba_t,
                mass=None if collision_only else mass,
                body_name=name,
            )
            if not collision_only:
                body = next(iter(spec.worldbody.find_all("body")))
                body.add_freejoint(name=f"{name}_joint")
            return spec

        cfg = EntityCfg(
            init_state=EntityCfg.InitialStateCfg(pos=pos_t, rot=rot_t),
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
    else:
        raise ValueError(f"Invalid backend: {backend}")

    behaviors = ()
    if attach_grasp:
        behaviors = (
            GraspPose.for_side_axis(
                axis=(0.0, 0.0, 1.0),
                approach_dirs=(
                    (1.0, 0.0, 0.0),
                    (-1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, -1.0, 0.0),
                ),
                pos=(0.0, 0.0, grasp_z),
            ),
        )
    return AssetSpec(config=cfg, behaviors=behaviors)


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

    Collision stays a single rigid body. Mass is a hollow-frame split (mostly
    legs) so the CoM sits low instead of at the solid seat/back centroid.
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
            handle_standoff=cfg.handle_standoff,
            door_joint_range=cfg.door_joint_range,
            door_slide_range=cfg.door_slide_range,
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
        handle_standoff: float = 0.0
        door_joint_range: tuple[float, float] = (-1.8, 1.8)
        door_slide_range: tuple[float, float] | None = None
        handle_joint_range: tuple[float, float] = (-1.6, 1.6)
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
    handle_standoff: float = 0.0,
    door_joint_range: Sequence[float] = (-1.8, 1.8),
    door_slide_range: Sequence[float] | None = None,
    handle_joint_range: Sequence[float] = (-1.6, 1.6),
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
    """Articulated door: ``frame`` —slide→ ``carriage`` —hinge→ ``panel`` —handle→ ``handle``.

    Fixed-base fixture (no freejoint). Isaac welds ``frame`` with
    ``fix_root_link=True``; mjlab auto-wraps a mocap root. ``door_dimensions``
    is full ``(width, thickness, height)``. ``handle_position`` is ``(x, z)``
    in the frame frame. ``handle_standoff`` is the panel-face to handle-bar
    gap (both sides). ``rgba`` is panel/handle; ``frame_rgba`` is jambs/lintel.

    ``door_slide_joint`` translates the panel along **+X** (positive opens
    toward the latch). ``door_joint`` is the hinge. Both default to **zero
    stiffness**. ``handle_joint`` spans at least ``±π/2`` so slide mode can
    stand the bar vertical.

    ``open_direction`` is ``"pull"`` (hinge toward +Y), ``"push"`` (hinge
    toward −Y), or ``"slide"`` (translate along +X; handle held vertical and
    the slide is never locked). A task can set this per env.

    Behaviors (disable with flags):
    - ``DoorBehavior`` (``door.door``): lock / pull / push / slide
    - ``GraspPose`` (``door.grasp``): prescribed handle-face grasps
    """
    from active_adaptation.assets.asset_cfg import AssetSpec
    from active_adaptation.envs.behaviors.door import DoorBehavior
    from active_adaptation.envs.behaviors.grasp_pose import GraspPose

    door_dimensions_t = _as_float_tuple(door_dimensions, 3)
    handle_position_t = _as_float_tuple(handle_position, 2)
    door_range_t = _as_float_tuple(door_joint_range, 2)
    slide_range_t = (
        None if door_slide_range is None else _as_float_tuple(door_slide_range, 2)
    )
    handle_range_t = _as_float_tuple(handle_joint_range, 2)
    pos_t = _as_float_tuple(pos, 3)
    rot_t = _as_float_tuple(rot, 4)
    rgba_t = _rgba(rgba)
    frame_rgba_t = _rgba(frame_rgba)
    shape = _parse_handle_shape(handle_shape)
    box_size_t = (
        None if handle_box_size is None else _as_float_tuple(handle_box_size, 2)
    )
    standoff = float(handle_standoff)
    if standoff < 0.0:
        raise ValueError(f"handle_standoff must be >= 0, got {handle_standoff}")
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
            handle_standoff=standoff,
            door_joint_range=door_range_t,
            door_slide_range=slide_range_t,
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
                "slide": ImplicitActuatorCfg(
                    joint_names_expr=["door_slide_joint"],
                    effort_limit_sim=80.0,
                    stiffness=0.0,
                    damping=4.0,
                    armature=0.01,
                    friction=0.01,
                ),
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
                handle_standoff=standoff,
                door_joint_range=door_range_t,
                door_slide_range=slide_range_t,
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
                        target_names_expr=("door_slide_joint",),
                        effort_limit=80.0,
                        stiffness=0.0,
                        damping=4.0,
                        armature=0.01,
                        frictionloss=0.01,
                    ),
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
                handle_standoff=standoff,
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


registry.register("asset", "dummy_table", make_table)
registry.register("asset", "dummy_chair", make_chair)
registry.register("asset", "dummy_stand", make_dummy_stand)
registry.register("asset", "dummy_door", make_door)
registry.register("asset", "dummy_drawer", make_drawer)

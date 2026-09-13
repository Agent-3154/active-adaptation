"""Procedural grasp practice board (fixed/kinematic or XY-slidable).

Panel + box bars on **+Y**, capsule bars on **−Y**. With ``slidable=true``::

    frame ─board_joint_x→ carriage ─board_joint_y→ board

Isaac pins the articulation root via ``fix_root_link=True``.
"""

from __future__ import annotations

from typing import Any, Sequence

from active_adaptation.assets._procedural import (
    Backend,
    HandleShape,
    _DEFAULT_POS,
    _DEFAULT_ROT,
    _add_capsule_leg,
    _as_float_tuple,
    _handle_box_half_extents,
    _isaac_spawn_kwargs,
    _rgba,
    _usd_from_mjspec_articulated,
    _usd_from_mjspec_rigid,
    registry,
)

_DEFAULT_GRASP_BOARD_PANEL_RGBA = (0.28, 0.30, 0.34, 1.0)
_DEFAULT_GRASP_BOARD_BOX_RGBA = (0.55, 0.42, 0.28, 1.0)
_DEFAULT_GRASP_BOARD_CAPSULE_RGBA = (0.42, 0.55, 0.48, 1.0)

# Articulated slide chain: frame ─board_joint_x→ carriage ─board_joint_y→ board
GRASP_BOARD_SLIDE_JOINT_NAMES: tuple[str, ...] = ("board_joint_x", "board_joint_y")
GRASP_BOARD_SLIDE_BODY_NAMES: tuple[str, ...] = ("frame", "carriage", "board")
GRASP_BOARD_SLIDE_INIT_JOINT_POS: dict[str, float] = {
    "board_joint_x": 0.0,
    "board_joint_y": 0.0,
}
_DEFAULT_GRASP_BOARD_SLIDE_RANGE: tuple[float, float] = (-0.25, 0.25)


def build_grasp_board_spec(
    *,
    panel_size: Sequence[float] = (1.1, 0.04, 1.0),
    handle_length: float = 0.16,
    handle_radius: float = 0.022,
    handle_box_size: Sequence[float] | None = (0.04, 0.03),
    bar_standoffs: Sequence[float] | None = None,
    standoff_range: Sequence[float] | None = None,
    n_levels: int = 3,
    panel_rgba: Sequence[float] = _DEFAULT_GRASP_BOARD_PANEL_RGBA,
    box_rgba: Sequence[float] = _DEFAULT_GRASP_BOARD_BOX_RGBA,
    capsule_rgba: Sequence[float] = _DEFAULT_GRASP_BOARD_CAPSULE_RGBA,
    mass: float | None = None,
    body_name: str = "board",
    slidable: bool = False,
    slide_range: Sequence[float] = _DEFAULT_GRASP_BOARD_SLIDE_RANGE,
):
    """Grasp practice board: panel + box bars (+Y) + capsule bars (−Y).

    Object frame (``board`` body): origin at floor under panel center; **+Z** up,
    **+X** along width, **+Y** through the panel (box face).
    ``panel_size`` is full ``(width, thickness, height)``.

    Each face gets an ``n_levels`` × 4 grid (``grasp_board_bar_specs``):
    equally spaced Z rows (default 3 at 20% / 50% / 80% of height) ×
    horizontal, vertical, −45°, +45° (columns spaced in **X**).

    ``bar_standoffs`` (length ``n_levels * 4``) sets panel→bar gap per
    column; both faces share the list. If omitted, samples from
    ``standoff_range``.

    If ``slidable``, builds a fixed-base chain (one joint per body for Isaac USD)::

        frame ─board_joint_x (slide +X)─▶ carriage ─board_joint_y (slide +Y)─▶ board

    Geoms live on ``board``. ``slide_range`` is ``[lo, hi]`` meters for both axes.
    """
    import math
    import mujoco

    from active_adaptation.envs.behaviors.grasp_pose import (
        grasp_board_bar_specs,
        parse_grasp_board_n_levels,
        sample_grasp_board_bar_standoffs,
    )

    width, thickness, height = _as_float_tuple(panel_size, 3)
    hl = float(handle_length)
    hr = float(handle_radius)
    if width <= 0 or thickness <= 0 or height <= 0:
        raise ValueError(f"panel_size must be positive, got {panel_size}")
    if hl <= 0 or hr <= 0:
        raise ValueError("handle_length and handle_radius must be positive")

    half_w, half_t, half_h = width * 0.5, thickness * 0.5, height * 0.5
    hx, hy, hz = _handle_box_half_extents(
        handle_length=hl, handle_radius=hr, handle_box_size=handle_box_size
    )
    # Vertical box: thin cross-section in X, protrusion in Y, length in Z.
    v_half_x = hz
    v_half_y = hy
    v_half_z = 0.5 * hl

    panel_rgba_t = _rgba(panel_rgba)
    box_rgba_t = _rgba(box_rgba)
    cap_rgba_t = _rgba(capsule_rgba)

    half_a = 0.25 * math.pi  # 45°
    qw = math.cos(0.5 * half_a)
    qy = math.sin(0.5 * half_a)
    # R_y(θ) maps local +X → (cos θ, 0, −sin θ). Match grasp_board_bar_specs axes
    # and capsule fromto: m45=(+X,−Z), p45=(+X,+Z).
    quat_for_m45 = (qw, 0.0, qy, 0.0)    # +45° about Y → (+X, −Z)
    quat_for_p45 = (qw, 0.0, -qy, 0.0)   # −45° about Y → (+X, +Z)
    half_hl = 0.5 * hl
    diag = half_hl / math.sqrt(2.0)

    bars = grasp_board_bar_specs(
        (width, thickness, height),
        n_levels=parse_grasp_board_n_levels(n_levels),
    )
    if bar_standoffs is None:
        sos = sample_grasp_board_bar_standoffs(standoff_range, n=len(bars))
    else:
        sos = tuple(float(x) for x in bar_standoffs)
        if len(sos) != len(bars):
            raise ValueError(
                f"bar_standoffs length {len(sos)} != bars {len(bars)}"
            )
        if any(s < 0.0 for s in sos):
            raise ValueError(f"bar_standoffs must be >= 0, got {sos}")

    slide_lo, slide_hi = _as_float_tuple(slide_range, 2)
    if slide_hi < slide_lo:
        slide_lo, slide_hi = slide_hi, slide_lo

    board_mass = 10.0 if mass is None else float(mass)

    spec = mujoco.MjSpec()
    if slidable:
        if body_name != "board":
            raise ValueError(
                "slidable grasp board requires body_name='board' "
                f"(got {body_name!r})"
            )
        frame = spec.worldbody.add_body(name="frame")
        frame.mass = 20.0
        frame.inertia = [1.0, 1.0, 1.0]
        carriage = frame.add_body(name="carriage", pos=(0.0, 0.0, 0.0))
        carriage.mass = 1.0
        carriage.inertia = [0.05, 0.05, 0.05]
        jx = carriage.add_joint(
            name="board_joint_x",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1.0, 0.0, 0.0],
        )
        jx.range = [slide_lo, slide_hi]
        body = carriage.add_body(name="board", pos=(0.0, 0.0, 0.0))
        body.mass = board_mass
        body.inertia = [1.0, 1.0, 1.0]
        jy = body.add_joint(
            name="board_joint_y",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 1.0, 0.0],
        )
        jy.range = [slide_lo, slide_hi]
        geom_body_name = "board"
    else:
        body = spec.worldbody.add_body(name=body_name)
        if mass is not None:
            body.mass = float(mass)
        geom_body_name = body_name

    body.add_geom(
        name=f"{geom_body_name}_panel_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_w, half_t, half_h),
        pos=(0.0, 0.0, half_h),
        rgba=panel_rgba_t,
    )

    def _add_box_bar(*, name: str, size, pos, quat=None, rgba=None) -> None:
        geom = body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(size),
            pos=list(pos),
            rgba=rgba,
        )
        if quat is not None:
            geom.quat = list(quat)

    def _add_face(*, y_sign: float, shape: HandleShape, rgba) -> None:
        side = "box" if shape == "box" else "capsule"
        y_half = hy if shape == "box" else hr

        for so, (tag, x0, z0, _axis) in zip(sos, bars, strict=True):
            y_center = y_sign * (half_t + so + y_half)
            name = f"{geom_body_name}_{side}_{tag}_collision"
            if tag.endswith("_h"):
                if shape == "box":
                    _add_box_bar(
                        name=name,
                        size=(hx, hy, hz),
                        pos=(x0, y_center, z0),
                        rgba=rgba,
                    )
                else:
                    _add_capsule_leg(
                        body,
                        name=name,
                        radius=hr,
                        fromto=[
                            x0 - half_hl, y_center, z0,
                            x0 + half_hl, y_center, z0,
                        ],
                        rgba=rgba,
                    )
            elif tag.endswith("_v"):
                if shape == "box":
                    _add_box_bar(
                        name=name,
                        size=(v_half_x, v_half_y, v_half_z),
                        pos=(x0, y_center, z0),
                        rgba=rgba,
                    )
                else:
                    _add_capsule_leg(
                        body,
                        name=name,
                        radius=hr,
                        fromto=[
                            x0, y_center, z0 - half_hl,
                            x0, y_center, z0 + half_hl,
                        ],
                        rgba=rgba,
                    )
            elif tag.endswith("_m45"):
                if shape == "box":
                    _add_box_bar(
                        name=name,
                        size=(hx, hy, hz),
                        pos=(x0, y_center, z0),
                        quat=quat_for_m45,
                        rgba=rgba,
                    )
                else:
                    _add_capsule_leg(
                        body,
                        name=name,
                        radius=hr,
                        fromto=[
                            x0 - diag, y_center, z0 + diag,  # (−X,+Z) → (+X,−Z)
                            x0 + diag, y_center, z0 - diag,
                        ],
                        rgba=rgba,
                    )
            elif tag.endswith("_p45"):
                if shape == "box":
                    _add_box_bar(
                        name=name,
                        size=(hx, hy, hz),
                        pos=(x0, y_center, z0),
                        quat=quat_for_p45,
                        rgba=rgba,
                    )
                else:
                    _add_capsule_leg(
                        body,
                        name=name,
                        radius=hr,
                        fromto=[
                            x0 - diag, y_center, z0 - diag,
                            x0 + diag, y_center, z0 + diag,
                        ],
                        rgba=rgba,
                    )
            else:
                raise ValueError(f"Unknown grasp-board bar tag: {tag!r}")

    _add_face(y_sign=+1.0, shape="box", rgba=box_rgba_t)
    _add_face(y_sign=-1.0, shape="capsule", rgba=cap_rgba_t)
    return spec


def make_grasp_board(
    backend: Backend,
    panel_size: Sequence[float] = (1.1, 0.04, 1.0),
    handle_length: float = 0.16,
    handle_radius: float = 0.022,
    handle_box_size: Sequence[float] | None = (0.04, 0.03),
    n_levels: int = 3,
    standoff_range: Sequence[float] | None = None,
    grasp_clearance: float = 0.02,
    panel_rgba: Sequence[float] = _DEFAULT_GRASP_BOARD_PANEL_RGBA,
    box_rgba: Sequence[float] = _DEFAULT_GRASP_BOARD_BOX_RGBA,
    capsule_rgba: Sequence[float] = _DEFAULT_GRASP_BOARD_CAPSULE_RGBA,
    mass: float = 10.0,
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    collision_only: bool = True,
    activate_contact_sensors: bool = True,
    attach_grasp: bool = True,
    slidable: bool = False,
    slide_range: Sequence[float] = _DEFAULT_GRASP_BOARD_SLIDE_RANGE,
    slide_stiffness: float = 0.0,
    slide_damping: float = 2.0,
    name: str = "board",
):
    """Grasp practice board (panel + box bars on +Y, capsules on −Y).

    - ``slidable=False`` (default): fixed / kinematic single body.
      ``collision_only=True`` → Isaac ``AssetBaseCfg``; ``False`` → kinematic
      ``RigidObjectCfg``.
    - ``slidable=True``: fixed-base articulation
      ``frame ─board_joint_x→ carriage ─board_joint_y→ board`` with Implicit /
      BuiltinPd actuators (default **zero stiffness**, light ``slide_damping``).
      Isaac sets ``fix_root_link=True`` so the board only slides in XY. Ignores
      ``collision_only``. ``slide_range`` is ``[lo, hi]`` meters for both axes.

    Returns ``AssetSpec`` with ``GraspPose`` (``board.grasp``). Grasps are in
    the ``board`` body frame (follow the slide when articulated).
    ``n_levels`` is the number of equally spaced Z rows (default 3).
    """
    from active_adaptation.assets.asset_cfg import AssetSpec
    from active_adaptation.envs.behaviors.grasp_pose import (
        GraspPose,
        grasp_board_bars_per_face,
        parse_grasp_board_n_levels,
        sample_grasp_board_bar_standoffs,
    )

    panel_t = _as_float_tuple(panel_size, 3)
    box_size_t = (
        None if handle_box_size is None else _as_float_tuple(handle_box_size, 2)
    )
    pos_t = _as_float_tuple(pos, 3)
    rot_t = _as_float_tuple(rot, 4)
    panel_rgba_t = _rgba(panel_rgba)
    box_rgba_t = _rgba(box_rgba)
    cap_rgba_t = _rgba(capsule_rgba)
    n_levels_i = parse_grasp_board_n_levels(n_levels)
    bar_sos = sample_grasp_board_bar_standoffs(
        standoff_range, n=grasp_board_bars_per_face(n_levels_i)
    )
    slide_range_t = _as_float_tuple(slide_range, 2)
    slidable = bool(slidable)
    if slidable and name != "board":
        raise ValueError(
            "slidable grasp board requires name='board' for body/joint names "
            f"(got {name!r})"
        )

    if slidable and backend == "isaaclab":
        import isaaclab.sim as sim_utils
        from active_adaptation.assets.asset_cfg import (
            ArticulationCfg,
            ImplicitActuatorCfg,
        )

        ProceduralGraspBoardCfg = _get_grasp_board_spawner_cls()
        spawn = ProceduralGraspBoardCfg(
            panel_size=panel_t,
            handle_length=float(handle_length),
            handle_radius=float(handle_radius),
            handle_box_size=box_size_t,
            bar_standoffs=bar_sos,
            n_levels=n_levels_i,
            slide_range=slide_range_t,
            panel_rgba=panel_rgba_t,
            box_rgba=box_rgba_t,
            capsule_rgba=cap_rgba_t,
            mass=float(mass),
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
                joint_pos=dict(GRASP_BOARD_SLIDE_INIT_JOINT_POS),
                joint_vel={".*": 0.0},
            ),
            actuators={
                "slide": ImplicitActuatorCfg(
                    joint_names_expr=["board_joint_.*"],
                    effort_limit_sim=80.0,
                    stiffness=float(slide_stiffness),
                    damping=float(slide_damping),
                    armature=0.01,
                    friction=0.0,
                ),
            },
            joint_names_simulation=list(GRASP_BOARD_SLIDE_JOINT_NAMES),
            body_names_simulation=list(GRASP_BOARD_SLIDE_BODY_NAMES),
        )
    elif slidable and backend == "mjlab":
        from active_adaptation.assets.asset_cfg import EntityCfg
        from mjlab.actuator import BuiltinPdActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg
        from mjlab.utils.spec_config import CollisionCfg

        def spec_fn_slide():
            return build_grasp_board_spec(
                panel_size=panel_t,
                handle_length=handle_length,
                handle_radius=handle_radius,
                handle_box_size=box_size_t,
                bar_standoffs=bar_sos,
                n_levels=n_levels_i,
                panel_rgba=panel_rgba_t,
                box_rgba=box_rgba_t,
                capsule_rgba=cap_rgba_t,
                mass=mass,
                body_name="board",
                slidable=True,
                slide_range=slide_range_t,
            )

        cfg = EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=pos_t,
                rot=rot_t,
                joint_pos=dict(GRASP_BOARD_SLIDE_INIT_JOINT_POS),
                joint_vel={".*": 0.0},
            ),
            spec_fn=spec_fn_slide,
            articulation=EntityArticulationInfoCfg(
                actuators=(
                    BuiltinPdActuatorCfg(
                        target_names_expr=("board_joint_.*",),
                        effort_limit=80.0,
                        stiffness=float(slide_stiffness),
                        damping=float(slide_damping),
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
            joint_names_simulation=list(GRASP_BOARD_SLIDE_JOINT_NAMES),
            body_names_simulation=list(GRASP_BOARD_SLIDE_BODY_NAMES),
        )
    elif backend == "isaaclab":
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        ProceduralGraspBoardRigidCfg = _get_grasp_board_rigid_spawner_cls()
        if collision_only:
            spawn_kw = _isaac_spawn_kwargs(
                mass=mass,
                collision_only=True,
                activate_contact_sensors=activate_contact_sensors,
            )
        else:
            spawn_kw = dict(
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.02,
                    rest_offset=0.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8,
                    dynamic_friction=0.8,
                    restitution=0.0,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                activate_contact_sensors=activate_contact_sensors,
            )
        spawn = ProceduralGraspBoardRigidCfg(
            body_name=name,
            panel_size=panel_t,
            handle_length=float(handle_length),
            handle_radius=float(handle_radius),
            handle_box_size=box_size_t,
            bar_standoffs=bar_sos,
            n_levels=n_levels_i,
            panel_rgba=panel_rgba_t,
            box_rgba=box_rgba_t,
            capsule_rgba=cap_rgba_t,
            **spawn_kw,
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
            return build_grasp_board_spec(
                panel_size=panel_t,
                handle_length=handle_length,
                handle_radius=handle_radius,
                handle_box_size=box_size_t,
                bar_standoffs=bar_sos,
                n_levels=n_levels_i,
                panel_rgba=panel_rgba_t,
                box_rgba=box_rgba_t,
                capsule_rgba=cap_rgba_t,
                mass=None if collision_only else mass,
                body_name=name,
                slidable=False,
            )

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

    behaviors: list = []
    if attach_grasp:
        behaviors.append(
            GraspPose.for_grasp_board(
                panel_size=panel_t,
                handle_length=float(handle_length),
                handle_radius=float(handle_radius),
                handle_box_size=box_size_t,
                bar_standoffs=bar_sos,
                grasp_clearance=float(grasp_clearance),
                n_levels=n_levels_i,
            )
        )
    return AssetSpec(config=cfg, behaviors=tuple(behaviors))



_GRASP_BOARD_RIGID_SPAWNER_CLS = None


def _get_grasp_board_rigid_spawner_cls():
    """Lazy Isaac RigidObject spawner for non-slidable grasp boards."""
    global _GRASP_BOARD_RIGID_SPAWNER_CLS
    if _GRASP_BOARD_RIGID_SPAWNER_CLS is not None:
        return _GRASP_BOARD_RIGID_SPAWNER_CLS

    from collections.abc import Callable

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab.utils import configclass
    from pxr import Usd

    @clone
    def spawn_grasp_board_rigid(
        prim_path: str,
        cfg: "ProceduralGraspBoardRigidCfg",
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> Usd.Prim:
        del kwargs
        stage = get_current_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            raise ValueError(f"A prim already exists at path: '{prim_path}'.")

        spec = build_grasp_board_spec(
            panel_size=cfg.panel_size,
            handle_length=cfg.handle_length,
            handle_radius=cfg.handle_radius,
            handle_box_size=cfg.handle_box_size,
            bar_standoffs=cfg.bar_standoffs,
            n_levels=cfg.n_levels,
            panel_rgba=cfg.panel_rgba,
            box_rgba=cfg.box_rgba,
            capsule_rgba=cfg.capsule_rgba,
            mass=None,
            body_name=cfg.body_name,
            slidable=False,
        )
        root = _usd_from_mjspec_rigid(stage, prim_path, spec)

        from isaaclab.sim.utils import bind_physics_material
        from pxr import Gf, UsdPhysics

        if translation is not None:
            root.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*translation))
        if orientation is not None:
            root.GetAttribute("xformOp:orient").Set(Gf.Quatf(*orientation))
        if cfg.collision_props is not None:
            geom_root = stage.GetPrimAtPath(f"{prim_path}/geometry")
            for child in geom_root.GetChildren():
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                schemas.define_collision_properties(
                    str(child.GetPath()), cfg.collision_props, stage=stage
                )
        if cfg.physics_material is not None:
            if not cfg.physics_material_path.startswith("/"):
                material_path = f"{prim_path}/{cfg.physics_material_path}"
            else:
                material_path = cfg.physics_material_path
            cfg.physics_material.func(material_path, cfg.physics_material)
            geom_root = stage.GetPrimAtPath(f"{prim_path}/geometry")
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
    class ProceduralGraspBoardRigidCfg(RigidObjectSpawnerCfg):
        func: Callable = spawn_grasp_board_rigid
        body_name: str = "board"
        panel_size: tuple[float, float, float] = (1.1, 0.04, 1.0)
        handle_length: float = 0.16
        handle_radius: float = 0.022
        handle_box_size: tuple[float, float] | None = (0.04, 0.03)
        bar_standoffs: tuple[float, ...] | None = None
        n_levels: int = 3
        panel_rgba: tuple[float, float, float, float] = _DEFAULT_GRASP_BOARD_PANEL_RGBA
        box_rgba: tuple[float, float, float, float] = _DEFAULT_GRASP_BOARD_BOX_RGBA
        capsule_rgba: tuple[float, float, float, float] = (
            _DEFAULT_GRASP_BOARD_CAPSULE_RGBA
        )
        physics_material_path: str = "material"
        physics_material: Any = None
        copy_from_source: bool = True

    _GRASP_BOARD_RIGID_SPAWNER_CLS = ProceduralGraspBoardRigidCfg
    return _GRASP_BOARD_RIGID_SPAWNER_CLS


_GRASP_BOARD_SPAWNER_CLS = None


def _get_grasp_board_spawner_cls():
    """Lazy Isaac spawner for the articulated (slidable) grasp board."""
    global _GRASP_BOARD_SPAWNER_CLS
    if _GRASP_BOARD_SPAWNER_CLS is not None:
        return _GRASP_BOARD_SPAWNER_CLS

    from collections.abc import Callable

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab.utils import configclass
    from pxr import Usd

    @clone
    def spawn_grasp_board(
        prim_path: str,
        cfg: "ProceduralGraspBoardCfg",
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> Usd.Prim:
        del kwargs
        stage = get_current_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            raise ValueError(f"A prim already exists at path: '{prim_path}'.")

        spec = build_grasp_board_spec(
            panel_size=cfg.panel_size,
            handle_length=cfg.handle_length,
            handle_radius=cfg.handle_radius,
            handle_box_size=cfg.handle_box_size,
            bar_standoffs=cfg.bar_standoffs,
            n_levels=cfg.n_levels,
            panel_rgba=cfg.panel_rgba,
            box_rgba=cfg.box_rgba,
            capsule_rgba=cfg.capsule_rgba,
            mass=cfg.mass,
            body_name="board",
            slidable=True,
            slide_range=cfg.slide_range,
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

        for body_name in GRASP_BOARD_SLIDE_BODY_NAMES:
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
            for body_name in GRASP_BOARD_SLIDE_BODY_NAMES:
                body_prim = stage.GetPrimAtPath(f"{prim_path}/{body_name}")
                if not body_prim.IsValid():
                    continue
                for child in body_prim.GetChildren():
                    if not child.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    bind_physics_material(
                        str(child.GetPath()), material_path, stage=stage
                    )

        # Isaac ``fix_root_link`` needs ArticulationRootAPI on a RigidBody.
        # ``_usd_from_mjspec_articulated`` puts it on the outer Xform; move it
        # onto ``frame`` so fix_root_link can weld that link to world (and then
        # re-homes ArticulationRootAPI on the parent ``prim_path``).
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
    class ProceduralGraspBoardCfg(SpawnerCfg):
        func: Callable = spawn_grasp_board
        panel_size: tuple[float, float, float] = (1.1, 0.04, 1.0)
        handle_length: float = 0.16
        handle_radius: float = 0.022
        handle_box_size: tuple[float, float] | None = (0.04, 0.03)
        bar_standoffs: tuple[float, ...] | None = None
        n_levels: int = 3
        slide_range: tuple[float, float] = _DEFAULT_GRASP_BOARD_SLIDE_RANGE
        panel_rgba: tuple[float, float, float, float] = _DEFAULT_GRASP_BOARD_PANEL_RGBA
        box_rgba: tuple[float, float, float, float] = _DEFAULT_GRASP_BOARD_BOX_RGBA
        capsule_rgba: tuple[float, float, float, float] = (
            _DEFAULT_GRASP_BOARD_CAPSULE_RGBA
        )
        mass: float = 10.0
        collision_props: Any = None
        physics_material_path: str = "material"
        physics_material: Any = None
        articulation_props: Any = None
        activate_contact_sensors: bool = True
        copy_from_source: bool = False

    _GRASP_BOARD_SPAWNER_CLS = ProceduralGraspBoardCfg
    return _GRASP_BOARD_SPAWNER_CLS


registry.register("asset", "dummy_grasp_board", make_grasp_board)

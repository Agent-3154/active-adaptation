"""Legacy USD-file dummy props (stand, basket), a procedural dumbbell, and re-exports.

Procedural furniture lives in ``furnitures``; the grasp board in ``grasp_board``.
Importing this module still registers those assets (via side-effect imports)
plus the legacy Isaac-only USD props below.

``dummy_dumbbell`` is a rigid handle capsule with larger end-caps so the
gripper cannot slide on along the handle axis.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.registry import Registry
from active_adaptation.assets._procedural import (
    _as_float_tuple,
    _rgba,
    _usd_from_mjspec_rigid,
)

# Register procedural assets.
from active_adaptation.assets import furnitures as _furnitures  # noqa: F401
from active_adaptation.assets import grasp_board as _grasp_board  # noqa: F401

# Backward-compatible re-exports.
from active_adaptation.assets.furnitures import (  # noqa: F401
    make_chair,
    make_door,
    make_drawer,
    make_table,
)
from active_adaptation.assets.grasp_board import (  # noqa: F401
    build_grasp_board_spec,
    make_grasp_board,
)

registry = Registry.instance()

Backend = Literal["isaaclab", "mjlab"]


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


# ---------------------------------------------------------------------------
# Dummy dumbbell (handle capsule + larger end-caps)
# ---------------------------------------------------------------------------

Axis = Literal["X", "Y", "Z"]
_AXIS_VEC: dict[str, tuple[float, float, float]] = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}
_DEFAULT_DUMBBELL_RGBA = (0.52, 0.48, 0.42, 1.0)
_DEFAULT_DUMBBELL_END_RGBA = (0.38, 0.36, 0.34, 1.0)


def _parse_axis(axis: str | Axis) -> str:
    key = str(axis).upper()
    if key not in _AXIS_VEC:
        raise ValueError(f"axis must be 'X', 'Y', or 'Z', got {axis!r}")
    return key


def _axis_fromto(axis: str, center: float, half: float) -> list[float]:
    vx, vy, vz = _AXIS_VEC[axis]
    lo, hi = center - half, center + half
    return [lo * vx, lo * vy, lo * vz, hi * vx, hi * vy, hi * vz]


def build_dummy_dumbbell_spec(
    *,
    handle_length: float = 0.30,
    handle_radius: float = 0.02,
    end_length: float = 0.05,
    end_radius: float = 0.06,
    axis: str | Axis = "Z",
    mass: float | None = 0.15,
    rgba: Sequence[float] = _DEFAULT_DUMBBELL_RGBA,
    end_rgba: Sequence[float] | None = None,
    body_name: str = "dumbbell",
    floating: bool = False,
    disable_gravity: bool = False,
):
    """Single rigid body: slender handle capsule + two larger end capsules.

    ``handle_length`` / ``end_length`` are cylindrical sections (USD capsule
    height). End-caps sit at ±handle_length/2 along ``axis`` and are fat
    enough to block axial insertion.
    """
    import mujoco

    axis_key = _parse_axis(axis)
    hl = float(handle_length)
    hr = float(handle_radius)
    el = float(end_length)
    er = float(end_radius)
    if hl <= 0.0 or hr <= 0.0 or el <= 0.0 or er <= 0.0:
        raise ValueError("dumbbell lengths and radii must be positive")
    if er <= hr:
        raise ValueError(
            f"end_radius ({er}) must be larger than handle_radius ({hr}) "
            "so the bells block axial insertion"
        )
    handle_rgba = _rgba(rgba)
    bell_rgba = _rgba(end_rgba) if end_rgba is not None else _rgba(
        _DEFAULT_DUMBBELL_END_RGBA
    )
    handle_half = 0.5 * hl
    end_half = 0.5 * el

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=body_name)
    if floating:
        if disable_gravity:
            body.gravcomp = 1.0
        body.add_freejoint(name=f"{body_name}_joint")
    geom_mass = None if mass is None else float(mass) / 3.0

    for name, fromto, radius, color in (
        (
            f"{body_name}_handle_collision",
            _axis_fromto(axis_key, 0.0, handle_half),
            hr,
            handle_rgba,
        ),
        (
            f"{body_name}_end0_collision",
            _axis_fromto(axis_key, -handle_half, end_half),
            er,
            bell_rgba,
        ),
        (
            f"{body_name}_end1_collision",
            _axis_fromto(axis_key, handle_half, end_half),
            er,
            bell_rgba,
        ),
    ):
        geom_kw = dict(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            rgba=color,
        )
        if geom_mass is not None:
            geom_kw["mass"] = geom_mass
        geom = body.add_geom(**geom_kw)
        geom.size = [radius, 0.0, 0.0]
        geom.fromto = fromto
    return spec


_DUMBBELL_SPAWNER_CLS = None


def _get_dummy_dumbbell_spawner_cls():
    global _DUMBBELL_SPAWNER_CLS
    if _DUMBBELL_SPAWNER_CLS is not None:
        return _DUMBBELL_SPAWNER_CLS

    from collections.abc import Callable

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab.utils import configclass
    from pxr import Usd

    @clone
    def spawn_dummy_dumbbell(
        prim_path: str,
        cfg: "ProceduralDummyDumbbellCfg",
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> Usd.Prim:
        del kwargs
        stage = get_current_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            raise ValueError(f"A prim already exists at path: '{prim_path}'.")

        spec = build_dummy_dumbbell_spec(
            handle_length=cfg.handle_length,
            handle_radius=cfg.handle_radius,
            end_length=cfg.end_length,
            end_radius=cfg.end_radius,
            axis=cfg.axis,
            mass=None,
            rgba=cfg.rgba,
            end_rgba=cfg.end_rgba,
            body_name=cfg.body_name,
            floating=False,
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
                bind_physics_material(
                    str(child.GetPath()), material_path, stage=stage
                )
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
        if cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_path, stage=stage)
        return root

    @configclass
    class ProceduralDummyDumbbellCfg(RigidObjectSpawnerCfg):
        func: Callable = spawn_dummy_dumbbell
        body_name: str = "dumbbell"
        handle_length: float = 0.30
        handle_radius: float = 0.02
        end_length: float = 0.05
        end_radius: float = 0.06
        axis: str = "Z"
        rgba: tuple[float, float, float, float] = _DEFAULT_DUMBBELL_RGBA
        end_rgba: tuple[float, float, float, float] = _DEFAULT_DUMBBELL_END_RGBA
        physics_material_path: str = "material"
        physics_material: Any = None
        copy_from_source: bool = False

    _DUMBBELL_SPAWNER_CLS = ProceduralDummyDumbbellCfg
    return _DUMBBELL_SPAWNER_CLS


def make_dummy_dumbbell(
    backend: Backend,
    handle_length: float = 0.30,
    handle_radius: float = 0.02,
    end_length: float = 0.05,
    end_radius: float = 0.06,
    axis: str | Axis = "Z",
    mass: float = 0.15,
    rgba: Sequence[float] = _DEFAULT_DUMBBELL_RGBA,
    end_rgba: Sequence[float] = _DEFAULT_DUMBBELL_END_RGBA,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    rot: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    disable_gravity: bool = False,
    activate_contact_sensors: bool = True,
    collision_only: bool = False,
    kinematic_enabled: bool = False,
):
    """Rigid dumbbell: grasp the thin handle; end-caps block axial insertion.

    Drop-in for the practice ``capsule`` primitive (same ``axis`` / ``pos``
    / ``kinematic_enabled`` knobs). ``end_radius`` must exceed
    ``handle_radius``.
    """
    if collision_only and kinematic_enabled:
        raise ValueError(
            "collision_only and kinematic_enabled cannot both be True: "
            "collision_only uses AssetBaseCfg (no RigidBodyAPI), while "
            "kinematic_enabled requires a RigidObject."
        )
    axis_key = _parse_axis(axis)
    pos_t = _as_float_tuple(pos, 3)
    rot_t = _as_float_tuple(rot, 4)
    rgba_t = _rgba(rgba)
    end_rgba_t = _rgba(end_rgba)
    hl = float(handle_length)
    hr = float(handle_radius)
    el = float(end_length)
    er = float(end_radius)

    if backend == "isaaclab":
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        ProceduralDummyDumbbellCfg = _get_dummy_dumbbell_spawner_cls()
        spawn_kw: dict = dict(
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )
        if not collision_only:
            spawn_kw.update(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=kinematic_enabled,
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
        spawn = ProceduralDummyDumbbellCfg(
            body_name="dumbbell",
            handle_length=hl,
            handle_radius=hr,
            end_length=el,
            end_radius=er,
            axis=axis_key,
            rgba=rgba_t,
            end_rgba=end_rgba_t,
            **spawn_kw,
        )
        if collision_only:
            return AssetBaseCfg(
                spawn=spawn,
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos_t, rot=rot_t),
            )
        return RigidObjectCfg(
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos_t, rot=rot_t),
        )

    if backend == "mjlab":
        from active_adaptation.assets.asset_cfg import EntityCfg
        from mjlab.utils.spec_config import CollisionCfg

        def spec_fn():
            return build_dummy_dumbbell_spec(
                handle_length=hl,
                handle_radius=hr,
                end_length=el,
                end_radius=er,
                axis=axis_key,
                mass=None if collision_only else mass,
                rgba=rgba_t,
                end_rgba=end_rgba_t,
                body_name="dumbbell",
                floating=not collision_only,
                disable_gravity=disable_gravity,
            )

        return EntityCfg(
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

    raise ValueError(f"Invalid backend: {backend}")


registry.register("asset", "dummy_stand", make_dummy_stand)
registry.register("asset", "dummy_basket", make_dummy_basket)
registry.register("asset", "dummy_basket_platform", make_dummy_basket_platform)
registry.register("asset", "dummy_dumbbell", make_dummy_dumbbell)

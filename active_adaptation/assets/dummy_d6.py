"""Fixed dummy base + 6-axis serial chain (D6) with PD drives and a bar handle.

Isaac USD only allows one joint per body, so the D6 is a serial chain::

    dummy ─d6_tx→ link_tx ─d6_ty→ link_ty ─d6_tz→ link_tz
          ─d6_rx→ link_rx ─d6_ry→ link_ry ─d6_rz→ handle

``dummy`` is a fixed / mocap root (Isaac ``fix_root_link``, mjlab no freejoint).
The handle is a capsule along **+X**; GraspPose is the bar midpoint.
"""

from __future__ import annotations

from typing import Any, Sequence

from active_adaptation.assets._procedural import (
    Backend,
    _DEFAULT_POS,
    _DEFAULT_ROT,
    _add_capsule_leg,
    _as_float_tuple,
    _rgba,
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
D6_BODY_NAMES: tuple[str, ...] = (
    "dummy",
    "link_tx",
    "link_ty",
    "link_tz",
    "link_rx",
    "link_ry",
    "handle",
)
D6_INIT_JOINT_POS: dict[str, float] = {name: 0.0 for name in D6_JOINT_NAMES}

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


def build_dummy_d6_spec(
    *,
    handle_length: float = 0.16,
    handle_radius: float = 0.022,
    handle_mass: float = 0.15,
    slide_range: Sequence[float] = (-0.12, 0.12),
    hinge_range: Sequence[float] = (-3.1416, 3.1416),
    rgba: Sequence[float] = (0.42, 0.55, 0.48, 1.0),
):
    """Fixed ``dummy`` + serial D6 + X-axis capsule ``handle`` at the origin."""
    import mujoco

    slide_lo, slide_hi = _as_float_tuple(slide_range, 2)
    hinge_lo, hinge_hi = _as_float_tuple(hinge_range, 2)
    rgba_t = _rgba(rgba)
    half = 0.5 * float(handle_length)

    spec = mujoco.MjSpec()
    dummy = spec.worldbody.add_body(name="dummy")
    dummy.mass = 5.0
    dummy.inertia = [0.05, 0.05, 0.05]

    parent = dummy
    for joint_name, body_name, axis in _SLIDE_AXES:
        child = parent.add_body(name=body_name, pos=(0.0, 0.0, 0.0))
        child.mass = 0.02
        child.inertia = [1e-4, 1e-4, 1e-4]
        joint = child.add_joint(
            name=joint_name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=list(axis),
        )
        joint.range = [slide_lo, slide_hi]
        parent = child

    for joint_name, body_name, axis in _HINGE_AXES:
        child = parent.add_body(name=body_name, pos=(0.0, 0.0, 0.0))
        if body_name == "handle":
            child.mass = float(handle_mass)
            child.inertia = [1e-3, 1e-3, 1e-3]
        else:
            child.mass = 0.02
            child.inertia = [1e-4, 1e-4, 1e-4]
        joint = child.add_joint(
            name=joint_name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=list(axis),
        )
        joint.range = [hinge_lo, hinge_hi]
        parent = child

    _add_capsule_leg(
        parent,
        name="handle_collision",
        radius=float(handle_radius),
        fromto=[-half, 0.0, 0.0, half, 0.0, 0.0],
        rgba=rgba_t,
    )
    return spec


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

        spec = build_dummy_d6_spec(
            handle_length=cfg.handle_length,
            handle_radius=cfg.handle_radius,
            handle_mass=cfg.handle_mass,
            slide_range=cfg.slide_range,
            hinge_range=cfg.hinge_range,
            rgba=cfg.rgba,
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

        for body_name in D6_BODY_NAMES:
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
            for body_name in D6_BODY_NAMES:
                body_prim = stage.GetPrimAtPath(f"{prim_path}/{body_name}")
                if not body_prim.IsValid():
                    continue
                for child in body_prim.GetChildren():
                    if not child.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    bind_physics_material(
                        str(child.GetPath()), material_path, stage=stage
                    )

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
        return root

    @configclass
    class ProceduralDummyD6Cfg(SpawnerCfg):
        func: Callable = spawn_dummy_d6
        handle_length: float = 0.16
        handle_radius: float = 0.022
        handle_mass: float = 0.15
        slide_range: tuple[float, float] = (-0.12, 0.12)
        hinge_range: tuple[float, float] = (-3.1416, 3.1416)
        rgba: tuple[float, float, float, float] = (0.42, 0.55, 0.48, 1.0)
        collision_props: Any = None
        physics_material_path: str = "material"
        physics_material: Any = None
        articulation_props: Any = None
        activate_contact_sensors: bool = True
        copy_from_source: bool = False

    _D6_SPAWNER_CLS = ProceduralDummyD6Cfg
    return _D6_SPAWNER_CLS


def make_dummy_d6(
    backend: Backend,
    handle_length: float = 0.16,
    handle_radius: float = 0.022,
    handle_mass: float = 0.15,
    slide_range: Sequence[float] = (-0.12, 0.12),
    hinge_range: Sequence[float] = (-3.1416, 3.1416),
    stiffness: float = 40.0,
    damping: float = 4.0,
    rgba: Sequence[float] = (0.42, 0.55, 0.48, 1.0),
    pos: Sequence[float] = _DEFAULT_POS,
    rot: Sequence[float] = _DEFAULT_ROT,
    activate_contact_sensors: bool = True,
    attach_grasp: bool = True,
):
    """Compliant D6 handle on a dummy base. Returns ``AssetSpec``."""
    from active_adaptation.assets.asset_cfg import AssetSpec
    from active_adaptation.envs.behaviors.grasp_pose import GraspPose

    pos_t = _as_float_tuple(pos, 3)
    rot_t = _as_float_tuple(rot, 4)
    rgba_t = _rgba(rgba)
    slide_t = _as_float_tuple(slide_range, 2)
    hinge_t = _as_float_tuple(hinge_range, 2)
    init_joint = dict(D6_INIT_JOINT_POS)

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
                solver_position_iteration_count=8,
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
                joint_pos=init_joint,
                joint_vel={".*": 0.0},
            ),
            actuators={
                "d6": ImplicitActuatorCfg(
                    joint_names_expr=["d6_.*"],
                    effort_limit_sim=50.0,
                    stiffness=float(stiffness),
                    damping=float(damping),
                    armature=0.01,
                    friction=0.0,
                ),
            },
            joint_names_simulation=list(D6_JOINT_NAMES),
            body_names_simulation=list(D6_BODY_NAMES),
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
                slide_range=slide_t,
                hinge_range=hinge_t,
                rgba=rgba_t,
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
            joint_names_simulation=list(D6_JOINT_NAMES),
            body_names_simulation=list(D6_BODY_NAMES),
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    behaviors = ()
    if attach_grasp:
        behaviors = (
            GraspPose.for_side_axis(
                axis=(1.0, 0.0, 0.0),
                approach_dirs=((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
                pos=(0.0, 0.0, 0.0),
            ),
        )
    return AssetSpec(config=cfg, behaviors=behaviors)


registry.register("asset", "dummy_d6", make_dummy_d6)

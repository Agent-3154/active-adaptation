from __future__ import annotations

from typing import Literal

from active_adaptation.assets.underwater.BlueROVHeavy import (
    ADDED_MASS,
    COBM,
    INIT_POS,
    NUM_ROTORS,
    ROTOR_FORCE_CONSTANTS,
    ROTOR_MAX_ROTATION_VEL_RAD_S,
    ROTOR_TIME_CONSTANTS,
)
from active_adaptation.envs.robots.underwater import HydrodynamicsCfg, UnderwaterRobot
from active_adaptation.registry import Registry
from active_adaptation import ROBOT_MODEL_DIR

registry = Registry.instance()

USD_PATH = ROBOT_MODEL_DIR / "underwater" / "BlueROVHeavyArm" / "model" / "model.usd"

# X5A-style arm (same joint naming as a2_manipulator).
_ARM_JOINTS = tuple(f"arm_joint{i}" for i in range(1, 9))
_ARM_BODIES = (
    "arm_base_link",
    "arm_link1",
    "arm_link2",
    "arm_link3",
    "arm_link4",
    "arm_link5",
    "gripper_base",
    "gripper_right",
    "gripper_left",
)

# Per-body displaced volume (m^3). Same base volume as BlueROVHeavy.
VOLUME = {
    "base_link": 0.0116499,
    "rotor_.*": 0.0,
    "arm_base_link": 0.00010617652,  # signed
    "arm_link1": 2.4842129e-05,  # signed
    "arm_link2": 0.00028625812,  # signed
    "arm_link3": 0.00019056577,  # watertight
    "arm_link4": 4.3607013e-05,  # signed
    "arm_link5": 0.00021319292,  # signed
    "gripper_base": 0.00014946794,  # signed
    "gripper_right": 3.1933421e-05,  # signed
    "gripper_left": 3.1933416e-05,  # watertight
}

# Per-body damping: base keeps MarineGym 6-DoF coeffs; arm floats are isotropic
# translational placeholders (~0.5*rho*Cd*A order for quadratic).
LINEAR_DAMPING = {
    "base_link": (4.03, 6.22, 5.18, 0.07, 0.07, 0.07),
    "rotor_.*": 0.0,
    "arm_base_link": 0.5,
    "arm_link1": 0.5,
    "arm_link2": 2.0,
    "arm_link3": 1.5,
    "arm_link4": 0.8,
    "arm_link5": 1.0,
    "gripper_base": 0.8,
    "gripper_right": 0.3,
    "gripper_left": 0.3,
}
QUADRATIC_DAMPING = {
    "base_link": (18.18, 21.66, 36.99, 1.55, 1.55, 1.55),
    "rotor_.*": 0.0,
    "arm_base_link": 2.0,
    "arm_link1": 2.0,
    "arm_link2": 10.0,
    "arm_link3": 8.0,
    "arm_link4": 4.0,
    "arm_link5": 5.0,
    "gripper_base": 4.0,
    "gripper_right": 1.5,
    "gripper_left": 1.5,
}

# Actuator gains: X5A URDF effort=100; stiffness/damping aligned with a2_manipulator.
ARM_EFFORT_LIMIT = 100.0
ARM_VELOCITY_LIMIT = 10.0
ARM_STIFFNESS = 40.0
ARM_DAMPING = 2.0
GRIPPER_EFFORT_LIMIT = 50.0
GRIPPER_VELOCITY_LIMIT = 0.1
GRIPPER_STIFFNESS = 80.0
GRIPPER_DAMPING = 2.0

INIT_JOINT_POS = {".*": 0.0}

JOINT_NAMES_SIMULATION = [
    *[f"rotor_{i}_joint" for i in range(NUM_ROTORS)],
    *_ARM_JOINTS,
]
BODY_NAMES_SIMULATION = [
    "base_link",
    *[f"rotor_{i}" for i in range(NUM_ROTORS)],
    *_ARM_BODIES,
]


def make_isaaclab_cfg(self_collisions: bool = False, fixed_base: bool = False):
    from active_adaptation.assets.asset_cfg import (
        AssetSpec,
        ArticulationCfg,
        ImplicitActuatorCfg,
        sim_utils,
    )

    asset_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collisions,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                fix_root_link=fixed_base,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=INIT_POS,
            joint_pos=INIT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Thruster dynamics are applied as body wrenches; joints are free-spinning placeholders.
            "rotors": ImplicitActuatorCfg(
                joint_names_expr=["rotor_.*_joint"],
                effort_limit_sim=0.0,
                velocity_limit_sim=max(ROTOR_MAX_ROTATION_VEL_RAD_S),
                stiffness=0.0,
                damping=0.0,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["arm_joint[1-6]"],
                effort_limit_sim=ARM_EFFORT_LIMIT,
                velocity_limit_sim=ARM_VELOCITY_LIMIT,
                stiffness=ARM_STIFFNESS,
                damping=ARM_DAMPING,
                friction=0.01,
                armature=0.01,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["arm_joint[7,8]"],
                effort_limit_sim=GRIPPER_EFFORT_LIMIT,
                velocity_limit_sim=GRIPPER_VELOCITY_LIMIT,
                stiffness=GRIPPER_STIFFNESS,
                damping=GRIPPER_DAMPING,
                friction=0.01,
                armature=0.01,
            ),
        },
        joint_names_simulation=JOINT_NAMES_SIMULATION,
        body_names_simulation=BODY_NAMES_SIMULATION,
    )
    return AssetSpec(
        config=asset_cfg,
        sensors={},
        wrapper=UnderwaterRobot(
            cfg=HydrodynamicsCfg(
                volume=VOLUME,
                coBM=COBM,
                added_mass=ADDED_MASS,
                linear_damping=LINEAR_DAMPING,
                quadratic_damping=QUADRATIC_DAMPING,
            ),
            rotor_time_constants=ROTOR_TIME_CONSTANTS,
            rotor_force_constants=ROTOR_FORCE_CONSTANTS,
        ),
    )


def make_mjlab_cfg(motrix: bool = False, fixed_base: bool = False):
    import mujoco
    from active_adaptation.assets.asset_cfg import AssetSpec, EntityCfg
    from mjlab.actuator import BuiltinPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.utils.spec_config import CollisionCfg

    mjcf_path = ROBOT_MODEL_DIR / "underwater" / "BlueROVHeavyArm" / "model.xml"

    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(mjcf_path))
        # Arm/gripper geoms are unnamed in the MJCF; CollisionCfg needs names.
        for body in spec.bodies:
            visual_i = 0
            collision_i = 0
            for geom in body.geoms:
                if geom.name:
                    continue
                is_visual = geom.contype == 0 and geom.conaffinity == 0
                if is_visual:
                    suffix = "_visual" if visual_i == 0 else f"_visual{visual_i}"
                    visual_i += 1
                else:
                    suffix = "_collision" if collision_i == 0 else f"_collision{collision_i}"
                    collision_i += 1
                geom.name = f"{body.name}{suffix}"
        # No freejoint ⇒ fixed base. mjlab auto-wraps a mocap root so per-env
        # placement via init_state / root pose still works.
        if fixed_base:
            for joint in list(spec.joints):
                if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                    spec.delete(joint)
        return spec

    cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=INIT_POS,
            joint_pos=INIT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        spec_fn=spec_fn,
        # Rotors stay unactuated (thrust is applied as body wrenches).
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPdActuatorCfg(
                    target_names_expr=("arm_joint[1-6]",),
                    effort_limit=ARM_EFFORT_LIMIT,
                    stiffness=ARM_STIFFNESS,
                    damping=ARM_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
                BuiltinPdActuatorCfg(
                    target_names_expr=("arm_joint[7-8]",),
                    effort_limit=GRIPPER_EFFORT_LIMIT,
                    stiffness=GRIPPER_STIFFNESS,
                    damping=GRIPPER_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
            ),
        ),
        collisions=(
            # Rotors are thrust dummies: omit them so disable_other_geoms turns
            # their colliders off. Gripper needs friction (condim=6); base/arm
            # are frictionless (condim=1).
            CollisionCfg(
                geom_names_expr=(
                    "base_link_collision.*",
                    "arm_.*_collision.*",
                    "gripper_.*_collision.*",
                ),
                contype=0,
                conaffinity=1,
                condim={
                    "gripper_.*_collision.*": 6,
                    "base_link_collision.*": 1,
                    "arm_.*_collision.*": 1,
                },
            ),
        ),
        joint_names_simulation=JOINT_NAMES_SIMULATION,
        body_names_simulation=BODY_NAMES_SIMULATION,
    )
    if motrix:
        from active_adaptation.envs.backends.motrix.mjcf import export_entity_mjcf

        cfg.motrix_mjcf_path_fn = lambda c: export_entity_mjcf(c, mjcf_path)

    return AssetSpec(
        config=cfg,
        sensors=(),
        wrapper=UnderwaterRobot(
            cfg=HydrodynamicsCfg(
                volume=VOLUME,
                coBM=COBM,
                added_mass=ADDED_MASS,
                linear_damping=LINEAR_DAMPING,
                quadratic_damping=QUADRATIC_DAMPING,
            ),
            rotor_time_constants=ROTOR_TIME_CONSTANTS,
            rotor_force_constants=ROTOR_FORCE_CONSTANTS,
        ),
    )


def make_cfg(
    backend: Literal["isaaclab", "mjlab", "motrix"],
    fixed_base: bool = False,
):
    if backend == "isaaclab":
        return make_isaaclab_cfg(fixed_base=fixed_base)
    elif backend == "mjlab":
        return make_mjlab_cfg(motrix=False, fixed_base=fixed_base)
    elif backend == "motrix":
        return make_mjlab_cfg(motrix=True, fixed_base=fixed_base)
    else:
        raise ValueError(f"Invalid backend: {backend}")


registry.register("asset", "bluerov_heavy_arm", make_cfg)

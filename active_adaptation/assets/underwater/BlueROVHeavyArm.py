from __future__ import annotations

from typing import Literal

from active_adaptation.assets.underwater.BlueROVHeavy import (
    ADDED_MASS,
    COBM,
    INIT_POS,
    LINEAR_DAMPING,
    NUM_ROTORS,
    QUADRATIC_DAMPING,
    ROTOR_FORCE_CONSTANTS,
    ROTOR_MAX_ROTATION_VEL_RAD_S,
    ROTOR_TIME_CONSTANTS,
)
from active_adaptation.envs.robots.underwater import HydrodynamicsCfg, UnderwaterRobot
from active_adaptation.registry import Registry
from active_adaptation import ROBOT_MODEL_DIR

registry = Registry.instance()

USD_PATH = ROBOT_MODEL_DIR / "underwater" / "BlueROVHeavyArm" / "model" / "model.usd"

# X5A-style arm (same joint naming as a2_manipulator). Arm / gripper volumes are
# zero placeholders until per-link displaced volumes are measured.
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


def make_isaaclab_cfg(self_collisions: bool = False):
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
                fix_root_link=False,
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


def make_mjlab_cfg(motrix: bool = False):
    raise NotImplementedError("MJLab backend is not supported for BlueROVHeavyArm")


def make_cfg(backend: Literal["isaaclab", "mjlab", "motrix"]):
    if backend == "isaaclab":
        return make_isaaclab_cfg()
    elif backend == "mjlab":
        return make_mjlab_cfg(motrix=False)
    elif backend == "motrix":
        return make_mjlab_cfg(motrix=True)
    else:
        raise ValueError(f"Invalid backend: {backend}")


registry.register("asset", "bluerov_heavy_arm", make_cfg)

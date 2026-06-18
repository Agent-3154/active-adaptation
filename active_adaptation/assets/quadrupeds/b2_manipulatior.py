"""Unitree B2 + Z1 arm — asset config.

MJCF/USD source: ``<repo>/.cache/aa-robot-models/b2z1/`` (``b2z1.xml``, ``b2z1_flattened.usd``).
"""

from __future__ import annotations

from typing import Literal

from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.registry import Registry
from active_adaptation.utils.symmetry import mirrored

registry = Registry.instance()

# note that we cannot import b2.py's parameters
# because the b2 descriptions are different

INIT_POS = (0.0, 0.0, 0.6)
INIT_JOINT_POS = {
    ".*R_hip_joint": -0.1,
    ".*L_hip_joint": 0.1,
    "F[L,R]_thigh_joint": 0.8,
    "R[L,R]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
    "z1_waist": 0.0,
    "z1_shoulder": 1.0,
    "z1_elbow": -1.8,
    "z1_wrist_angle": -0.1,
    "z1_forearm_roll": 0.0,
    "z1_wrist_rotate": 0.0,
    "z1_jointGripper": 0.0,
}

# ignore dummy bodies that doesn't move
# e.g., lidars and cameras
B2_JOINTS = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)

B2_BODIES = (
    "base_link",
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)

Z1_ARM_JOINTS = (
    "z1_waist",
    "z1_shoulder",
    "z1_elbow",
    "z1_wrist_angle",
    "z1_forearm_roll",
    "z1_wrist_rotate",
    "z1_jointGripper",
)

Z1_ARM_BODIES = (
    "arm_plat_link",
    "link00",
    "link01",
    "link02",
    "link03",
    "link04",
    "link05",
    "link06",
    "gripperStator",
    "gripperMover",
    "ee_gripper_link",
)

_B2_JOINT_SYMMETRY = mirrored(
    {
        "FL_hip_joint": (-1, "FR_hip_joint"),
        "RL_hip_joint": (-1, "RR_hip_joint"),
        "FL_thigh_joint": (1, "FR_thigh_joint"),
        "RL_thigh_joint": (1, "RR_thigh_joint"),
        "FL_calf_joint": (1, "FR_calf_joint"),
        "RL_calf_joint": (1, "RR_calf_joint"),
    }
)

_B2_SPATIAL_SYMMETRY = mirrored(
    {
        "FL_hip": "FR_hip",
        "RL_hip": "RR_hip",
        "FL_thigh": "FR_thigh",
        "RL_thigh": "RR_thigh",
        "FL_calf": "FR_calf",
        "RL_calf": "RR_calf",
        "FL_foot": "FR_foot",
        "RL_foot": "RR_foot",
        "base_link": "base_link",
    }
)

_Z1_JOINT_SYMMETRY = {
    "z1_waist": (-1, "z1_waist"),
    "z1_shoulder": (1, "z1_shoulder"),
    "z1_elbow": (1, "z1_elbow"),
    "z1_wrist_angle": (-1, "z1_wrist_angle"),
    "z1_forearm_roll": (-1, "z1_forearm_roll"),
    "z1_wrist_rotate": (-1, "z1_wrist_rotate"),
    "z1_jointGripper": (1, "z1_jointGripper"),
}

JOINT_SYMMETRY_MAPPING = {**_B2_JOINT_SYMMETRY, **_Z1_JOINT_SYMMETRY}

SPATIAL_SYMMETRY_MAPPING = {
    **_B2_SPATIAL_SYMMETRY,
    **{body: body for body in Z1_ARM_BODIES},
}

JOINT_NAMES_SIMULATION = [*B2_JOINTS, *Z1_ARM_JOINTS]
BODY_NAMES_SIMULATION = [*B2_BODIES, *Z1_ARM_BODIES]

LEG_HIP_THIGH_EFFORT = 200.0
LEG_CALF_EFFORT = 320.0
LEG_VELOCITY_LIMIT = 30.0
LEGS_STIFFNESS = 100.0
LEGS_DAMPING = 2.0

ARM_EFFORT_LIMIT = 30.0
ARM_SHOULDER_EFFORT = 60.0
ARM_VELOCITY_LIMIT = 3.1415
ARM_STIFFNESS = 40.0
ARM_DAMPING = 1.0
ARM_SHOULDER_DAMPING = 2.0
GRIPPER_STIFFNESS = 80.0


def make_isaaclab_cfg(self_collisions: bool = False):
    from isaaclab.sensors import ContactSensorCfg
    from active_adaptation.assets.asset_cfg import (
        ArticulationCfg,
        ImplicitActuatorCfg,
        sim_utils,
    )

    USD_PATH = ROBOT_MODEL_DIR / "b2z1" / "b2z1_flattened.usd"  # do not change

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
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=INIT_POS,
            joint_pos=INIT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "base_legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
                effort_limit_sim=LEG_HIP_THIGH_EFFORT,
                velocity_limit_sim=LEG_VELOCITY_LIMIT,
                stiffness=LEGS_STIFFNESS,
                damping=LEGS_DAMPING,
                friction=0.01,
                armature=0.01,
            ),
            "base_calf": ImplicitActuatorCfg(
                joint_names_expr=[".*_calf_joint"],
                effort_limit_sim=LEG_CALF_EFFORT,
                velocity_limit_sim=LEG_VELOCITY_LIMIT,
                stiffness=LEGS_STIFFNESS,
                damping=LEGS_DAMPING,
                friction=0.01,
                armature=0.01,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "z1_waist",
                    "z1_elbow",
                    "z1_wrist_angle",
                    "z1_forearm_roll",
                    "z1_wrist_rotate",
                ],
                effort_limit_sim=ARM_EFFORT_LIMIT,
                velocity_limit_sim=ARM_VELOCITY_LIMIT,
                stiffness=ARM_STIFFNESS,
                damping=ARM_DAMPING,
                friction=0.01,
                armature=0.01,
            ),
            "arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["z1_shoulder"],
                effort_limit_sim=ARM_SHOULDER_EFFORT,
                velocity_limit_sim=ARM_VELOCITY_LIMIT,
                stiffness=ARM_STIFFNESS,
                damping=ARM_SHOULDER_DAMPING,
                friction=0.01,
                armature=0.01,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["z1_jointGripper"],
                effort_limit_sim=ARM_EFFORT_LIMIT,
                velocity_limit_sim=10.0,
                stiffness=GRIPPER_STIFFNESS,
                damping=ARM_DAMPING,
                friction=0.01,
                armature=0.01,
            ),
        },
        joint_symmetry_mapping=JOINT_SYMMETRY_MAPPING,
        spatial_symmetry_mapping=SPATIAL_SYMMETRY_MAPPING,
        joint_names_simulation=JOINT_NAMES_SIMULATION,
        body_names_simulation=BODY_NAMES_SIMULATION,
    )
    sensors = {
        "contact_forces": ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            track_air_time=True,
            history_length=3,
        )
    }
    return asset_cfg, sensors


def make_mjlab_cfg(self_collisions: bool = False):
    import mujoco
    from active_adaptation.assets.asset_cfg import EntityCfg
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.utils.spec_config import CollisionCfg

    XML_PATH = ROBOT_MODEL_DIR / "b2z1" / "b2z1.xml"  # do not change

    def spec_fn():
        return mujoco.MjSpec.from_file(str(XML_PATH))

    cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=INIT_POS,
            joint_pos=INIT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        spec_fn=spec_fn,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
                    effort_limit=LEG_HIP_THIGH_EFFORT,
                    stiffness=LEGS_STIFFNESS,
                    damping=LEGS_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_calf_joint",),
                    effort_limit=LEG_CALF_EFFORT,
                    stiffness=LEGS_STIFFNESS,
                    damping=LEGS_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(
                        "z1_waist",
                        "z1_elbow",
                        "z1_wrist_angle",
                        "z1_forearm_roll",
                        "z1_wrist_rotate",
                    ),
                    effort_limit=ARM_EFFORT_LIMIT,
                    stiffness=ARM_STIFFNESS,
                    damping=ARM_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=("z1_shoulder",),
                    effort_limit=ARM_SHOULDER_EFFORT,
                    stiffness=ARM_STIFFNESS,
                    damping=ARM_SHOULDER_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=("z1_jointGripper",),
                    effort_limit=ARM_EFFORT_LIMIT,
                    stiffness=GRIPPER_STIFFNESS,
                    damping=ARM_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
            ),
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*_collision.*",),
                contype=0,
                conaffinity=1,
                # Harden all collision geoms.
                solref=(0.01, 1),
                # Configure feet colliders. Other colliders are frictionless (condim=1).
                condim={".*_foot_collision$": 6, ".*_collision.*": 1},
                priority={".*_foot_collision$": 1},
                friction={".*_foot_collision$": (1, 5e-3, 5e-4)}
            ),
        ),
        joint_symmetry_mapping=JOINT_SYMMETRY_MAPPING,
        spatial_symmetry_mapping=SPATIAL_SYMMETRY_MAPPING,
        joint_names_simulation=JOINT_NAMES_SIMULATION,
        body_names_simulation=BODY_NAMES_SIMULATION,
    )
    sensors = (
        ContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(
                mode="body",
                pattern=".*",
                entity="robot",
            ),
            secondary=ContactMatch(
                mode="body",
                pattern="terrain",
                entity=None,
            ),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            track_air_time=True,
            history_length=3,
        ),
    )
    return cfg, sensors


def make_cfg(backend: Literal["isaaclab", "mjlab"]):
    if backend == "isaaclab":
        return make_isaaclab_cfg()
    elif backend == "mjlab":
        return make_mjlab_cfg()
    else:
        raise ValueError(f"Invalid backend: {backend}")


registry.register("asset", "unitree_b2_z1", make_cfg)

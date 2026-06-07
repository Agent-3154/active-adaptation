"""Unitree A2 + AgileX Piper arm — asset config.

MJCF source: place ``aa-projects/assetx/artifacts/a2_piper/`` (``model.xml`` + ``meshes/``)
under ``<repo>/.cache/aa-robot-models/a2_piper/`` (same layout as the Hugging Face bundle).

USD for Isaac must be provided as ``a2_piper.usd`` in that folder (export separately if needed).
"""

from __future__ import annotations

from typing import Literal

from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.assets.quadrupeds.a2 import (
    BODY_NAMES_SIMULATION as A2_BODY_NAMES_SIMULATION,
    EFFORT_LIMIT,
    INIT_JOINT_POS as A2_INIT_JOINT_POS,
    INIT_POS,
    JOINT_NAMES_SIMULATION as A2_JOINT_NAMES_SIMULATION,
    JOINT_SYMMETRY_MAPPING as A2_JOINT_SYMMETRY_MAPPING,
    SPATIAL_SYMMETRY_MAPPING as A2_SPATIAL_SYMMETRY_MAPPING,
    VELOCITY_LIMIT,
    LEGS_STIFFNESS,
    LEGS_DAMPING,
)
from active_adaptation.registry import Registry

registry = Registry.instance()

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
    "grasp_point",
)

_ARM_JOINT_SYMMETRY = {
    "arm_joint1": (-1, "arm_joint1"),
    "arm_joint2": (1, "arm_joint2"),
    "arm_joint3": (1, "arm_joint3"),
    "arm_joint4": (-1, "arm_joint4"),
    "arm_joint5": (1, "arm_joint5"),
    "arm_joint6": (-1, "arm_joint6"),
    "arm_joint7": (-1, "arm_joint8"),
    "arm_joint8": (-1, "arm_joint7"),
}

INIT_JOINT_POS = {**A2_INIT_JOINT_POS, "arm_joint[1-8]": 0.0}

JOINT_SYMMETRY_MAPPING = {**A2_JOINT_SYMMETRY_MAPPING, **_ARM_JOINT_SYMMETRY}

SPATIAL_SYMMETRY_MAPPING = {
    **A2_SPATIAL_SYMMETRY_MAPPING,
    **{body: body for body in _ARM_BODIES},
}

JOINT_NAMES_SIMULATION = [*A2_JOINT_NAMES_SIMULATION, *_ARM_JOINTS]
BODY_NAMES_SIMULATION = [*A2_BODY_NAMES_SIMULATION, *_ARM_BODIES]

ARM_EFFORT_LIMIT = 80.0
ARM_VELOCITY_LIMIT = 20.0
ARM_STIFFNESS = 40.0
ARM_DAMPING = 1.0


def make_isaaclab_cfg(self_collisions: bool = False):
    from isaaclab.sensors import ContactSensorCfg
    from active_adaptation.assets.asset_cfg import (
        ArticulationCfg,
        ImplicitActuatorCfg,
        sim_utils,
    )

    asset_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_MODEL_DIR / "a2_piper" / "a2_piper.usd"),
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
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit_sim=EFFORT_LIMIT,
                velocity_limit_sim=VELOCITY_LIMIT,
                stiffness=LEGS_STIFFNESS,
                damping=LEGS_DAMPING,
                friction=0.01,
                armature=0.01,
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
                stiffness=80.0,
                damping=1.0,
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


def make_mjlab_cfg():
    import mujoco
    from active_adaptation.assets.asset_cfg import EntityCfg
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.utils.spec_config import CollisionCfg

    def spec_fn():
        mjcf_path = str(ROBOT_MODEL_DIR / "a2_piper" / "model.xml")
        return mujoco.MjSpec.from_file(mjcf_path)

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
                    target_names_expr=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
                    effort_limit=EFFORT_LIMIT,
                    stiffness=LEGS_STIFFNESS,
                    damping=LEGS_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=("arm_joint[1-8]",),
                    effort_limit=ARM_EFFORT_LIMIT,
                    stiffness=ARM_STIFFNESS,
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
                condim=3,
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
    if backend == "mjlab":
        return make_mjlab_cfg()
    raise ValueError(f"Invalid backend: {backend}")


registry.register("asset", "unitree_a2_piper", make_cfg)

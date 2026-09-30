"""Boston Dynamics Spot with the arm and gripper.

MJCF/USD: ``<ROBOT_MODEL_DIR>/spot_arm/``, cooked from the assetx recipe
``spot_arm`` (``aa-cook-assets spot_arm``).
The gripper is one revolute jaw (``arm_f1x``). Both finger meshes live on
``arm_link_fngr``, which is the contact body. The jaw opens along EEF ``+Z``.
``arm_f1x`` spawns open (near the negative soft limit). Closed rest is ``q ≈ 0``.
"""

from __future__ import annotations

from typing import Literal

from active_adaptation.assets.cooked import cooked_model_dir
from active_adaptation.assets.quadrupeds.spot import (
    BODY_NAMES_SIMULATION as SPOT_BODY_NAMES_SIMULATION,
    HIP_EFFORT_LIMIT,
    INIT_JOINT_POS as SPOT_INIT_JOINT_POS,
    INIT_POS,
    JOINT_NAMES_SIMULATION as SPOT_JOINT_NAMES_SIMULATION,
    JOINT_SYMMETRY_MAPPING as SPOT_JOINT_SYMMETRY_MAPPING,
    KNEE_EFFORT_LIMIT,
    LEGS_DAMPING,
    LEGS_STIFFNESS,
    LEGS_VELOCITY_LIMIT,
    SPATIAL_SYMMETRY_MAPPING as SPOT_SPATIAL_SYMMETRY_MAPPING,
    _mjlab_collisions,
    load_mjcf,
)
from active_adaptation.registry import Registry

registry = Registry.instance()

_ARM_JOINTS = (
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
)
_ARM_BODIES = (
    "arm_link_sh0",
    "arm_link_sh1",
    "arm_link_hr0",
    "arm_link_el0",
    "arm_link_el1",
    "arm_link_wr0",
    "arm_link_wr1",
    "arm_link_fngr",
    "grasp_point",
)

# Arm is on the sagittal plane: yaw/roll flip under left-right mirror, pitch does not.
_ARM_JOINT_SYMMETRY = {
    "arm_sh0": (-1, "arm_sh0"),
    "arm_sh1": (1, "arm_sh1"),
    "arm_el0": (1, "arm_el0"),
    "arm_el1": (-1, "arm_el1"),
    "arm_wr0": (1, "arm_wr0"),
    "arm_wr1": (-1, "arm_wr1"),
    "arm_f1x": (1, "arm_f1x"),
}

# Menagerie stow, with the jaw open instead of the closed keyframe (q=0).
INIT_JOINT_POS = {
    **SPOT_INIT_JOINT_POS,
    "arm_sh0": 0.0,
    "arm_sh1": -3.14,
    "arm_el0": 3.06,
    "arm_el1": 0.0,
    "arm_wr0": 0.0,
    "arm_wr1": 0.0,
    "arm_f1x": -1.57,
}

JOINT_SYMMETRY_MAPPING = {**SPOT_JOINT_SYMMETRY_MAPPING, **_ARM_JOINT_SYMMETRY}
SPATIAL_SYMMETRY_MAPPING = {
    **SPOT_SPATIAL_SYMMETRY_MAPPING,
    **{body: body for body in _ARM_BODIES},
}
JOINT_NAMES_SIMULATION = [*SPOT_JOINT_NAMES_SIMULATION, *_ARM_JOINTS]
BODY_NAMES_SIMULATION = [*SPOT_BODY_NAMES_SIMULATION, *_ARM_BODIES]

ARM_EFFORT_LIMIT = 60.0
ARM_VELOCITY_LIMIT = 10.0
ARM_STIFFNESS = 60.0
ARM_DAMPING = 2.0
GRIPPER_EFFORT_LIMIT = 15.0
GRIPPER_STIFFNESS = 20.0
GRIPPER_DAMPING = 0.5


def make_isaaclab_cfg(self_collisions: bool = False):
    from isaaclab.sensors import ContactSensorCfg
    from active_adaptation.assets.asset_cfg import (
        AssetSpec,
        ArticulationCfg,
        ImplicitActuatorCfg,
        sim_utils,
    )
    from active_adaptation.envs.behaviors.gripper import GripperBehavior

    model_dir = cooked_model_dir("spot_arm", usd=True)
    asset_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(model_dir / "usd" / "spot.usdc"),
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
            # The assetx USD has no DriveAPI (actuators are stripped); without
            # one PhysX ignores the implicit actuator gains.
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=INIT_POS,
            joint_pos=INIT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hip": ImplicitActuatorCfg(
                joint_names_expr=[".*_hx", ".*_hy"],
                effort_limit_sim=HIP_EFFORT_LIMIT,
                velocity_limit_sim=LEGS_VELOCITY_LIMIT,
                stiffness=LEGS_STIFFNESS,
                damping=LEGS_DAMPING,
                friction=0.01,
                armature=0.01,
            ),
            "knee": ImplicitActuatorCfg(
                joint_names_expr=[".*_kn"],
                effort_limit_sim=KNEE_EFFORT_LIMIT,
                velocity_limit_sim=LEGS_VELOCITY_LIMIT,
                stiffness=LEGS_STIFFNESS,
                damping=LEGS_DAMPING,
                friction=0.01,
                armature=0.01,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"],
                effort_limit_sim=ARM_EFFORT_LIMIT,
                velocity_limit_sim=ARM_VELOCITY_LIMIT,
                stiffness=ARM_STIFFNESS,
                damping=ARM_DAMPING,
                friction=0.01,
                armature=0.01,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["arm_f1x"],
                effort_limit_sim=GRIPPER_EFFORT_LIMIT,
                velocity_limit_sim=ARM_VELOCITY_LIMIT,
                stiffness=GRIPPER_STIFFNESS,
                damping=GRIPPER_DAMPING,
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
    return AssetSpec(
        config=asset_cfg,
        sensors=sensors,
        behaviors=(
            GripperBehavior(
                eef_body_name="grasp_point",
                joint_names="arm_f1x",
                body_names="arm_link_fngr",
                open_direction=(0.0, 0.0, 1.0),
            ),
        ),
    )


def make_mjlab_cfg():
    from active_adaptation.assets.asset_cfg import AssetSpec, EntityCfg
    from active_adaptation.envs.behaviors.gripper import GripperBehavior
    from mjlab.actuator import BuiltinPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    model_dir = cooked_model_dir("spot_arm", usd=False)

    def spec_fn():
        return load_mjcf(model_dir / "model.xml")

    cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=INIT_POS,
            joint_pos=INIT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        spec_fn=spec_fn,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPdActuatorCfg(
                    target_names_expr=(".*_hx", ".*_hy"),
                    effort_limit=HIP_EFFORT_LIMIT,
                    stiffness=LEGS_STIFFNESS,
                    damping=LEGS_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
                BuiltinPdActuatorCfg(
                    target_names_expr=(".*_kn",),
                    effort_limit=KNEE_EFFORT_LIMIT,
                    stiffness=LEGS_STIFFNESS,
                    damping=LEGS_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
                BuiltinPdActuatorCfg(
                    target_names_expr=("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"),
                    effort_limit=ARM_EFFORT_LIMIT,
                    stiffness=ARM_STIFFNESS,
                    damping=ARM_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
                BuiltinPdActuatorCfg(
                    target_names_expr=("arm_f1x",),
                    effort_limit=GRIPPER_EFFORT_LIMIT,
                    stiffness=GRIPPER_STIFFNESS,
                    damping=GRIPPER_DAMPING,
                    armature=0.01,
                    frictionloss=0.01,
                ),
            ),
        ),
        collisions=_mjlab_collisions(),
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
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
            history_length=3,
        ),
    )
    return AssetSpec(
        config=cfg,
        sensors=sensors,
        behaviors=(
            GripperBehavior(
                eef_body_name="grasp_point",
                joint_names="arm_f1x",
                body_names="arm_link_fngr",
                open_direction=(0.0, 0.0, 1.0),
            ),
        ),
    )


def make_cfg(backend: Literal["isaaclab", "mjlab"]):
    if backend == "isaaclab":
        return make_isaaclab_cfg()
    if backend == "mjlab":
        return make_mjlab_cfg()
    raise ValueError(f"Invalid backend: {backend}")


registry.register("asset", "spot_arm", make_cfg)

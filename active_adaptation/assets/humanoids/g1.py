from __future__ import annotations

from pathlib import Path
from typing import Literal

from active_adaptation.registry import Registry
from active_adaptation.utils.symmetry import mirrored

registry = Registry.instance()

ASSETS_DIR = Path(__file__).resolve().parents[1] / "G1"
MJCF_PATH = ASSETS_DIR / "mjcf" / "g1.xml"
USD_PATH = ASSETS_DIR / "waist_unlocked.usd"

INIT_POS = (0.0, 0.0, 0.85)
INIT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
}

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

JOINT_SYMMETRY_MAPPING = mirrored(
    {
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
        "left_knee_joint": (1, "right_knee_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "waist_roll_joint": (-1, "waist_roll_joint"),
        "waist_pitch_joint": (1, "waist_pitch_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
        "left_elbow_joint": (1, "right_elbow_joint"),
        "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
        "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
        "left_wrist_yaw_joint": (-1, "right_wrist_yaw_joint"),
    }
)

SPATIAL_SYMMETRY_MAPPING = mirrored(
    {
        "left_hip_pitch_link": "right_hip_pitch_link",
        "left_hip_roll_link": "right_hip_roll_link",
        "left_hip_yaw_link": "right_hip_yaw_link",
        "left_knee_link": "right_knee_link",
        "left_ankle_pitch_link": "right_ankle_pitch_link",
        "left_ankle_roll_link": "right_ankle_roll_link",
        "pelvis": "pelvis",
        "torso_link": "torso_link",
        "waist_yaw_link": "waist_yaw_link",
        "waist_roll_link": "waist_roll_link",
        "left_shoulder_pitch_link": "right_shoulder_pitch_link",
        "left_shoulder_roll_link": "right_shoulder_roll_link",
        "left_shoulder_yaw_link": "right_shoulder_yaw_link",
        "left_elbow_link": "right_elbow_link",
        "left_wrist_roll_link": "right_wrist_roll_link",
        "left_wrist_yaw_link": "right_wrist_yaw_link",
        "left_wrist_pitch_link": "right_wrist_pitch_link",
    }
)

JOINT_NAMES_SIMULATION = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

BODY_NAMES_SIMULATION = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


def make_isaaclab_cfg(self_collisions: bool = False):
    from isaaclab.sensors import ContactSensorCfg
    from active_adaptation.assets.asset_cfg import ArticulationCfg, ImplicitActuatorCfg, sim_utils

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
            "hip_yaw": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_yaw_joint"],
                effort_limit_sim=88.0,
                velocity_limit_sim=32.0,
                stiffness=STIFFNESS_7520_14 * 2.0,
                damping=DAMPING_7520_14,
                friction=0.01,
                armature=ARMATURE_7520_14,
            ),
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_pitch_joint"],
                effort_limit_sim=88.0,
                velocity_limit_sim=32.0,
                stiffness=STIFFNESS_7520_14 * 4.0,
                damping=DAMPING_7520_14,
                friction=0.01,
                armature=ARMATURE_7520_14,
            ),
            "hip_roll_knee": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"],
                effort_limit_sim=139.0,
                velocity_limit_sim=20.0,
                stiffness=STIFFNESS_7520_22,
                damping=DAMPING_7520_22,
                friction=0.01,
                armature=ARMATURE_7520_22,
            ),
            "ankle_waist": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle.*", "waist.*"],
                effort_limit_sim=50.0,
                velocity_limit_sim=37.0,
                stiffness=2.0 * STIFFNESS_5020,
                damping=2.0 * DAMPING_5020,
                friction=0.01,
                armature=2.0 * ARMATURE_5020,
            ),
            "upper_arm": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder.*", ".*_elbow.*", ".*_wrist_roll_joint"],
                effort_limit_sim=25.0,
                velocity_limit_sim=37.0,
                stiffness=STIFFNESS_5020,
                damping=DAMPING_5020,
                friction=0.01,
                armature=ARMATURE_5020,
            ),
            "wrist_pitch_yaw": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
                effort_limit_sim=5.0,
                velocity_limit_sim=22.0,
                stiffness=STIFFNESS_4010,
                damping=DAMPING_4010,
                friction=0.01,
                armature=ARMATURE_4010,
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
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.utils.spec_config import CollisionCfg
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.sensor import ContactSensorCfg, ContactMatch

    def spec_fn():
        return mujoco.MjSpec.from_file(str(MJCF_PATH))

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
                    target_names_expr=(".*_hip_yaw_joint",),
                    effort_limit=88.0,
                    stiffness=STIFFNESS_7520_14 * 2.0,
                    damping=DAMPING_7520_14,
                    armature=ARMATURE_7520_14,
                    frictionloss=0.01,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_hip_pitch_joint",),
                    effort_limit=88.0,
                    stiffness=STIFFNESS_7520_14 * 4.0,
                    damping=DAMPING_7520_14,
                    armature=ARMATURE_7520_14,
                    frictionloss=0.01,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
                    effort_limit=139.0,
                    stiffness=STIFFNESS_7520_22,
                    damping=DAMPING_7520_22,
                    armature=ARMATURE_7520_22,
                    frictionloss=0.01,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_ankle.*", "waist.*"),
                    effort_limit=50.0,
                    stiffness=2.0 * STIFFNESS_5020,
                    damping=2.0 * DAMPING_5020,
                    armature=2.0 * ARMATURE_5020,
                    frictionloss=0.01,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_shoulder.*", ".*_elbow.*", ".*_wrist_roll_joint"),
                    effort_limit=25.0,
                    stiffness=STIFFNESS_5020,
                    damping=DAMPING_5020,
                    armature=ARMATURE_5020,
                    frictionloss=0.01,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
                    effort_limit=5.0,
                    stiffness=STIFFNESS_4010,
                    damping=DAMPING_4010,
                    armature=ARMATURE_4010,
                    frictionloss=0.01,
                ),
            ),
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*",),
                disable_other_geoms=False,
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
            primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain", entity=None),
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


registry.register("asset", "g1_waist_unlocked", make_cfg)

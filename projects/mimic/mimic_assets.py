import mimickit
from pathlib import Path

from active_adaptation.registry import Registry
from active_adaptation.assets.asset_cfg import (
    AssetCfg,
    InitialStateCfg,
    ActuatorCfg,
    ContactSensorCfg
)

registry = Registry.instance()

ASSETS_ROOT = Path(mimickit.__path__[0]).parent / "data" / "assets"

G1_CFG = AssetCfg(
    usd_path=ASSETS_ROOT / "g1" / "g1.usd",
    mjcf_path=ASSETS_ROOT / "g1" / "g1.xml",
    self_collisions=False,
    init_state=InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        rot=(1.0, 0.0, 0.0, 0.0),
        # joint_pos={".*": 0.0},
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            # ".*_hip_pitch_joint": -0.28,
            # ".*_knee_joint": 0.669,
            # ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0}
    ),
    actuators={
        "all": ActuatorCfg(
            joint_names_expr=".*",
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            friction=None,
            armature=None
        )
    },
    sensors_isaaclab=[
        ContactSensorCfg(
            name="contact_forces",
            primary="robot/.*",
            secondary=[],
            track_air_time=True,
            history_length=3
        ),
    ],
)
registry.register("asset", "mimickit:g1", G1_CFG)
import os
import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from active_adaptation.assets.base import ArticulationCfg
from active_adaptation.utils.symmetry import mirrored

ASSET_PATH = os.path.dirname(__file__)

SIRIUS_WHEEL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/sirius_wheel/sirius_wheel_sphere.usd",
        # usd_path=f"{ASSET_PATH}/sirius_wheel/sirius_wheel_mesh.usd",
        activate_contact_sensors=True,
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
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*_HAA": 0.,
            "[L,R]F_HFE":  0.4,
            "[L,R]H_HFE": -0.4,
            "[L,R]F_KFE": -1.2,
            "[L,R]H_KFE":  1.2,
            ".*_WHEEL": 0.
        },
        joint_vel={".*": 0.}
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={
                ".*_HAA": 50.,
                ".*_HFE": 50.,
                ".*_KFE": 80.,
                ".*_WHEEL": 80.
            },
            velocity_limit_sim=80.,
            # saturation_effort=100.0,
            stiffness={
                ".*_HAA": 30.,
                ".*_HFE": 30.,
                ".*_KFE": 30.,
                ".*_WHEEL": 0.
            },
            damping={
                ".*_HAA": 1.,
                ".*_HFE": 1.,
                ".*_HFE": 1.,
                ".*_WHEEL": 10.
            },
            armature=0.01,
            friction=0.01
        )
    },
    joint_symmetry_mapping=mirrored({
        "LF_HAA": (-1, "RF_HAA"),
        "LH_HAA": (-1, "RH_HAA"),
        "LF_HFE": (1, "RF_HFE"),
        "LH_HFE": (1, "RH_HFE"),
        "LF_KFE": (1, "RF_KFE"),
        "LH_KFE": (1, "RH_KFE"),
        "LF_WHEEL": (1, "RF_WHEEL"),
        "LH_WHEEL": (1, "RH_WHEEL"),
    }),
    spatial_symmetry_mapping=mirrored({
        "trunk": "trunk",
        "front": "front",
        "back": "back",
        "LF_hip": "RF_hip",
        "LH_hip": "RH_hip",
        "LF_calf": "RF_calf",
        "LH_calf": "RH_calf",
        "LF_thigh": "RF_thigh",
        "LH_thigh": "RH_thigh",
        "LF_FOOT": "RF_FOOT",
        "LH_FOOT": "RH_FOOT",
    }),
)

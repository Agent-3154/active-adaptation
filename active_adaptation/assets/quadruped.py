import os
from pathlib import Path
from active_adaptation.assets.asset_cfg import AssetCfg, InitialStateCfg, ActuatorCfg, ContactSensorCfg
from active_adaptation.registry import Registry

registry = Registry.instance()

FILE_DIR = Path(__file__).parent

UNITREE_GO2_CFG = AssetCfg(
    mjcf_path=FILE_DIR / "Go2" / "mjcf" / "go2.xml",
    usd_path=FILE_DIR / "Go2" / "go2.usd",
    init_state=InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.7,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    self_collisions=False,
    actuators={
        "base_legs": ActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            # effort_limit={
            #     ".*_hip_joint": 23.5,
            #     ".*_thigh_joint": 23.5,
            #     ".*_calf_joint": 35.5,
            # },
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
            armature=0.01,
        ),
    },
    joint_symmetry_mapping = { 
        "FL_hip_joint": (-1, "FR_hip_joint"),
        "FR_hip_joint": (-1, "FL_hip_joint"),
        "RL_hip_joint": (-1, "RR_hip_joint"),
        "RR_hip_joint": (-1, "RL_hip_joint"),
        "FL_thigh_joint": (1, "FR_thigh_joint"),
        "FR_thigh_joint": (1, "FL_thigh_joint"),
        "RL_thigh_joint": (1, "RR_thigh_joint"),
        "RR_thigh_joint": (1, "RL_thigh_joint"),
        "FL_calf_joint": (1, "FR_calf_joint"),
        "FR_calf_joint": (1, "FL_calf_joint"),
        "RL_calf_joint": (1, "RR_calf_joint"),
        "RR_calf_joint": (1, "RL_calf_joint")
    },
    spatial_symmetry_mapping = {
        "FL_hip": "FR_hip",
        "FR_hip": "FL_hip",
        "RL_hip": "RR_hip",
        "RR_hip": "RL_hip",
        "FL_thigh": "FR_thigh",
        "FR_thigh": "FL_thigh",
        "RL_thigh": "RR_thigh",
        "RR_thigh": "RL_thigh",
        "FL_calf": "FR_calf",
        "FR_calf": "FL_calf",
        "RL_calf": "RR_calf",
        "RR_calf": "RL_calf",
        "FL_foot": "FR_foot",
        "FR_foot": "FL_foot",
        "RL_foot": "RR_foot",
        "RR_foot": "RL_foot",
        "base": "base",
        "Head_upper": "Head_upper",
        "Head_lower": "Head_lower",
    },
    sensors_isaaclab=[
        ContactSensorCfg(
            name="contact_forces",
            primary=".*",
            secondary=[],
            track_air_time=True,
            history_length=3
        ),
    ],
    body_names_isaac=[
        "base",
        "FL_hip",
        "FR_hip",
        "Head_upper",
        "RL_hip",
        "RR_hip",
        "FL_thigh",
        "FR_thigh",
        "Head_lower",
        "RL_thigh",
        "RR_thigh",
        "FL_calf",
        "FR_calf",
        "RL_calf",
        "RR_calf",
        "FL_foot",
        "FR_foot",
        "RL_foot",
        "RR_foot"
    ],
    joint_names_isaac=[
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
        "RR_calf_joint"
    ],
)
registry.register("asset", "go2", UNITREE_GO2_CFG)

# UNITREE_B1Z1_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ASSET_PATH}/b1/b1_plus_z1.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=1,
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.6),
#         joint_pos={
#             ".*L_hip_joint": 0.2,
#             ".*R_hip_joint": -0.2,
#             "F[L,R]_thigh_joint": 0.6,
#             "R[L,R]_thigh_joint": 1.0,
#             ".*_calf_joint": -1.3,
#             'arm_joint1': 0.0,
#             'arm_joint2': 1.0, # 1.5
#             'arm_joint3': -1.8, # -1.5
#             'arm_joint4': -0.1, # -0.54
#             'arm_joint5': 0.0,
#             'arm_joint6': 0.0,
#             'jointGripper': 0.0,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "base_legs": ImplicitActuatorCfg(
#             joint_names_expr=".*",
#             effort_limit_sim=200.0,
#             # saturation_effort=35.5,
#             velocity_limit_sim=40.0,
#             stiffness={
#                 ".*hip_joint": 100.0,
#                 ".*thigh_joint": 100.0,
#                 ".*calf_joint": 100.0,
#                 "arm_joint.*": 40.0,
#             },
#             damping={
#                 ".*hip_joint": 2.0,
#                 ".*thigh_joint": 2.0,
#                 ".*calf_joint": 2.0,
#                 "arm_joint.*": 1.0,
#             },
#             friction=0.01,
#             armature=0.01,
#         ),
#     },
# )
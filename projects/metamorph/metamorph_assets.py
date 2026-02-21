import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, ContactSensor
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from metamorphosis.asset_cfg import ProceduralQuadrupedCfg, QuadrupedBuilder
from active_adaptation.registry import Registry

registry = Registry.instance()

QUADRUPED_CONFIG = ArticulationCfg(
    spawn=ProceduralQuadrupedCfg(
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        base_length_range=(0.5, 1.0),
        base_width_range=(0.3, 0.4),
        base_height_range=(0.15, 0.25),
        leg_length_range=(0.4, 0.8),
        calf_length_ratio=(0.9, 1.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            ".*_hip_joint": 0.0,
            "F[L,R]_thigh_joint": torch.pi / 4,
            "R[L,R]_thigh_joint": torch.pi / 4,
            ".*_calf_joint": -torch.pi / 2,
        },
        pos=(0.0, 0.0, 1.0),
    ),
    actuators={
        ".*": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=1000.0,
            stiffness=80.0,
            damping=2.0,
            armature=0.01,
            friction=0.01,
        ),
    },
)

registry.register("asset", "quadruped", QUADRUPED_CONFIG)

from active_adaptation.registry import Registry
from active_adaptation.utils.symmetry import mirrored
from active_adaptation import ROBOT_MODEL_DIR
from typing import Literal


registry = Registry.instance()

INIT_POS = (0.0, 0.0, 0.6)
INIT_JOINT_POS = {
    ".*_hip_joint": 0.0,
    "F[L,R]_thigh_joint": 0.6,
    "R[L,R]_thigh_joint": 1.0,
    ".*_calf_joint": -1.3,
}

JOINT_SYMMETRY_MAPPING = mirrored({
    "FL_hip_joint": (-1, "FR_hip_joint"),
    "RL_hip_joint": (-1, "RR_hip_joint"),
    "FL_thigh_joint": (1, "FR_thigh_joint"),
    "RL_thigh_joint": (1, "RR_thigh_joint"),
    "FL_calf_joint": (1, "FR_calf_joint"),
    "RL_calf_joint": (1, "RR_calf_joint"),
})

SPATIAL_SYMMETRY_MAPPING = mirrored({
    "FL_hip": "FR_hip",
    "RL_hip": "RR_hip",
    "FL_thigh": "FR_thigh",
    "RL_thigh": "RR_thigh",
    "FL_calf": "FR_calf",
    "RL_calf": "RR_calf",
    "FL_foot": "FR_foot",
    "RL_foot": "RR_foot",
    "base_link": "base_link",
})

JOINT_NAMES_SIMULATION = [
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
]

BODY_NAMES_SIMULATION = [
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
]

EFFORT_LIMIT = 120.0
VELOCITY_LIMIT = 30.0
LEGS_STIFFNESS = 50.0
LEGS_DAMPING = 1.0

def make_isaaclab_cfg(self_collisions: bool = False):
    from isaaclab.sensors import ContactSensorCfg
    from active_adaptation.assets.asset_cfg import (
        ArticulationCfg,
        ImplicitActuatorCfg,
        sim_utils
    )
    asset_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_MODEL_DIR / "a2" / "a2.usd"), 
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
                effort_limit_sim=120.0,
                velocity_limit_sim=30.0,
                stiffness=LEGS_STIFFNESS,
                damping=LEGS_DAMPING,
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


def make_mjlab_cfg(motrix: bool = False):
    import mujoco
    from active_adaptation.assets.asset_cfg import EntityCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.utils.spec_config import CollisionCfg
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.sensor import ContactSensorCfg, ContactMatch

    mjcf_path = ROBOT_MODEL_DIR / "a2" / "a2.xml"

    def spec_fn():
        return mujoco.MjSpec.from_file(str(mjcf_path))

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
                    stiffness=50.0,
                    damping=2.0,
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
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
            history_length=3,
        ),
    )
    if motrix:
        from active_adaptation.envs.backends.motrix.mjcf import export_entity_mjcf

        cfg.motrix_mjcf_path_fn = lambda c: export_entity_mjcf(c, mjcf_path)
    return cfg, sensors


def make_cfg(backend: Literal["isaaclab", "mjlab", "motrix"]):
    if backend == "isaaclab":
        return make_isaaclab_cfg()
    elif backend == "mjlab":
        return make_mjlab_cfg(motrix=False)
    elif backend == "motrix":
        return make_mjlab_cfg(motrix=True)
    else:
        raise ValueError(f"Invalid backend: {backend}")

registry.register("asset", "unitree_a2", make_cfg)

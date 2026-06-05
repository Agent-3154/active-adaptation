from active_adaptation.registry import Registry
from active_adaptation.utils.symmetry import mirrored
from active_adaptation import ROBOT_MODEL_DIR
from typing import Literal


registry = Registry.instance()

INIT_POS = (0.0, 0.0, 0.4)
INIT_JOINT_POS = {
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "F[L,R]_thigh_joint": 0.7,
    "R[L,R]_thigh_joint": 0.8,
    ".*_calf_joint": -1.5,
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
    "base": "base",
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
    "base",
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

def make_mjlab_cfg(motrix: bool = False):
    import mujoco
    from active_adaptation.assets.asset_cfg import EntityCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.utils.spec_config import CollisionCfg
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.sensor import ContactSensorCfg, ContactMatch

    mjcf_path = ROBOT_MODEL_DIR / "go2_unilab" / "go2.xml"

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
                    effort_limit=23.5,
                    stiffness=25.0,
                    damping=0.5,
                    # armature=0.01,
                    # frictionloss=0.01,
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
    sensors=(
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


def make_cfg(backend: Literal["mjlab", "motrix"]):
    if backend == "mjlab":
        return make_mjlab_cfg(motrix=False)
    elif backend == "motrix":
        return make_mjlab_cfg(motrix=True)
    else:
        raise ValueError(f"Invalid backend: {backend}")

registry.register("asset", "unitree_go2", make_cfg)

"""Unitree AS2 quadruped.

MJCF/USD: ``<ROBOT_MODEL_DIR>/as2/`` (``aa-cook-assets as2``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from active_adaptation.registry import Registry
from active_adaptation.utils.symmetry import mirrored

registry = Registry.instance()

# Pinned vendor for ``aa-cook-assets as2`` (unitree_ros as2_description).
UNITREE_AS2_VENDOR = (
    "https://github.com/unitreerobotics/unitree_ros/tree/"
    "38a9b00ecbff0b70e85a7627eb9c65db9a2341f7/robots/as2_description"
)
AS2_MJCF = "as2.xml"

_COOK_FOOT_GEOMS = (
    ("FL", "FL_calf_collision3", "FL_calf_visual1"),
    ("FR", "FR_calf_collision3", "FR_calf_visual1"),
    ("RL", "RL_calf_collision3", "RL_calf_visual1"),
    ("RR", "RR_calf_collision3", "RR_calf_visual1"),
)


def build_as2_mjcf(vendor_dir: Path):
    """Compose vendor MJCF into the cooked ``ROBOT_MODEL_DIR/as2`` bundle."""
    from assetx.core.asset import MujocoAsset
    from assetx.core.transforms import (
        Compose,
        GeomsToBody,
        NormalizeGeomNames,
        NormalizeJointNames,
        RemoveActuators,
        RemoveSceneSettings,
        RemoveSensors,
    )

    return Compose([
        NormalizeGeomNames(),
        NormalizeJointNames(),
        RemoveSensors(names=[".*pos", ".*torque", "imu.*", ".*vel"]),
        RemoveActuators(names=[".*"]),
        RemoveSceneSettings(),
        *(
            GeomsToBody([collision, visual], f"{prefix}_foot", mass=0.05)
            for prefix, collision, visual in _COOK_FOOT_GEOMS
        ),
    ]).transform(MujocoAsset.from_file(vendor_dir / AS2_MJCF))

# Vendor model spawns the base at z=0.5 with the feet near contact.
INIT_POS = (0.0, 0.0, 0.5)
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

HIP_EFFORT_LIMIT = 60.0
THIGH_EFFORT_LIMIT = 60.0
CALF_EFFORT_LIMIT = 90.0
LEGS_VELOCITY_LIMIT = 20.0
LEGS_STIFFNESS = 60.0
LEGS_DAMPING = 1.5

_FOOT_GEOM = ".*_foot_collision.*"


def _mjlab_collisions():
    from mjlab.utils.spec_config import CollisionCfg

    return (
        CollisionCfg(
            geom_names_expr=(".*_collision.*",),
            contype=0,
            conaffinity=1,
            solref=(0.004, 1),
            condim={_FOOT_GEOM: 6, ".*": 3},
            priority={_FOOT_GEOM: 1, ".*": 0},
            friction={_FOOT_GEOM: (0.8, 0.02, 0.01)},
            solimp={_FOOT_GEOM: (0.9, 0.95, 0.003)},
            margin={_FOOT_GEOM: 0.005},
        ),
    )


def make_isaaclab_cfg(self_collisions: bool = False):
    from isaaclab.sensors import ContactSensorCfg
    from active_adaptation.assets.asset_cfg import (
        AssetSpec,
        ArticulationCfg,
        ImplicitActuatorCfg,
        sim_utils,
    )
    from active_adaptation.assets.cooked import cooked_model_dir

    model_dir = cooked_model_dir("as2", usd=True)
    asset_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(model_dir / "usd" / "as2.usdc"),
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
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=INIT_POS,
            joint_pos=INIT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hip": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint"],
                effort_limit_sim=HIP_EFFORT_LIMIT,
                velocity_limit_sim=LEGS_VELOCITY_LIMIT,
                stiffness=LEGS_STIFFNESS,
                damping=LEGS_DAMPING,
                friction=0.01,
                armature=0.026,
            ),
            "thigh": ImplicitActuatorCfg(
                joint_names_expr=[".*_thigh_joint"],
                effort_limit_sim=THIGH_EFFORT_LIMIT,
                velocity_limit_sim=LEGS_VELOCITY_LIMIT,
                stiffness=LEGS_STIFFNESS,
                damping=LEGS_DAMPING,
                friction=0.01,
                armature=0.026,
            ),
            "calf": ImplicitActuatorCfg(
                joint_names_expr=[".*_calf_joint"],
                effort_limit_sim=CALF_EFFORT_LIMIT,
                velocity_limit_sim=LEGS_VELOCITY_LIMIT,
                stiffness=LEGS_STIFFNESS,
                damping=LEGS_DAMPING,
                friction=0.01,
                armature=0.038,
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
    return AssetSpec(config=asset_cfg, sensors=sensors)


def make_mjlab_cfg():
    from active_adaptation.assets.asset_cfg import AssetSpec, EntityCfg
    from active_adaptation.assets.cooked import cooked_model_dir
    from active_adaptation.assets.quadrupeds.spot import load_mjcf
    from mjlab.actuator import BuiltinPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    model_dir = cooked_model_dir("as2", usd=False)

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
                    target_names_expr=(".*_hip_joint",),
                    effort_limit=HIP_EFFORT_LIMIT,
                    stiffness=LEGS_STIFFNESS,
                    damping=LEGS_DAMPING,
                    armature=0.026,
                    frictionloss=0.01,
                ),
                BuiltinPdActuatorCfg(
                    target_names_expr=(".*_thigh_joint",),
                    effort_limit=THIGH_EFFORT_LIMIT,
                    stiffness=LEGS_STIFFNESS,
                    damping=LEGS_DAMPING,
                    armature=0.026,
                    frictionloss=0.01,
                ),
                BuiltinPdActuatorCfg(
                    target_names_expr=(".*_calf_joint",),
                    effort_limit=CALF_EFFORT_LIMIT,
                    stiffness=LEGS_STIFFNESS,
                    damping=LEGS_DAMPING,
                    armature=0.038,
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
    return AssetSpec(config=cfg, sensors=sensors)


def make_cfg(backend: Literal["isaaclab", "mjlab"]):
    if backend == "isaaclab":
        return make_isaaclab_cfg()
    if backend == "mjlab":
        return make_mjlab_cfg()
    raise ValueError(f"Invalid backend: {backend}")


registry.register("asset", "unitree_as2", make_cfg)

"""Boston Dynamics Spot (no arm).

MJCF/USD: ``<ROBOT_MODEL_DIR>/spot/``, cooked from the assetx recipe ``spot``
(``aa-cook-assets spot``). Feet are fixed child bodies ``*_foot`` of the lower legs, with the origin at
the contact sphere center (geom ``*_foot_collision0``, r=0.036).
"""

from typing import Literal

from active_adaptation.assets.cooked import cooked_model_dir
from active_adaptation.registry import Registry
from active_adaptation.utils.symmetry import mirrored

registry = Registry.instance()

# Menagerie home keyframe stands at z=0.46 with the feet already in contact.
# Spawn slightly above that so the first physics step is not a penetration spike.
INIT_POS = (0.0, 0.0, 0.60)
INIT_JOINT_POS = {
    ".*_hx": 0.0,
    ".*_hy": 1.04,
    ".*_kn": -1.8,
}

JOINT_SYMMETRY_MAPPING = mirrored({
    "fl_hx": (-1, "fr_hx"),
    "hl_hx": (-1, "hr_hx"),
    "fl_hy": (1, "fr_hy"),
    "hl_hy": (1, "hr_hy"),
    "fl_kn": (1, "fr_kn"),
    "hl_kn": (1, "hr_kn"),
})

SPATIAL_SYMMETRY_MAPPING = mirrored({
    "fl_hip": "fr_hip",
    "hl_hip": "hr_hip",
    "fl_uleg": "fr_uleg",
    "hl_uleg": "hr_uleg",
    "fl_lleg": "fr_lleg",
    "hl_lleg": "hr_lleg",
    "fl_foot": "fr_foot",
    "hl_foot": "hr_foot",
    "base_link": "base_link",
})

JOINT_NAMES_SIMULATION = [
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
]

BODY_NAMES_SIMULATION = [
    "base_link",
    "fl_hip",
    "fr_hip",
    "hl_hip",
    "hr_hip",
    "fl_uleg",
    "fr_uleg",
    "hl_uleg",
    "hr_uleg",
    "fl_lleg",
    "fr_lleg",
    "hl_lleg",
    "hr_lleg",
    "fl_foot",
    "fr_foot",
    "hl_foot",
    "hr_foot",
]

# Isaac Lab Spot hip torque; knee is higher through the linkage (lookup peaks ~110).
HIP_EFFORT_LIMIT = 45.0
KNEE_EFFORT_LIMIT = 90.0
LEGS_VELOCITY_LIMIT = 20.0
LEGS_STIFFNESS = 60.0
LEGS_DAMPING = 1.5

# Sphere feet. Mesh colliders stay frictionless (condim=1).
_FOOT_GEOM = ".*_foot_collision.*"


def load_mjcf(path):
    """Load an assetx MJCF and drop vendor keyframes.

    mjlab keeps only ``spec.keys[0]``. The menagerie home key would otherwise
    override ``INIT_POS`` / ``INIT_JOINT_POS``.
    """
    import mujoco

    spec = mujoco.MjSpec.from_file(str(path))
    while spec.keys:
        spec.delete(spec.keys[0])
    spec.nkey = 0
    return spec


def _mjlab_collisions():
    from mjlab.utils.spec_config import CollisionCfg

    return (
        CollisionCfg(
            geom_names_expr=(".*_collision.*",),
            contype=0,
            conaffinity=1,
            solref=(0.004, 1),
            condim={_FOOT_GEOM: 6, ".*": 1},
            priority={_FOOT_GEOM: 1, ".*": 0},
            friction={_FOOT_GEOM: (1.0, 0.02, 0.01)},
            # The MJCF ramps impedance over the whole 36 mm radius, so the
            # shin mesh reaches the ground. A rubber pad stiffens in a few mm.
            solimp={_FOOT_GEOM: (0.9, 0.95, 0.003)},
            # Inflate the sphere so force starts 5 mm before geometric contact.
            # gap stays 0: the margin band is an active contact, not a detection buffer.
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

    model_dir = cooked_model_dir("spot", usd=True)
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
    from mjlab.actuator import BuiltinPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    model_dir = cooked_model_dir("spot", usd=False)

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


registry.register("asset", "spot", make_cfg)

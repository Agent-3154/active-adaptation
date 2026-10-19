"""Unitree G1 for the mjlab backend, using the XML shipped with mjlab.

Isaac is not supported: ``g1_waist_unlocked`` remains the USD asset.
Joint and body simulation order matches that asset so MDP tensors line up.
"""

from __future__ import annotations

from typing import Literal

from active_adaptation.assets.humanoids.g1 import (
    BODY_NAMES_SIMULATION,
    INIT_JOINT_POS,
    INIT_POS,
    JOINT_NAMES_SIMULATION,
    JOINT_SYMMETRY_MAPPING,
    SPATIAL_SYMMETRY_MAPPING,
)
from active_adaptation.registry import Registry

registry = Registry.instance()

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


def make_isaaclab_cfg():
    raise NotImplementedError(
        "g1_mjlab uses the MJCF shipped with mjlab. Use g1_waist_unlocked on Isaac."
    )


def make_mjlab_cfg():
    import mujoco
    from mjlab import MJLAB_SRC_PATH
    from mjlab.actuator import BuiltinPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.utils.actuator import reflected_inertia_from_two_stage_planetary
    from mjlab.utils.spec_config import CollisionCfg

    from active_adaptation.assets.asset_cfg import AssetSpec, EntityCfg

    xml_path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"

    armature_5020 = reflected_inertia_from_two_stage_planetary(
        (0.139e-4, 0.017e-4, 0.169e-4),
        (1, 1 + (46 / 18), 1 + (56 / 16)),
    )
    armature_7520_14 = reflected_inertia_from_two_stage_planetary(
        (0.489e-4, 0.098e-4, 0.533e-4),
        (1, 4.5, 1 + (48 / 22)),
    )
    armature_7520_22 = reflected_inertia_from_two_stage_planetary(
        (0.489e-4, 0.109e-4, 0.738e-4),
        (1, 4.5, 5),
    )
    armature_4010 = reflected_inertia_from_two_stage_planetary(
        (0.068e-4, 0.0, 0.0),
        (1, 5, 5),
    )

    def spec_fn():
        return mujoco.MjSpec.from_file(str(xml_path))

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
                    target_names_expr=(
                        ".*_elbow_joint",
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_wrist_roll_joint",
                    ),
                    effort_limit=25.0,
                    stiffness=_stiffness(armature_5020),
                    damping=_damping(armature_5020),
                    armature=armature_5020,
                ),
                BuiltinPdActuatorCfg(
                    target_names_expr=(
                        ".*_hip_pitch_joint",
                        ".*_hip_yaw_joint",
                        "waist_yaw_joint",
                    ),
                    effort_limit=88.0,
                    stiffness=_stiffness(armature_7520_14),
                    damping=_damping(armature_7520_14),
                    armature=armature_7520_14,
                ),
                BuiltinPdActuatorCfg(
                    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
                    effort_limit=139.0,
                    stiffness=_stiffness(armature_7520_22),
                    damping=_damping(armature_7520_22),
                    armature=armature_7520_22,
                ),
                BuiltinPdActuatorCfg(
                    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
                    effort_limit=5.0,
                    stiffness=_stiffness(armature_4010),
                    damping=_damping(armature_4010),
                    armature=armature_4010,
                ),
                # Waist pitch/roll and ankles are 4-bar linkages driven by two 5020s.
                BuiltinPdActuatorCfg(
                    target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
                    effort_limit=50.0,
                    stiffness=_stiffness(armature_5020) * 2,
                    damping=_damping(armature_5020) * 2,
                    armature=armature_5020 * 2,
                ),
                BuiltinPdActuatorCfg(
                    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
                    effort_limit=50.0,
                    stiffness=_stiffness(armature_5020) * 2,
                    damping=_damping(armature_5020) * 2,
                    armature=armature_5020 * 2,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*_collision",),
                contype=0,
                conaffinity=1,
                condim={
                    r"^(left|right)_foot[1-7]_collision$": 3,
                    ".*_collision": 1,
                },
                priority={r"^(left|right)_foot[1-7]_collision$": 1, ".*": 0},
                friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
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
    return AssetSpec(config=cfg, sensors=sensors)


def make_cfg(backend: Literal["isaaclab", "mjlab"]):
    if backend == "mjlab":
        return make_mjlab_cfg()
    if backend == "isaaclab":
        return make_isaaclab_cfg()
    raise ValueError(f"Invalid backend: {backend}")


registry.register("asset", "g1_mjlab", make_cfg)

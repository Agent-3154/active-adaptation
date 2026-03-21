"""Unitree A2 + AgileX Piper arm — asset config.

MJCF source: place ``aa-projects/assetx/artifacts/a2_piper/`` (``model.xml`` + ``meshes/``)
under ``<repo>/.cache/aa-robot-models/a2_piper/`` (same layout as the Hugging Face bundle).

USD for Isaac must be provided as ``a2_piper.usd`` in that folder (export separately if needed).
"""

from __future__ import annotations

from dataclasses import replace

from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.assets.asset_cfg import ActuatorCfg
from active_adaptation.assets.quadruped import UNITREE_A2_CFG
from active_adaptation.registry import Registry

registry = Registry.instance()

_ARM_JOINTS = tuple(f"arm_joint{i}" for i in range(1, 9))
_ARM_BODIES = (
    "arm_base_link",
    "arm_link1",
    "arm_link2",
    "arm_link3",
    "arm_link4",
    "arm_link5",
    "gripper_base",
    "gripper_right",
    "gripper_left",
)

def _a2_piper_cfg():
    base = UNITREE_A2_CFG

    init_joint_pos = dict(base.init_state.joint_pos)
    init_joint_pos["arm_joint[1-8]"] = 0.0
    init_state = replace(base.init_state, joint_pos=init_joint_pos)

    arm_actuator = ActuatorCfg(
        joint_names_expr="arm_joint[1-8]",
        effort_limit=80.0,
        velocity_limit=20.0,
        stiffness=50.0,
        damping=2.0,
        friction=0.01,
        armature=0.01,
    )
    actuators = dict(base.actuators)
    actuators["arm"] = arm_actuator

    # Symmetry: legs keep A2 left–right pairs; single arm joints map to themselves (+1).
    joint_symmetry = dict(base.joint_symmetry_mapping)
    for jn in _ARM_JOINTS:
        joint_symmetry[jn] = (1, jn)

    spatial_symmetry = dict(base.spatial_symmetry_mapping)
    for bn in _ARM_BODIES:
        spatial_symmetry[bn] = bn

    joint_names_simulation = list(base.joint_names_simulation) + list(_ARM_JOINTS)
    body_names_simulation = list(base.body_names_simulation) + list(_ARM_BODIES)

    return replace(
        base,
        mjcf_path=ROBOT_MODEL_DIR / "a2_piper" / "model.xml",
        usd_path=ROBOT_MODEL_DIR / "a2_piper" / "a2_piper.usd",
        init_state=init_state,
        actuators=actuators,
        joint_symmetry_mapping=joint_symmetry,
        spatial_symmetry_mapping=spatial_symmetry,
        joint_names_simulation=joint_names_simulation,
        body_names_simulation=body_names_simulation,
    )


UNITREE_A2_PIPER_CFG = _a2_piper_cfg()
registry.register("asset", "unitree_a2_piper", UNITREE_A2_PIPER_CFG)

__all__ = ["UNITREE_A2_PIPER_CFG"]

import os
import copy
import torch

from .quadruped import *
from .humanoid import *
from .scene import *
from .sirius import *
from .registry import AssetRegistry


registry = AssetRegistry.instance()
registry.register("go2", UNITREE_GO2_CFG)
registry.register("aliengo", UNITREE_ALIENGO_CFG)
registry.register("sirius_wheel", SIRIUS_WHEEL_CFG)
registry.register("b1z1", UNITREE_B1Z1_CFG)

def get_asset_meta(asset: Articulation):
    if not asset.is_initialized:
        raise RuntimeError("Articulation is not initialized. Please wait until `sim.reset` is called.")
    meta = {
        "init_state": asset.cfg.init_state.to_dict(),
        "body_names_isaac": asset.body_names,
        "joint_names_isaac": asset.joint_names,
        "actuators": {},
    }
    if asset.is_initialized: # parsed values
        meta["default_joint_pos"] = asset.data.default_joint_pos[0].tolist()
        meta["stiffness"] = asset.data.joint_stiffness[0].tolist()
        meta["damping"] = asset.data.joint_damping[0].tolist()

    for actuator_name, actuator in asset.actuators.items():
        meta["actuators"][actuator_name] = actuator.cfg.to_dict()
    return meta


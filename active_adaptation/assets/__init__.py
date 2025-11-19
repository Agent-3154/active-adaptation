from .asset_cfg import (
    AssetCfg,
    InitialStateCfg,
    ActuatorCfg,
    ContactSensorCfg,
    get_input_joint_indexing,
    get_output_joint_indexing,
    get_output_body_indexing,
)
from . import quadruped
# from . import humanoid

__all__ = [
    "AssetCfg",
    "InitialStateCfg",
    "ActuatorCfg",
    "ContactSensorCfg",
    "get_input_joint_indexing",
    "get_output_joint_indexing",
    "get_output_body_indexing",
]


# def get_asset_meta(asset: Articulation):
#     if not asset.is_initialized:
#         raise RuntimeError("Articulation is not initialized. Please wait until `sim.reset` is called.")
#     meta = {
#         "init_state": asset.cfg.init_state.to_dict(),
#         "body_names_isaac": asset.body_names,
#         "joint_names_isaac": asset.joint_names,
#         "actuators": {},
#     }
#     if asset.is_initialized: # parsed values
#         meta["default_joint_pos"] = asset.data.default_joint_pos[0].tolist()
#         meta["stiffness"] = asset.data.joint_stiffness[0].tolist()
#         meta["damping"] = asset.data.joint_damping[0].tolist()

#     for actuator_name, actuator in asset.actuators.items():
#         meta["actuators"][actuator_name] = actuator.cfg.to_dict()
#     return meta


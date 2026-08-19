"""Isaac Sim 6.0.1 Fabric transform-sync workaround (IsaacSim#740).

Prior to 6.0.1, Fabric transform sync was ignored in headless. In 6.0.1 the
cost grows linearly with env count. PhysX reads ``carb.settings``. Apply after
``SimulationApp`` and again after ``SimulationContext`` loads ``omni.physx.fabric``.
"""

from __future__ import annotations

from typing import Mapping

from active_adaptation.envs.utils.quat_layout import isaaclab_uses_xyzw

_FABRIC_UPDATE_TRANSFORMS = "/physics/fabricUpdateTransformations"
_FABRIC_GPU_INTEROP = "/physics/fabricUseGPUInterop"


def app_config_needs_fabric_transforms(app_config: Mapping) -> bool:
    """Match Isaac Lab ``AppLauncher._rendering_enabled()``."""
    headless = bool(app_config.get("headless", True))
    enable_cameras = bool(app_config.get("enable_cameras", False))
    livestream = int(app_config.get("livestream", 0) or 0)
    xr = bool(app_config.get("xr", False))
    return (not headless) or enable_cameras or livestream >= 1 or xr


def apply_isaacsim_740_fabric_workaround(
    *, rendering_needed: bool, log: bool = False
) -> None:
    """Set Fabric transform sync for Sim 6 / Lab 3. No-op on Lab 2 / Sim 5.1."""
    if not isaaclab_uses_xyzw():
        return
    import carb.settings

    settings = carb.settings.get_settings()
    # NVIDIA: disable when headless / transforms are unused.
    settings.set_bool(_FABRIC_UPDATE_TRANSFORMS, bool(rendering_needed))
    # NVIDIA: if transform sync stays on and env count > 64, GPU interop is required.
    settings.set_bool(_FABRIC_GPU_INTEROP, True)
    if log:
        print(
            f"[IsaacSim#740] {_FABRIC_UPDATE_TRANSFORMS}={bool(rendering_needed)} "
            f"{_FABRIC_GPU_INTEROP}=True"
        )

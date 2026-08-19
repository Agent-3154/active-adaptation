"""Isaac Lab 2 passthrough / Lab 3 canonical (torch + WXYZ) asset façade.

MDP terms keep calling ``asset.data.root_link_quat_w`` and
``write_root_state_to_sim``. On Lab 3 this wrapper:

- unwraps ``ProxyArray`` via ``.torch``
- rolls XYZW quaternions / packed pose-state slices to WXYZ on read
- rolls WXYZ → XYZW on pose writes and dispatches ``*_index`` writers
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from active_adaptation.envs.utils.quat_layout import (
    POSE_FIELDS,
    QUAT_FIELDS,
    STATE_FIELDS,
    isaaclab_uses_xyzw,
    pose_wxyz_to_xyzw,
    pose_xyzw_to_wxyz,
    state_wxyz_to_xyzw,
    state_xyzw_to_wxyz,
    xyzw_to_wxyz,
)


def as_torch(value: Any) -> Any:
    """Return a torch tensor from Lab 3 ``ProxyArray`` / warp arrays, else ``value``."""
    if value is None or isinstance(value, torch.Tensor):
        return value
    torch_view = getattr(value, "torch", None)
    if isinstance(torch_view, torch.Tensor):
        return torch_view
    if type(value).__module__.startswith("warp") or type(value).__name__ in ("array", "ProxyArray"):
        import warp as wp

        return wp.to_torch(value)
    return value


def isaac_root_view(asset: Any) -> Any:
    """PhysX tensor view: Lab 3 ``root_view``, Lab 2 ``root_physx_view``."""
    native = getattr(asset, "_asset", asset)
    if isaaclab_uses_xyzw():
        return native.root_view
    return native.root_physx_view


def _as_int32_index(value: Any) -> Any:
    """Lab 3 Warp index kernels want int32, not torch.long."""
    if isinstance(value, torch.Tensor) and value.dtype == torch.int64:
        return value.to(dtype=torch.int32)
    return value


class CanonicalData:
    """Presents AA-canonical torch WXYZ views over native asset/sensor data."""

    def __init__(self, src: Any, xyzw: bool):
        self._src = src
        self._xyzw = xyzw

    def __getattr__(self, name: str) -> Any:
        value = as_torch(getattr(self._src, name))
        if callable(value) and name not in QUAT_FIELDS | POSE_FIELDS | STATE_FIELDS:
            return value
        if not self._xyzw or not isinstance(value, torch.Tensor) or value.ndim == 0:
            return value
        if name in QUAT_FIELDS and value.shape[-1] == 4:
            return xyzw_to_wxyz(value)
        if name in POSE_FIELDS and value.shape[-1] >= 7:
            return pose_xyzw_to_wxyz(value)
        if name in STATE_FIELDS and value.shape[-1] >= 7:
            return state_xyzw_to_wxyz(value)
        return value


class CanonicalIsaacAsset:
    """Articulation / rigid-object proxy: WXYZ torch reads, Lab 3 ``_index`` writes."""

    def __init__(self, asset: Any, xyzw: bool):
        object.__setattr__(self, "_asset", asset)
        object.__setattr__(self, "_xyzw", xyzw)
        object.__setattr__(self, "data", CanonicalData(asset.data, xyzw))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._asset, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_asset", "_xyzw", "data"):
            object.__setattr__(self, name, value)
            return
        setattr(self._asset, name, value)

    def _coerce_env_ids(self, env_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if env_ids is not None:
            return _as_int32_index(env_ids)
        num_envs = int(self._asset.num_instances)
        device = getattr(self._asset, "device", None)
        if device is None and hasattr(self._asset, "data"):
            pos = as_torch(getattr(self._asset.data, "root_link_pos_w", None))
            if isinstance(pos, torch.Tensor):
                device = pos.device
                num_envs = pos.shape[0]
        return torch.arange(num_envs, device=device, dtype=torch.int32)

    def _call_write_index(
        self,
        stem: str,
        payload_kw: str,
        payload: torch.Tensor,
        env_ids: Optional[torch.Tensor],
        **kwargs,
    ) -> None:
        asset = self._asset
        index_fn = getattr(asset, f"{stem}_index", None)
        if index_fn is None:
            raise AttributeError(
                f"Isaac Lab 3 {type(asset).__name__} is missing {stem}_index"
            )
        env_ids = self._coerce_env_ids(env_ids)
        kwargs = {
            key: _as_int32_index(val) if key.endswith("_ids") else val
            for key, val in kwargs.items()
        }
        index_fn(**{payload_kw: payload, "env_ids": env_ids}, **kwargs)

    def write_root_pose_to_sim(
        self, root_pose: torch.Tensor, env_ids: Optional[torch.Tensor] = None
    ) -> None:
        native = pose_wxyz_to_xyzw(root_pose) if self._xyzw else root_pose
        self._call_write_index("write_root_pose_to_sim", "root_pose", native, env_ids)

    def write_root_link_pose_to_sim(
        self, root_pose: torch.Tensor, env_ids: Optional[torch.Tensor] = None
    ) -> None:
        native = pose_wxyz_to_xyzw(root_pose) if self._xyzw else root_pose
        self._call_write_index(
            "write_root_link_pose_to_sim", "root_pose", native, env_ids
        )

    def write_root_velocity_to_sim(
        self, root_velocity: torch.Tensor, env_ids: Optional[torch.Tensor] = None
    ) -> None:
        self._call_write_index(
            "write_root_velocity_to_sim", "root_velocity", root_velocity, env_ids
        )

    def write_root_link_velocity_to_sim(
        self, root_velocity: torch.Tensor, env_ids: Optional[torch.Tensor] = None
    ) -> None:
        self._call_write_index(
            "write_root_link_velocity_to_sim",
            "root_velocity",
            root_velocity,
            env_ids,
        )

    def write_root_state_to_sim(
        self, root_state: torch.Tensor, env_ids: Optional[torch.Tensor] = None
    ) -> None:
        native = state_wxyz_to_xyzw(root_state) if self._xyzw else root_state
        pose = native[..., :7]
        vel = native[..., 7:13]
        self._call_write_index("write_root_pose_to_sim", "root_pose", pose, env_ids)
        self._call_write_index(
            "write_root_velocity_to_sim", "root_velocity", vel, env_ids
        )

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        extra = {} if joint_ids is None else {"joint_ids": joint_ids}
        self._call_write_index(
            "write_joint_position_to_sim",
            "position",
            position,
            env_ids,
            **extra,
        )
        self._call_write_index(
            "write_joint_velocity_to_sim",
            "velocity",
            velocity,
            env_ids,
            **extra,
        )

    def write_joint_position_to_sim(
        self,
        position: torch.Tensor,
        joint_ids: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        extra = {} if joint_ids is None else {"joint_ids": joint_ids}
        self._call_write_index(
            "write_joint_position_to_sim", "position", position, env_ids, **extra
        )

    def write_joint_velocity_to_sim(
        self,
        velocity: torch.Tensor,
        joint_ids: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        extra = {} if joint_ids is None else {"joint_ids": joint_ids}
        self._call_write_index(
            "write_joint_velocity_to_sim", "velocity", velocity, env_ids, **extra
        )

    def write_joint_stiffness_to_sim(
        self,
        stiffness: torch.Tensor,
        joint_ids: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        extra = {} if joint_ids is None else {"joint_ids": joint_ids}
        self._call_write_index(
            "write_joint_stiffness_to_sim", "stiffness", stiffness, env_ids, **extra
        )

    def write_joint_damping_to_sim(
        self,
        damping: torch.Tensor,
        joint_ids: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        extra = {} if joint_ids is None else {"joint_ids": joint_ids}
        self._call_write_index(
            "write_joint_damping_to_sim", "damping", damping, env_ids, **extra
        )

    def set_joint_position_target(
        self,
        target: torch.Tensor,
        joint_ids: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        extra = {} if joint_ids is None else {"joint_ids": joint_ids}
        self._call_write_index(
            "set_joint_position_target", "target", target, env_ids, **extra
        )

    def set_joint_velocity_target(
        self,
        target: torch.Tensor,
        joint_ids: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        extra = {} if joint_ids is None else {"joint_ids": joint_ids}
        self._call_write_index(
            "set_joint_velocity_target", "target", target, env_ids, **extra
        )

    def set_joint_effort_target(
        self,
        target: torch.Tensor,
        joint_ids: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        extra = {} if joint_ids is None else {"joint_ids": joint_ids}
        self._call_write_index(
            "set_joint_effort_target", "target", target, env_ids, **extra
        )


class CanonicalIsaacSensor:
    """Sensor proxy: ``.data`` fields as torch; roll IMU/frame quats on Lab 3."""

    def __init__(self, sensor: Any, xyzw: bool):
        object.__setattr__(self, "_sensor", sensor)
        object.__setattr__(self, "data", CanonicalData(sensor.data, xyzw))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sensor, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_sensor", "data"):
            object.__setattr__(self, name, value)
            return
        setattr(self._sensor, name, value)

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        value = as_torch(self._sensor.compute_first_contact(dt, abs_tol=abs_tol))
        if isinstance(value, torch.Tensor) and value.dtype != torch.bool:
            value = value != 0
        return value

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        value = as_torch(self._sensor.compute_first_air(dt, abs_tol=abs_tol))
        if isinstance(value, torch.Tensor) and value.dtype != torch.bool:
            value = value != 0
        return value


def maybe_wrap_asset(asset: Any) -> Any:
    """Lab 3 → ``CanonicalIsaacAsset``; Lab 2 returns ``asset`` unchanged."""
    if isinstance(asset, CanonicalIsaacAsset):
        return asset
    if not isaaclab_uses_xyzw():
        return asset
    if getattr(asset, "data", None) is None:
        raise RuntimeError(
            f"Isaac Lab 3 asset {type(asset).__name__} has no .data to wrap"
        )
    return CanonicalIsaacAsset(asset, xyzw=True)


def maybe_wrap_sensor(sensor: Any) -> Any:
    """Lab 3 → ``CanonicalIsaacSensor``; Lab 2 returns ``sensor`` unchanged."""
    if isinstance(sensor, CanonicalIsaacSensor):
        return sensor
    if not isaaclab_uses_xyzw():
        return sensor
    if getattr(sensor, "data", None) is None:
        raise RuntimeError(
            f"Isaac Lab 3 sensor {type(sensor).__name__} has no .data to wrap"
        )
    return CanonicalIsaacSensor(sensor, xyzw=True)

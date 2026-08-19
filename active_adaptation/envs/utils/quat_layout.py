"""Quaternion layout helpers for Isaac Lab 2 (WXYZ) vs Lab 3 (XYZW).

AA MDP, ``utils/math.py``, mjlab, Viser, and checkpoints stay WXYZ. Convert
only at the Isaac adapter / cfg boundary.
"""
from __future__ import annotations

from typing import Sequence

import torch

_LAB3_XYZW: bool | None = None

QUAT_FIELDS = frozenset({
    "root_link_quat_w",
    "root_quat_w",
    "root_com_quat_w",
    "body_link_quat_w",
    "body_quat_w",
    "body_com_quat_w",
    "body_com_quat_b",
    "com_quat_b",
    "quat_w",
    "target_quat_w",
})

POSE_FIELDS = frozenset({
    "root_link_pose_w",
    "root_com_pose_w",
    "root_pose_w",
    "body_link_pose_w",
    "body_com_pose_w",
    "body_pose_w",
    "pose_w",
    "target_pose_w",
})

STATE_FIELDS = frozenset({
    "root_link_state_w",
    "root_state_w",
    "root_com_state_w",
    "default_root_state",
})


def xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """``(x, y, z, w)`` → ``(w, x, y, z)``."""
    return torch.roll(quat, 1, dims=-1)


def wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """``(w, x, y, z)`` → ``(x, y, z, w)``."""
    return torch.roll(quat, -1, dims=-1)


def _roll_quat_slice(packed: torch.Tensor, *, to_wxyz: bool) -> torch.Tensor:
    """Roll the quaternion columns ``[..., 3:7]`` of a packed pose/state tensor."""
    quat = packed[..., 3:7]
    rolled = xyzw_to_wxyz(quat) if to_wxyz else wxyz_to_xyzw(quat)
    return torch.cat([packed[..., :3], rolled, packed[..., 7:]], dim=-1)


def pose_xyzw_to_wxyz(pose: torch.Tensor) -> torch.Tensor:
    return _roll_quat_slice(pose, to_wxyz=True)


def pose_wxyz_to_xyzw(pose: torch.Tensor) -> torch.Tensor:
    return _roll_quat_slice(pose, to_wxyz=False)


def state_xyzw_to_wxyz(state: torch.Tensor) -> torch.Tensor:
    return _roll_quat_slice(state, to_wxyz=True)


def state_wxyz_to_xyzw(state: torch.Tensor) -> torch.Tensor:
    return _roll_quat_slice(state, to_wxyz=False)


def isaaclab_uses_xyzw() -> bool:
    """True on Isaac Lab 3 (``convert_quat`` removed; APIs are XYZW)."""
    global _LAB3_XYZW
    if _LAB3_XYZW is not None:
        return _LAB3_XYZW
    try:
        import isaaclab.utils.math as math_utils
    except ImportError:
        _LAB3_XYZW = False
        return False
    _LAB3_XYZW = not hasattr(math_utils, "convert_quat")
    return _LAB3_XYZW


def isaac_cfg_quat(wxyz: Sequence[float]) -> tuple[float, ...]:
    """AA-canonical WXYZ → native Isaac cfg quaternion."""
    wxyz_t = tuple(float(x) for x in wxyz)
    if len(wxyz_t) != 4:
        raise ValueError(f"Expected a 4-tuple quaternion, got {wxyz!r}")
    if not isaaclab_uses_xyzw():
        return wxyz_t
    w, x, y, z = wxyz_t
    return (x, y, z, w)


def self_check() -> None:
    """Torch-only layout round-trips. No Isaac import required."""
    wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.70710678, 0.0, 0.0, 0.70710678]])
    xyzw = wxyz_to_xyzw(wxyz)
    assert torch.allclose(xyzw[0], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(xyzw_to_wxyz(xyzw), wxyz)
    pose_wxyz = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
    pose_xyzw = pose_wxyz_to_xyzw(pose_wxyz)
    assert torch.allclose(pose_xyzw[0, :3], pose_wxyz[0, :3])
    assert torch.allclose(pose_xyzw[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(pose_xyzw_to_wxyz(pose_xyzw), pose_wxyz)
    state_wxyz = torch.zeros(2, 13)
    state_wxyz[:, 3] = 1.0
    state_wxyz[:, 7:] = torch.arange(12).reshape(2, 6).float()
    state_xyzw = state_wxyz_to_xyzw(state_wxyz)
    assert torch.allclose(state_xyzw[:, 7:], state_wxyz[:, 7:])
    assert torch.allclose(state_xyzw_to_wxyz(state_xyzw), state_wxyz)


if __name__ == "__main__":
    self_check()
    print("quat_layout self_check ok")

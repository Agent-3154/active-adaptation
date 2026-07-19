"""Shared Warp helpers for loco-manip world-goal sampling / updates.

Used by ``LocoManipNew`` (mode 0) and ``LocoManipSparse`` (goal-reaching) so the
polar world goal and heading-frame EEF refresh stay bit-identical.
"""

from __future__ import annotations

import torch
import warp as wp


def quat_wxyz_to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Torch / Isaac (w, x, y, z) → Warp (x, y, z, w)."""
    return quat_wxyz[:, [1, 2, 3, 0]].contiguous()


@wp.func
def sample_world_goal(
    seed: wp.uint32,
    root_pos_w: wp.vec3,
    root_yaw_quat: wp.quat,
    eef_z_min: float,
    eef_z_max: float,
    world_radius_min: float,
    world_radius_max: float,
    standoff_reach_min: float,
    standoff_reach_max: float,
    linvel_gain_min: float,
    linvel_gain_max: float,
    yaw_gain_min: float,
    yaw_gain_max: float,
):
    """Sample a persistent world EEF goal + standoff about the current root.

    Returns:
        (seed, cmd_eef_pos_w, standoff_pos_w, standoff_yaw_w, linvel_gain, yaw_gain)
    """
    oz = wp.randf(seed, eef_z_min, eef_z_max)
    alpha = wp.randf(seed, 0.0, 2.0 * wp.pi)
    radius = wp.randf(seed, world_radius_min, world_radius_max)
    offset_xy = wp.vec3(radius * wp.cos(alpha), radius * wp.sin(alpha), 0.0)
    root_xy = wp.vec3(root_pos_w[0], root_pos_w[1], 0.0)
    world_xy = root_xy + wp.quat_rotate(root_yaw_quat, offset_xy)
    cmd_eef_pos_w = wp.vec3(world_xy[0], world_xy[1], oz)

    goal_xy = wp.vec3(world_xy[0], world_xy[1], 0.0)
    to_goal = goal_xy - root_xy
    dist = wp.sqrt(to_goal[0] * to_goal[0] + to_goal[1] * to_goal[1])
    reach = wp.randf(seed, standoff_reach_min, standoff_reach_max)
    if dist > reach:
        inv_dist = 1.0 / dist
        standoff_xy = goal_xy - wp.vec3(
            to_goal[0] * inv_dist * reach,
            to_goal[1] * inv_dist * reach,
            0.0,
        )
    else:
        standoff_xy = root_xy
    standoff_yaw_w = wp.atan2(
        goal_xy[1] - standoff_xy[1], goal_xy[0] - standoff_xy[0]
    )
    linvel_gain = wp.randf(seed, linvel_gain_min, linvel_gain_max)
    yaw_gain = wp.randf(seed, yaw_gain_min, yaw_gain_max)
    return (
        seed,
        cmd_eef_pos_w,
        standoff_xy,
        standoff_yaw_w,
        linvel_gain,
        yaw_gain,
    )


@wp.func
def update_world_command(
    root_pos_w: wp.vec3,
    root_yaw_quat: wp.quat,
    heading_w: float,
    cmd_eef_pos_w: wp.vec3,
    standoff_pos_w: wp.vec3,
    standoff_yaw_w: float,
    world_linvel_gain: float,
    world_yaw_gain: float,
    linvel_x_min: float,
    linvel_x_max: float,
    linvel_y_min: float,
    linvel_y_max: float,
    yaw_rate_min: float,
    yaw_rate_max: float,
):
    """Refresh heading-frame EEF cmd + standoff loco from a persistent world goal.

    Returns:
        (cmd_eef_pos_b, cmd_linvel_b, cmd_yawvel_b, base_pos_error)
    """
    root_xy = wp.vec3(root_pos_w[0], root_pos_w[1], 0.0)

    eef_delta_w = cmd_eef_pos_w - root_xy
    cmd_eef_pos_b = wp.quat_rotate_inv(root_yaw_quat, eef_delta_w)

    standoff_delta_w = standoff_pos_w - root_xy
    standoff_delta_xy_w = wp.vec3(standoff_delta_w[0], standoff_delta_w[1], 0.0)
    delta_b = wp.quat_rotate_inv(root_yaw_quat, standoff_delta_xy_w)
    dx = delta_b[0]
    dy = delta_b[1]
    dist = wp.sqrt(dx * dx + dy * dy)

    vx = wp.clamp(world_linvel_gain * dx, linvel_x_min, linvel_x_max)
    vy = wp.clamp(world_linvel_gain * dy, linvel_y_min, linvel_y_max)
    cmd_linvel_b = wp.vec3(vx, vy, 0.0)

    yaw_err = standoff_yaw_w - heading_w
    yaw_err = wp.atan2(wp.sin(yaw_err), wp.cos(yaw_err))
    cmd_yawvel_b = wp.clamp(world_yaw_gain * yaw_err, yaw_rate_min, yaw_rate_max)
    return cmd_eef_pos_b, cmd_linvel_b, cmd_yawvel_b, dist

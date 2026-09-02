# This file contains additional math utilities
# that are not covered by IsaacLab

import torch
import torch.distributions as D

from .helpers import batchify


def rot6d(quat: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix (Zhou et al. 6D representation).

    Args:
        quat: Quaternions ``(..., 4)`` in ``(w, x, y, z)``.
    Returns:
        ``(..., 6)`` as ``[c0_x, c0_y, c0_z, c1_x, c1_y, c1_z]``.
    """
    mat = matrix_from_quat(quat)
    return mat[..., :, :2].reshape(quat.shape[:-1] + (6,))


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    r"""Wraps input angles (in radians) to the range :math:`[-\pi, \pi]`.

    This function wraps angles in radians to the range :math:`[-\pi, \pi]`, such that
    :math:`\pi` maps to :math:`\pi`, and :math:`-\pi` maps to :math:`-\pi`. In general,
    odd positive multiples of :math:`\pi` are mapped to :math:`\pi`, and odd negative
    multiples of :math:`\pi` are mapped to :math:`-\pi`.

    The function behaves similar to MATLAB's `wrapToPi <https://www.mathworks.com/help/map/ref/wraptopi.html>`_
    function.

    Args:
        angles: Input angles of any shape.

    Returns:
        Angles in the range :math:`[-\pi, \pi]`.
    """
    # wrap to [0, 2*pi)
    wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
    # map to [-pi, pi]
    # we check for zero in wrapped angle to make it go to pi when input angle is odd multiple of pi
    return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)


def quat_rotate(quat: torch.Tensor, vec: torch.Tensor):
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    xyz = quat[..., 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[..., 0:1] * t + xyz.cross(t, dim=-1))


def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor):
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    xyz = quat[..., 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[..., 0:1] * t + xyz.cross(t, dim=-1))


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize a d-dimensional vector.

    Args:
        x: The vector to normalize. Shape is (..., d).

    Returns:
        The normalized vector. Shape is (..., d).
    """
    return x / x.norm(dim=-1, keepdim=True).clamp(1e-6)


def clamp_norm(x: torch.Tensor, min: float=0., max: float=torch.inf):
    """Clamp the norm of a d-dimensional vector.

    Args:
        x: The vector to clamp. Shape is (..., d).
        min: The minimum norm.
        max: The maximum norm.
    
    Returns:
        The clamped vector. Shape is (..., d).
    """
    unit = x / (x_norm := x.norm(dim=-1, keepdim=True)).clamp(1e-6)
    x = torch.where(x_norm < min, unit * min, x)
    x = torch.where(x_norm > max, unit * max, x)
    return x

def clamp_along(x: torch.Tensor, axis: torch.Tensor, min: float, max: float):
    projection = (x * axis).sum(dim=-1, keepdim=True)
    return x - projection * axis + projection.clamp(min, max) * axis


def yaw_rotate(yaw: torch.Tensor, vec: torch.Tensor):
    """
    Rotate a vector by a yaw angle (in radians).
    """
    yaw = yaw.reshape(vec.shape[:-1])
    yaw_cos = torch.cos(yaw)
    yaw_sin = torch.sin(yaw)
    vec = vec.expand(*yaw.shape, 3)
    return torch.stack(
        [
            yaw_cos * vec[..., 0] - yaw_sin * vec[..., 1],
            yaw_sin * vec[..., 0] + yaw_cos * vec[..., 1],
            vec[..., 2],
        ],
        dim=-1,
    )


def euler_rotate(rpy: torch.Tensor, vec: torch.Tensor):
    quat = quat_from_euler_xyz(rpy)
    return quat_rotate(quat, vec)


def quat_from_yaw(yaw: torch.Tensor):
    return torch.cat(
        [
            torch.cos(yaw / 2).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.sin(yaw / 2).unsqueeze(-1),
        ],
        dim=-1,
    )


def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.view(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = torch.zeros_like(quat_yaw)
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    q1, q2 = torch.broadcast_tensors(q1, q2)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)


def quat_from_euler_xyz(rpy: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    roll, pitch, yaw = rpy.unbind(-1)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qw, qx, qy, qz], dim=-1)


def euler_from_quat(quat: torch.Tensor):
    w, x, y, z = quat.unbind(-1)
    # Convert quaternion to roll, pitch, yaw Euler angles
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sin_pitch) >= 1,
        torch.full_like(sin_pitch, torch.pi / 2.0) * torch.sign(sin_pitch),
        torch.asin(sin_pitch)
    )

    sin_yaw = 2.0 * (w * z + x * y) 
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    return torch.stack([roll, pitch, yaw], dim=-1)


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as angle-axis to quaternions.

    Args:
        angle: Rotation angle in radians. Shape ``(...,)``.
        axis: Unit rotation axis. Shape ``(..., 3)``.

    Returns:
        Quaternion in ``(w, x, y, z)``. Shape ``(..., 4)``.
    """
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


def apply_delta_pose(
    source_pos: torch.Tensor,
    source_rot: torch.Tensor,
    delta_pose: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a delta pose to a source frame.

    ``delta_pose[:, :3]`` is a Cartesian displacement; ``delta_pose[:, 3:6]`` is
    an orientation increment in axis-angle form (radians).

    Args:
        source_pos: Source position. Shape ``(N, 3)``.
        source_rot: Source orientation ``(w, x, y, z)``. Shape ``(N, 4)``.
        delta_pose: Pose delta. Shape ``(N, 6)``.
        eps: Threshold below which orientation delta is treated as zero.

    Returns:
        ``(target_pos, target_rot)`` with shapes ``(N, 3)`` and ``(N, 4)``.
    """
    target_pos = source_pos + delta_pose[:, :3]
    rot_actions = delta_pose[:, 3:6]
    angle = torch.linalg.vector_norm(rot_actions, dim=1)
    safe_angle = torch.clamp_min(angle, eps)
    axis = rot_actions / safe_angle.unsqueeze(-1)
    identity_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=source_pos.device, dtype=source_pos.dtype
    ).repeat(source_pos.shape[0], 1)
    rot_delta_quat = torch.where(
        angle.unsqueeze(-1).repeat(1, 4) > eps,
        quat_from_angle_axis(angle, axis),
        identity_quat,
    )
    target_rot = quat_mul(rot_delta_quat, source_rot)
    return target_pos, target_rot


def quat_angle_magnitude(quat: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    """Compute the rotation angle represented by a quaternion.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: Clamp for the scalar part to avoid undefined gradients near zero.

    Returns:
        Rotation angle in radians. Shape is (...,).
    """
    xyz_norm = torch.linalg.norm(quat[..., 1:], dim=-1)
    return 2.0 * torch.atan2(xyz_norm, quat[..., 0].abs().clamp_min(eps))


def sample_quat_yaw(size, yaw_range=(0, torch.pi * 2), device: torch.device = "cpu"):
    yaw = torch.rand(size, device=device).uniform_(*yaw_range)
    quat = torch.cat(
        [
            torch.cos(yaw / 2).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.sin(yaw / 2).unsqueeze(-1),
        ],
        dim=-1,
    )
    return quat


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=0.0))


def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions ``(w, x, y, z)``."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quat_from_view_z_up(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Look-at orientation with world +X forward, +Z up (Z-up stage).

    Args:
        eye: Camera positions, shape ``(..., 3)``.
        target: Look-at points, shape ``(..., 3)``.

    Returns:
        Quaternion ``(w, x, y, z)`` with the same batch shape as ``eye``.
    """
    forward = target - eye
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-5)

    world_up = torch.zeros_like(forward)
    world_up[..., 2] = 1.0
    aux_up = torch.zeros_like(forward)
    aux_up[..., 1] = 1.0

    x_axis = forward
    y_axis = torch.cross(world_up, x_axis, dim=-1)
    parallel = y_axis.norm(dim=-1, keepdim=True) < 1e-5
    y_axis = torch.where(parallel, torch.cross(aux_up, x_axis, dim=-1), y_axis)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True).clamp_min(1e-5)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)

    rot = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return quat_from_matrix(rot)


def root_pose_from_view_z_up(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root link pose ``[pos(3), quat(4)]`` from eye/target, Z-up world convention."""
    return torch.cat([eye, quat_from_view_z_up(eye, target)], dim=-1)


def normal_noise(x: torch.Tensor, std: float):
    """
    Add normal noise to a tensor. Clamp the noise to 3 standard deviations for stability.
    """
    return x + torch.randn_like(x).clamp(-3., 3.) * std


def uniform_noise(x: torch.Tensor, min: float, max: float):
    """
    Add uniform noise to a tensor.
    """
    return x + torch.rand_like(x) * (max - min) + min


def slerp(quat0: torch.Tensor, quat1: torch.Tensor, t: torch.Tensor):
    """
    Spherical linear interpolation between two quaternions.
    
    Args:
        quat0: First quaternion tensor
        quat1: Second quaternion tensor
        t: Interpolation factor (0 = quat0, 1 = quat1)
    
    Returns:
        Interpolated quaternion (normalized)
    """
    # Compute dot product
    dot = torch.sum(quat0 * quat1, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # If dot product is negative, negate one quaternion to take shorter path
    # (since q and -q represent the same rotation)
    quat1 = torch.where(dot < 0, -quat1, quat1)
    dot = torch.abs(dot)
    
    # Compute angle
    theta = torch.arccos(dot)
    sin_theta = torch.sin(theta)
    
    # Handle case when quaternions are very close (fallback to linear interpolation)
    # Use a small epsilon to avoid division by zero
    eps = 1e-6
    mask = sin_theta > eps
    
    # Compute interpolation coefficients
    t1 = torch.where(mask, torch.sin((1 - t) * theta) / sin_theta, 1 - t)
    t2 = torch.where(mask, torch.sin(t * theta) / sin_theta, t)
    
    # Interpolate and normalize
    result = quat0 * t1 + quat1 * t2
    return result / torch.linalg.norm(result, dim=-1, keepdim=True)


def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low


def quat_rotate_x_to(direction: torch.Tensor) -> torch.Tensor:
    """WXYZ quaternion that rotates ``+X`` onto ``direction`` (..., 3)."""
    d = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    quat = torch.stack(
        [
            1.0 + d[..., 0],
            torch.zeros_like(d[..., 0]),
            -d[..., 2],
            d[..., 1],
        ],
        dim=-1,
    )
    antiparallel = d[..., 0] < -0.999
    if antiparallel.any():
        quat = quat.clone()
        quat[antiparallel] = quat.new_tensor([0.0, 0.0, 1.0, 0.0])
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class MultiUniform(D.Distribution):
    """
    A distribution over the union of multiple disjoint intervals.
    """
    def __init__(self, ranges: torch.Tensor):
        batch_shape = ranges.shape[:-2]
        if not ranges[..., 0].le(ranges[..., 1]).all():
            raise ValueError("Ranges must be non-empty and ordered.")
        super().__init__(batch_shape, validate_args=False)
        self.ranges = ranges
        self.ranges_len = ranges.diff(dim=-1).squeeze(1)
        self.total_len = self.ranges_len.sum(-1)
        self.starts = torch.zeros_like(ranges[..., 0])
        self.starts[..., 1:] = self.ranges_len.cumsum(-1)[..., :-1]

    def sample(self, sample_shape: torch.Size = ()) -> torch.Tensor:
        sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self.batch_shape
        uniform = torch.rand(shape, device=self.ranges.device) * self.total_len
        i = torch.searchsorted(self.starts, uniform) - 1
        return self.ranges[i, 0] + uniform - self.starts[i]



class EMA:
    """
    Exponential Moving Average.
    
    Args:
        x: The tensor to compute the EMA of.
        gammas: The decay rates. Can be a single float or a list of floats.
    
    Example:
        >>> ema = EMA(x, gammas=[0.9, 0.99])
        >>> ema.update(x)
        >>> ema.ema
    """
    def __init__(self, x: torch.Tensor, gammas):
        self.gammas = torch.tensor(gammas, device=x.device)
        shape = (x.shape[0], len(self.gammas), *x.shape[1:])
        self.sum = torch.zeros(shape, device=x.device)
        shape = (x.shape[0], len(self.gammas), 1)
        self.cnt = torch.zeros(shape, device=x.device)

    def reset(self, env_ids: torch.Tensor):
        self.sum[env_ids] = 0.0
        self.cnt[env_ids] = 0.0
        
    def update(self, x: torch.Tensor):
        self.sum.mul_(self.gammas.unsqueeze(-1)).add_(x.unsqueeze(1))
        self.cnt.mul_(self.gammas.unsqueeze(-1)).add_(1.0)
        self.ema = self.sum / self.cnt
        return self.ema


def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std

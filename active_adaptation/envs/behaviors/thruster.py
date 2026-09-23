"""Pluggable throttle → RPM → axial thrust maps for :class:`UnderwaterRobot`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


class ThrusterForceModel(Protocol):
    """Maps filtered throttle ``[-1, 1]`` to axial force (N).

    ``nominal_force_constant`` is the scale already baked into ``rpm_to_thrust``.
    :class:`~active_adaptation.envs.behaviors.underwater.UnderwaterRobot`
    multiplies by ``force_constants / nominal_force_constant`` so BlueROV's
    per-rotor ``4.4e-7`` / ``0.8e-7`` ratios stay unchanged.
    """

    nominal_force_constant: float

    def throttle_to_rpm(self, throttle: torch.Tensor) -> torch.Tensor: ...

    def rpm_to_thrust(self, rpm: torch.Tensor) -> torch.Tensor: ...


class T200ThrusterModel:
    """Blue Robotics T200 curve used by the stock BlueROV plant."""

    nominal_force_constant: float = 4.4e-7
    max_rpm: float = 3900.0
    deadband: float = 0.075

    def throttle_to_rpm(self, throttle: torch.Tensor) -> torch.Tensor:
        rpm = torch.where(
            throttle > self.deadband,
            3.6599e3 * throttle + 3.4521e2,
            torch.where(
                throttle < -self.deadband,
                3.4944e3 * throttle - 4.3350e2,
                torch.zeros_like(throttle),
            ),
        )
        return torch.clamp(rpm, -self.max_rpm, self.max_rpm)

    def rpm_to_thrust(self, rpm: torch.Tensor) -> torch.Tensor:
        return 9.81 * torch.where(
            rpm > 0,
            4.7368e-7 * torch.square(rpm) - 1.9275e-4 * rpm + 8.4452e-2,
            -3.8442e-7 * torch.square(rpm) - 1.6186e-4 * rpm - 3.9139e-2,
        )


@dataclass(frozen=True)
class PolynomialThrusterCfg:
    """Bidirectional affine throttle→RPM plus quadratic RPM→thrust."""

    min_rpm: float
    max_rpm: float
    throttle_deadband: float
    positive_rpm_slope: float
    positive_rpm_intercept: float
    negative_rpm_slope: float
    negative_rpm_intercept: float
    positive_thrust_coefficients: tuple[float, float, float]
    negative_thrust_coefficients: tuple[float, float, float]
    thrust_scale: float = 1.0
    nominal_force_constant: float = 1.0


class PolynomialThrusterModel:
    """Invertible calibration used by OT100 / derated ESC tables."""

    def __init__(self, cfg: PolynomialThrusterCfg):
        self.cfg = cfg
        self.nominal_force_constant = float(cfg.nominal_force_constant)

    def throttle_to_rpm(self, throttle: torch.Tensor) -> torch.Tensor:
        throttle = torch.clamp(throttle, -1.0, 1.0)
        deadband = self.cfg.throttle_deadband
        positive = (
            self.cfg.positive_rpm_slope * throttle + self.cfg.positive_rpm_intercept
        )
        negative = (
            self.cfg.negative_rpm_slope * throttle + self.cfg.negative_rpm_intercept
        )
        rpm = torch.where(
            throttle > deadband,
            positive,
            torch.where(
                throttle < -deadband,
                negative,
                torch.zeros_like(throttle),
            ),
        )
        return torch.clamp(rpm, self.cfg.min_rpm, self.cfg.max_rpm)

    def rpm_to_thrust(self, rpm: torch.Tensor) -> torch.Tensor:
        pos_q, pos_l, pos_c = self.cfg.positive_thrust_coefficients
        neg_q, neg_l, neg_c = self.cfg.negative_thrust_coefficients
        scale = self.cfg.thrust_scale
        positive = scale * (pos_q * rpm.square() + pos_l * rpm + pos_c)
        negative = scale * (neg_q * rpm.square() + neg_l * rpm + neg_c)
        return torch.where(
            rpm > 0.0,
            positive,
            torch.where(rpm < 0.0, negative, torch.zeros_like(rpm)),
        )


class AffineThrottleThrustModel:
    """Normalized throttle maps linearly to ``[−max_reverse, +max_forward]`` N."""

    def __init__(
        self,
        max_forward_n: float,
        max_reverse_n: float,
        rpm_scale: float = 1000.0,
    ):
        if max_forward_n <= 0.0 or max_reverse_n <= 0.0:
            raise ValueError("thrust limits must be positive magnitudes")
        self.max_forward_n = float(max_forward_n)
        self.max_reverse_n = float(max_reverse_n)
        self.rpm_scale = float(rpm_scale)
        self.nominal_force_constant = 1.0

    def throttle_to_rpm(self, throttle: torch.Tensor) -> torch.Tensor:
        return torch.clamp(throttle, -1.0, 1.0) * self.rpm_scale

    def rpm_to_thrust(self, rpm: torch.Tensor) -> torch.Tensor:
        throttle = rpm / self.rpm_scale
        return torch.where(
            throttle >= 0.0,
            throttle * self.max_forward_n,
            throttle * self.max_reverse_n,
        )


@dataclass(frozen=True)
class VirtualThrusterCfg:
    """Apply thrust as a base wrench when the USD/MJCF has no ``rotor_*`` bodies.

    Positions and directions are in the vehicle body frame (FLU). Directions
    need not be unit length; they are normalized at bind time.
    """

    positions_b: Sequence[Sequence[float]]
    directions_b: Sequence[Sequence[float]]
    names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if len(self.positions_b) != len(self.directions_b):
            raise ValueError("virtual thruster positions and directions must match")
        if len(self.positions_b) == 0:
            raise ValueError("virtual thruster layout must be non-empty")
        if self.names is not None and len(self.names) != len(self.positions_b):
            raise ValueError("virtual thruster names must match layout length")


T200_THRUSTER_MODEL = T200ThrusterModel()

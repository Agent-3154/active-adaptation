"""Batched trajectory curves as TensorClasses.

Time is physical ``t`` in ``[0, duration]``. Callers that prefer normalized
``τ ∈ [0, 1]`` should pass ``duration=1`` and scale velocities accordingly.

``MinimumJerk`` is the zero-accel (``a0 = aT = 0``) case and stores BCs for
closed-form Hermite evaluation. ``QuinticPolynomial`` stores ``τ``-space
coefficients and optionally accepts nonzero endpoint accelerations.
"""

from __future__ import annotations

import torch
from tensordict import TensorClass


def _broadcast_batch(*tensors: torch.Tensor) -> torch.Size:
    """Infer TensorClass batch size from leading dims of BC tensors (drop feature dim)."""
    shapes = [t.shape[:-1] for t in tensors]
    return torch.broadcast_shapes(*shapes)


class QuinticPolynomial(TensorClass):
    """Quintic in normalized time: ``x(τ) = Σ_{k=0}^{5} coeffs[..., k, :] * τ^k``, ``τ = t/T``.

    Fields
    ------
    coeffs: (..., 6, D)
        Polynomial coefficients ``[c0, ..., c5]`` in ``τ ∈ [0, 1]``.
    duration: (..., 1)
        Trajectory duration ``T`` (maps physical ``t`` ↔ ``τ``).
    """

    coeffs: torch.Tensor
    duration: torch.Tensor

    def eval(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate position and velocity at times ``t``.

        Args:
            t: (..., N) physical times in ``[0, T]``. Broadcasts against batch.

        Returns:
            x: (..., N, D) position
            v: (..., N, D) velocity ``dx/dt``
        """
        # Evaluate in normalized τ = t/T for conditioning; coeffs are in τ.
        tau = (t / self.duration.clamp_min(1e-8)).unsqueeze(-1)  # (..., N, 1)
        tau_s = tau.squeeze(-1)
        powers = torch.stack([tau_s.pow(k) for k in range(6)], dim=-1)  # (..., N, 6)
        dp_dtau = torch.stack(
            [
                torch.zeros_like(tau_s),
                torch.ones_like(tau_s),
                2 * tau_s,
                3 * tau_s.pow(2),
                4 * tau_s.pow(3),
                5 * tau_s.pow(4),
            ],
            dim=-1,
        )
        x = torch.einsum("...nk,...kd->...nd", powers, self.coeffs)
        dx_dtau = torch.einsum("...nk,...kd->...nd", dp_dtau, self.coeffs)
        v = dx_dtau / self.duration.clamp_min(1e-8).unsqueeze(-2)
        return x, v

    @classmethod
    def create(
        cls,
        x0: torch.Tensor,
        v0: torch.Tensor,
        xT: torch.Tensor,
        vT: torch.Tensor,
        duration: torch.Tensor | float = 1.0,
        a0: torch.Tensor | None = None,
        aT: torch.Tensor | None = None,
    ) -> QuinticPolynomial:
        """Fit a quintic to endpoint position / velocity (/ acceleration).

        Coefficients are stored in normalized time ``τ = t/T ∈ [0, 1]``.
        Endpoint velocities / accelerations are physical (``d/dt``, ``d²/dt²``).

        Args:
            x0, xT: (..., D) start / end position
            v0, vT: (..., D) start / end velocity
            duration: (..., 1) or scalar duration ``T > 0``
            a0, aT: optional (..., D) endpoint accelerations; default ``0``
        """
        x0 = torch.atleast_2d(x0)
        v0 = torch.atleast_2d(v0)
        xT = torch.atleast_2d(xT)
        vT = torch.atleast_2d(vT)
        if a0 is None:
            a0 = torch.zeros_like(x0)
        else:
            a0 = torch.atleast_2d(a0)
        if aT is None:
            aT = torch.zeros_like(xT)
        else:
            aT = torch.atleast_2d(aT)
        if not torch.is_tensor(duration):
            duration = torch.full((*x0.shape[:-1], 1), float(duration), device=x0.device, dtype=x0.dtype)
        else:
            duration = duration.reshape(*duration.shape[:-1], 1).to(device=x0.device, dtype=x0.dtype)

        T = duration
        # Fit x(τ) on [0, 1]; chain rule: x_τ = v * T, x_ττ = a * T²
        v0_n, vT_n = v0 * T, vT * T
        a0_n, aT_n = a0 * (T * T), aT * (T * T)

        c0 = x0
        c1 = v0_n
        c2 = a0_n * 0.5

        # Residuals at τ = 1 after subtracting start terms.
        bx = xT - c0 - c1 - c2
        bv = vT_n - c1 - 2.0 * c2
        ba = aT_n - 2.0 * c2

        # Inverse of the τ=1 Vandermonde block for (c3, c4, c5).
        c3 = 10.0 * bx - 4.0 * bv + 0.5 * ba
        c4 = -15.0 * bx + 7.0 * bv - ba
        c5 = 6.0 * bx - 3.0 * bv + 0.5 * ba

        coeffs = torch.stack([c0, c1, c2, c3, c4, c5], dim=-2)  # (..., 6, D)
        batch_size = _broadcast_batch(x0, v0, xT, vT, duration)
        return cls(coeffs=coeffs, duration=duration, batch_size=batch_size)


class MinimumJerk(TensorClass):
    """Minimum-jerk trajectory with endpoint position and velocity BCs.

    Equivalent to a quintic with ``a0 = aT = 0``, stored as boundaries and
    evaluated with the normalized-time basis (Flash & Hogan).

    Fields
    ------
    x0, v0, xT, vT: (..., D)
    duration: (..., 1)
    """

    x0: torch.Tensor
    v0: torch.Tensor
    xT: torch.Tensor
    vT: torch.Tensor
    duration: torch.Tensor

    def eval(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate position and velocity at times ``t``.

        Args:
            t: (..., N) physical times in ``[0, T]``.

        Returns:
            x: (..., N, D)
            v: (..., N, D)
        """
        T = self.duration  # (..., 1)
        tau = (t / T.clamp_min(1e-8)).unsqueeze(-1)  # (..., N, 1)

        # Hermite / min-jerk basis on [0, 1] with free endpoint velocities:
        #   x(τ) = x0 * α0(τ) + xT * α1(τ) + (v0 T) * β0(τ) + (vT T) * β1(τ)
        # where (α0, α1, β0, β1) are the quintic Hermite basis with zero accel.
        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau

        alpha0 = 1.0 - 10.0 * tau3 + 15.0 * tau4 - 6.0 * tau5
        alpha1 = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
        beta0 = tau - 6.0 * tau3 + 8.0 * tau4 - 3.0 * tau5
        beta1 = -4.0 * tau3 + 7.0 * tau4 - 3.0 * tau5

        # dα/dτ, dβ/dτ for chain rule v = (dx/dτ) / T
        dalpha0 = -30.0 * tau2 + 60.0 * tau3 - 30.0 * tau4
        dalpha1 = 30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4
        dbeta0 = 1.0 - 18.0 * tau2 + 32.0 * tau3 - 15.0 * tau4
        dbeta1 = -12.0 * tau2 + 28.0 * tau3 - 15.0 * tau4

        x0 = self.x0.unsqueeze(-2)  # (..., 1, D)
        v0 = self.v0.unsqueeze(-2)
        xT = self.xT.unsqueeze(-2)
        vT = self.vT.unsqueeze(-2)
        T_b = T.unsqueeze(-2)  # (..., 1, 1)

        x = alpha0 * x0 + alpha1 * xT + beta0 * (v0 * T_b) + beta1 * (vT * T_b)
        dx_dtau = dalpha0 * x0 + dalpha1 * xT + dbeta0 * (v0 * T_b) + dbeta1 * (vT * T_b)
        v = dx_dtau / T_b.clamp_min(1e-8)
        return x, v

    def as_quintic(self) -> QuinticPolynomial:
        """Convert to coefficient form (``a0 = aT = 0``)."""
        return QuinticPolynomial.create(
            self.x0, self.v0, self.xT, self.vT, duration=self.duration
        )

    @classmethod
    def create(
        cls,
        x0: torch.Tensor,
        v0: torch.Tensor,
        xT: torch.Tensor,
        vT: torch.Tensor,
        duration: torch.Tensor | float = 1.0,
    ) -> MinimumJerk:
        """Build from endpoint position / velocity (zero endpoint acceleration).

        Args:
            x0, xT: (..., D)
            v0, vT: (..., D)
            duration: (..., 1) or scalar ``T > 0``
        """
        x0 = torch.atleast_2d(x0)
        v0 = torch.atleast_2d(v0)
        xT = torch.atleast_2d(xT)
        vT = torch.atleast_2d(vT)
        if not torch.is_tensor(duration):
            duration = torch.full((*x0.shape[:-1], 1), float(duration), device=x0.device, dtype=x0.dtype)
        else:
            duration = duration.reshape(*duration.shape[:-1], 1).to(device=x0.device, dtype=x0.dtype)

        batch_size = _broadcast_batch(x0, v0, xT, vT, duration)
        return cls(
            x0=x0,
            v0=v0,
            xT=xT,
            vT=vT,
            duration=duration,
            batch_size=batch_size,
        )

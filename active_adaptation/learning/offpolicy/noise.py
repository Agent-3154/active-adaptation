"""Exploration noise processes as ``nn.Module``s with state in buffers.

``OUNoise``
    Discrete AR(1) / OU-style correlated noise matching the SAC rollout process
    ``eps_t = rho * eps_{t-1} + sqrt(1 - rho^2) * N(0, I)`` (unit variance).
    Per-env correlation ``rho`` is resampled on episode ``is_init``.

``ColoredNoise`` / ``PinkNoise``
    Torch port of `pink-noise-rl`_ (Timmer & König power-law PSD). A finite
    buffer of length ``seq_len`` is generated once at init and streamed in a
    circle; ``is_init`` only jumps the read index (no per-step FFT).

``ApproxPinkNoise``
    Online ``1/f`` approximation: sum of ``k`` AR(1) processes with
    geometrically spaced time constants (no FFT buffer).

``forward(x, is_init)`` returns noise shaped like ``x`` (``[N, A]``).
``is_init`` is ``bool[N, 1]``.

Per-env updates use ``torch.where`` (elementwise select). ``torch.cond`` is for
scalar whole-graph branches under ``torch.compile`` / export — the wrong tool
when only a subset of envs reset.

.. _pink-noise-rl: https://github.com/martius-lab/pink-noise-rl
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def powerlaw_psd_gaussian(
    exponent: float,
    size: int | Sequence[int],
    *,
    fmin: float = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Gaussian ``(1/f)**beta`` noise (Timmer & König), last dim = time.

    Torch port of ``pink.colorednoise.powerlaw_psd_gaussian``. Normalised to
    unit variance along the time axis.

    Args:
        exponent: Power-spectrum exponent ``beta`` in ``S(f) ∝ 1/f**beta``.
            Pink / flicker noise uses ``1``; brown uses ``2``.
        size: Output shape. The last dimension is time (FFT length); all other
            dimensions are independent series (e.g. ``(num_envs, action_dim, T)``).
        fmin: Low-frequency cutoff relative to unit sampling rate in ``[0, 0.5]``.
            Internally clamped to at least ``1/T``. Spectrum is flat below ``fmin``.
        device: Device for the returned tensor (default CPU).
        dtype: Floating dtype for the returned tensor.
    """
    try:
        size_list = list(size)
    except TypeError:
        size_list = [int(size)]

    samples = int(size_list[-1])
    if device is None:
        device = torch.device("cpu")

    f = torch.fft.rfftfreq(samples, device=device, dtype=dtype)
    fmin = max(float(fmin), 1.0 / samples)

    s_scale = f.clone()
    ix = int((s_scale < fmin).sum().item())
    if ix and ix < s_scale.numel():
        s_scale[:ix] = s_scale[ix]
    s_scale = torch.where(
        s_scale > 0,
        s_scale.pow(-exponent / 2.0),
        torch.zeros_like(s_scale),
    )

    w = s_scale[1:].clone()
    w[-1] *= (1 + (samples % 2)) / 2.0
    sigma = 2.0 * torch.sqrt(torch.sum(w.square())) / samples

    n_freq = f.numel()
    gen_size = size_list[:-1] + [n_freq]
    scale_view = s_scale.reshape(*([1] * (len(gen_size) - 1)), n_freq)

    sr = torch.randn(*gen_size, device=device, dtype=dtype) * scale_view
    si = torch.randn(*gen_size, device=device, dtype=dtype) * scale_view

    if samples % 2 == 0:
        si[..., -1] = 0
        sr[..., -1] *= math.sqrt(2.0)
    si[..., 0] = 0
    sr[..., 0] *= math.sqrt(2.0)

    s = torch.complex(sr, si)
    return torch.fft.irfft(s, n=samples, dim=-1) / sigma


class OUNoise(nn.Module):
    """AR(1) correlated exploration noise with state in module buffers.

    Unit-variance AR(1)::

        eps_t = rho * eps_{t-1} + sqrt(1 - rho^2) * xi_t,  xi ~ N(0, I)

    On ``is_init``, ``rho`` is redrawn in ``[rho_min, rho_max]`` and noise is zeroed.

    Args:
        num_envs: Parallel environment count ``N`` (leading batch size).
        action_dim: Action dimension ``A``.
        rho_min: Lower bound for per-env AR(1) correlation resampled on reset.
            ``0`` is white; closer to ``1`` is more temporally correlated.
        rho_max: Upper bound for per-env AR(1) correlation on reset.
        sigma: Multiplicative scale on the returned noise (after unit-variance AR(1)).
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        *,
        rho_min: float = 0.0,
        rho_max: float = 0.9,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.sigma = float(sigma)

        self.register_buffer("noise", torch.zeros(self.num_envs, self.action_dim))
        self.register_buffer("rho", torch.zeros(self.num_envs, 1))
        self.reset()

    @torch.no_grad()
    def reset(self, is_init: torch.Tensor | None = None) -> None:
        """Resample ``rho`` / zero noise. ``is_init`` is ``[N, 1]`` bool, or all envs."""
        if is_init is None:
            is_init = torch.ones(
                self.num_envs, 1, dtype=torch.bool, device=self.noise.device
            )
        new_rho = torch.empty_like(self.rho).uniform_(self.rho_min, self.rho_max)
        self.rho.copy_(torch.where(is_init, new_rho, self.rho))
        self.noise.copy_(torch.where(is_init, torch.zeros_like(self.noise), self.noise))

    @torch.no_grad()
    def forward(self, is_init: torch.Tensor) -> torch.Tensor:
        # is_init: [N, 1] bool — broadcasts over action dim for noise reset.
        self.reset(is_init)
        white = torch.randn_like(self.noise)
        eps = self.rho * self.noise + (1.0 - self.rho.square()).clamp_min(0.0).sqrt() * white
        self.noise.copy_(eps)
        return eps * self.sigma


class ColoredNoise(nn.Module):
    """Colored-noise process from a single precomputed FFT buffer (pink-noise-rl).

    Buffer shape ``[num_envs, action_dim, seq_len]`` is generated once in
    ``__init__`` and reused forever (circular read). ``is_init`` only randomizes
    the per-env read phase — no FFT on the step path.

    Args:
        num_envs: Parallel environment count ``N``.
        action_dim: Action dimension ``A``.
        seq_len: FFT buffer length ``T`` (correlation window). Not the task
            horizon; the buffer is streamed in a circle for infinite-horizon use.
            Larger ``T`` → longer memory and lower representable frequencies.
        beta: Power-spectrum exponent ``S(f) ∝ 1/f**beta`` (``1`` = pink).
        fmin: Low-frequency cutoff for the FFT synthesis (ignored if
            ``max_period`` is set).
        max_period: If given, sets ``fmin = 1 / max_period`` (longest correlation
            length in steps). Overrides ``fmin``.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        seq_len: int,
        *,
        beta: float = 1.0,
        fmin: float = 0.0,
        max_period: float | None = None,
    ):
        super().__init__()
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.seq_len = int(seq_len)
        self.beta = float(beta)
        self.fmin = 1.0 / float(max_period) if max_period is not None else float(fmin)

        buffer = powerlaw_psd_gaussian(
            self.beta,
            (self.num_envs, self.action_dim, self.seq_len),
            fmin=self.fmin,
        )
        self.register_buffer("buffer", buffer)
        self.register_buffer("idx", torch.zeros(self.num_envs, dtype=torch.long))

    @torch.no_grad()
    def reset(self, is_init: torch.Tensor | None = None) -> None:
        """Randomize read phase. ``is_init`` is ``[N, 1]`` bool, or all envs."""
        if is_init is None:
            is_init = torch.ones(
                self.num_envs, 1, dtype=torch.bool, device=self.buffer.device
            )
        new_idx = torch.randint(
            0, self.seq_len, (self.num_envs,), device=self.idx.device, dtype=self.idx.dtype
        )
        self.idx.copy_(torch.where(is_init.reshape(-1), new_idx, self.idx))

    @torch.no_grad()
    def forward(self, is_init: torch.Tensor) -> torch.Tensor:
        # is_init: [N, 1] bool — cheap phase jump, buffer unchanged.
        self.reset(is_init)
        env_ix = torch.arange(self.num_envs, device=self.buffer.device)
        noise = self.buffer[env_ix, :, self.idx]
        self.idx.copy_((self.idx + 1) % self.seq_len)
        return noise


class PinkNoise(ColoredNoise):
    """Colored noise with ``beta=1`` (pink / flicker noise).

    Args:
        num_envs: Parallel environment count ``N``.
        action_dim: Action dimension ``A``.
        seq_len: FFT buffer length ``T`` (see :class:`ColoredNoise`).
        fmin: Low-frequency cutoff (ignored if ``max_period`` is set).
        max_period: Optional longest correlation length; sets ``fmin = 1/max_period``.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        seq_len: int,
        *,
        fmin: float = 0.0,
        max_period: float | None = None,
    ):
        super().__init__(
            num_envs,
            action_dim,
            seq_len,
            beta=1.0,
            fmin=fmin,
            max_period=max_period,
        )


class ApproxPinkNoise(nn.Module):
    """Online pink-noise approximation via a sum of AR(1) / OU processes.

    Superposition of ``k`` independent unit-variance AR(1) processes with
    geometrically spaced time constants ``tau`` (hence ``rho = exp(-1/tau)``)
    approximates a ``1/f`` spectrum over ``[1/tau_max, 1/tau_min]`` without an
    FFT buffer (Bernamont / Erland–Greenwood style).

    On ``is_init``, latent states are zeroed. Output is scaled to unit variance
    (``/sqrt(k)``) then by ``sigma``.

    Args:
        num_envs: Parallel environment count ``N``.
        action_dim: Action dimension ``A``.
        k: Number of OU / AR(1) components. More components → flatter ``1/f``
            over a wider band (typical ``5``–``16``).
        tau_min: Shortest correlation time (steps). Sets the high-frequency end
            of the approximate pink band (``rho = exp(-1/tau)``).
        tau_max: Longest correlation time (steps). Sets the low-frequency end.
            Prefer ``tau_max >> tau_min`` (log-spaced).
        sigma: Multiplicative scale on the returned noise after ``1/sqrt(k)``
            variance normalization.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        k: int = 8,
        *,
        tau_min: float = 1.0,
        tau_max: float = 256.0,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.k = int(k)
        self.sigma = float(sigma)

        # Geometrically spaced correlation times → AR(1) coefficients.
        if self.k == 1:
            taus = torch.tensor([(tau_min + tau_max) * 0.5])
        else:
            taus = torch.logspace(
                math.log10(tau_min), math.log10(tau_max), self.k
            )
        rho = torch.exp(-1.0 / taus).view(1, self.k, 1)  # [1, K, 1]
        self.register_buffer("rho", rho)
        # Per-env, per-component state: [N, K, A]
        self.register_buffer(
            "noise",
            torch.zeros(self.num_envs, self.k, self.action_dim),
        )

    @torch.no_grad()
    def reset(self, is_init: torch.Tensor | None = None) -> None:
        """Zero OU states. ``is_init`` is ``[N, 1]`` bool, or all envs."""
        if is_init is None:
            is_init = torch.ones(
                self.num_envs, 1, dtype=torch.bool, device=self.noise.device
            )
        # Broadcast [N, 1] → [N, K, A]
        mask = is_init.unsqueeze(-1)
        self.noise.copy_(torch.where(mask, torch.zeros_like(self.noise), self.noise))

    @torch.no_grad()
    def forward(self, is_init: torch.Tensor) -> torch.Tensor:
        self.reset(is_init)
        white = torch.randn_like(self.noise)
        eps = self.rho * self.noise + (1.0 - self.rho.square()).clamp_min(0.0).sqrt() * white
        self.noise.copy_(eps)
        # Equal-weight sum; /sqrt(k) keeps unit variance if components are independent.
        return eps.sum(dim=1) * (self.sigma / math.sqrt(self.k))


if __name__ == "__main__":
    """Compare white / OU / approx-pink / FFT-pink in time and frequency.

    Run::

        python -m active_adaptation.learning.offpolicy.noise

    Writes ``noise_comparison.svg`` (and ``.png`` if matplotlib is installed).
    """
    from pathlib import Path

    torch.manual_seed(0)
    T = 4096
    seq_len = 2048
    show = 400
    reset_at = 200

    processes: dict[str, nn.Module | None] = {
        "white": None,
        "OU (ρ=0.9)": OUNoise(1, 1, rho_min=0.9, rho_max=0.9),
        "approx pink (k=8)": ApproxPinkNoise(1, 1, k=8, tau_min=1.0, tau_max=256.0),
        "FFT pink": PinkNoise(1, 1, seq_len=seq_len),
    }

    traces: dict[str, torch.Tensor] = {}
    for name, proc in processes.items():
        if proc is None:
            traces[name] = torch.randn(T)
            continue
        proc.reset()
        xs = [
            proc(torch.tensor([[t == reset_at]])).reshape(())
            for t in range(T)
        ]
        traces[name] = torch.stack(xs)

    def _psd(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x - x.mean()
        window = torch.hann_window(x.numel(), periodic=False)
        spec = torch.fft.rfft(x * window).abs().square().clamp_min(1e-18)
        freq = torch.fft.rfftfreq(x.numel())
        return freq[1:], spec[1:]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    def _svg() -> str:
        """Minimal dependency-free figure for time traces + log-log PSD."""
        W, H, pad = 960, 640, 56
        mid = H // 2
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
            "<style>text{font-family:sans-serif;font-size:12px;fill:#222}"
            ".title{font-size:14px;font-weight:600}.muted{fill:#666;font-size:11px}</style>",
            '<rect width="100%" height="100%" fill="white"/>',
            '<text class="title" x="16" y="24">Exploration noise: time series &amp; PSD</text>',
            '<text class="muted" x="16" y="42">'
            "Dashed line = is_init reset. Pink ≈ 1/f; white ≈ flat; OU = single Lorentzian.</text>",
        ]

        # ---- time panel ----
        x0, y0, pw, ph = pad, 58, W - 2 * pad, mid - 80
        parts.append(
            f'<text class="title" x="{x0}" y="{y0 - 8}">Time series (first {show} steps, offset)</text>'
        )
        parts.append(
            f'<line x1="{x0 + reset_at / show * pw}" y1="{y0}" '
            f'x2="{x0 + reset_at / show * pw}" y2="{y0 + ph}" '
            'stroke="#888" stroke-dasharray="4 3" stroke-width="1"/>'
        )
        for i, (name, col) in enumerate(zip(traces, colors)):
            x = traces[name][:show]
            # normalize each trace to [-1,1] then offset
            x = x / x.std().clamp_min(1e-6)
            offset = 3 - i
            ys = offset + 0.35 * x
            # map to panel
            y_min, y_max = -0.5, 4.0
            pts = []
            for ti in range(show):
                px = x0 + ti / (show - 1) * pw
                py = y0 + ph - (ys[ti].item() - y_min) / (y_max - y_min) * ph
                pts.append(f"{px:.1f},{py:.1f}")
            parts.append(
                f'<polyline fill="none" stroke="{col}" stroke-width="1.2" points="{" ".join(pts)}"/>'
            )
            parts.append(
                f'<text x="{x0 + pw + 6}" y="{y0 + 14 + i * 16}" fill="{col}">{name}</text>'
            )

        # ---- PSD panel ----
        x0, y0, pw, ph = pad, mid + 24, W - 2 * pad, H - mid - pad - 10
        parts.append(f'<text class="title" x="{x0}" y="{y0 - 8}">Power spectral density (log–log)</text>')

        f_lo, f_hi = 1.0 / T, 0.5
        # Collect specs for shared y-range
        specs = {n: _psd(traces[n]) for n in traces}
        s_all = torch.cat([s for _, s in specs.values()])
        s_lo, s_hi = s_all.quantile(0.05).item(), s_all.quantile(0.99).item()

        def _map(f: float, s: float) -> tuple[float, float]:
            u = (math.log10(f) - math.log10(f_lo)) / (math.log10(f_hi) - math.log10(f_lo))
            v = (math.log10(s) - math.log10(s_lo)) / (math.log10(s_hi) - math.log10(s_lo) + 1e-12)
            u = min(1.0, max(0.0, u))
            v = min(1.0, max(0.0, v))
            return x0 + u * pw, y0 + ph - v * ph

        # Reference slopes
        for exp, dash in ((0.0, "2 2"), (-1.0, "6 3")):
            f_ref = torch.logspace(math.log10(f_lo * 2), math.log10(0.2), 40)
            s_ref = (s_lo * s_hi) ** 0.5 * (f_ref / f_ref[0]).pow(exp)
            pts = []
            for f, s in zip(f_ref, s_ref):
                px, py = _map(float(f), float(s))
                pts.append(f"{px:.1f},{py:.1f}")
            parts.append(
                f'<polyline fill="none" stroke="#444" stroke-width="1" '
                f'stroke-dasharray="{dash}" points="{" ".join(pts)}"/>'
            )

        for name, col in zip(traces, colors):
            f, s = specs[name]
            # downsample for SVG size
            step = max(1, f.numel() // 400)
            f, s = f[::step], s[::step]
            pts = " ".join(
                f"{_map(float(fi), float(si))[0]:.1f},{_map(float(fi), float(si))[1]:.1f}"
                for fi, si in zip(f, s)
            )
            parts.append(
                f'<polyline fill="none" stroke="{col}" stroke-width="1.3" opacity="0.9" points="{pts}"/>'
            )

        parts.append(f'<text class="muted" x="{x0}" y="{y0 + ph + 16}">frequency →</text>')
        parts.append(f'<text class="muted" x="{x0 - 8}" y="{y0 + 12}" transform="rotate(-90 {x0 - 8},{y0 + 12})">power →</text>')
        parts.append("</svg>")
        return "\n".join(parts)

    out_svg = Path("noise_comparison.svg")
    out_svg.write_text(_svg(), encoding="utf-8")
    print(f"wrote {out_svg.resolve()}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; SVG only. pip install matplotlib for PNG.")
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        t = torch.arange(show)
        ax = axes[0]
        for i, (name, x) in enumerate(traces.items()):
            ax.plot(t, x[:show] / x[:show].std().clamp_min(1e-6) + 3.5 * (len(traces) - 1 - i),
                    lw=0.9, label=name, color=colors[i])
        ax.axvline(reset_at, color="0.5", ls="--", lw=0.8, label="is_init")
        ax.set_title("Time series (normalized, vertically offset)")
        ax.set_xlabel("step")
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8)

        ax = axes[1]
        for i, (name, x) in enumerate(traces.items()):
            f, s = _psd(x)
            ax.loglog(f.numpy(), s.numpy(), lw=1.0, alpha=0.85, label=name, color=colors[i])
        f_ref = torch.logspace(-3, -0.5, 50)
        ax.loglog(f_ref, 3e-2 * f_ref.pow(-1.0), "k--", lw=1.0, label=r"$\propto 1/f$")
        ax.loglog(f_ref, 3e-3 * torch.ones_like(f_ref), "k:", lw=1.0, label=r"$\propto f^0$")
        ax.set_title("Power spectral density")
        ax.set_xlabel("frequency (cycles / step)")
        ax.set_ylabel("power")
        ax.legend(loc="lower left", fontsize=8)
        ax.set_xlim(1.0 / T, 0.5)
        fig.suptitle("Exploration noise processes", fontsize=12)
        out_png = Path("noise_comparison.png")
        fig.savefig(out_png, dpi=150)
        print(f"wrote {out_png.resolve()}")
        plt.show()
    
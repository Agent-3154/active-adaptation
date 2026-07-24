"""Underwater / OceanSim observation terms.

Reference: ``reference/OceanSim/isaacsim/oceansim/sensors/``.

Implemented: ``baro_pressure``, ``dvl_linvel``, ``dvl_beam_range``, ``uw_camera``.
Stubs: ``imaging_sonar``.
"""

from __future__ import annotations

import math
import torch

from typing import TYPE_CHECKING, Optional, Sequence, Tuple
from typing_extensions import override

import active_adaptation
from active_adaptation.utils.math import quat_from_euler_xyz, quat_mul, quat_rotate, quat_rotate_inverse
from active_adaptation.utils.symmetry import SymmetryTransform

from .base import ObservationV2


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import TiledCamera
    from active_adaptation.envs.env_base import _EnvBase

if active_adaptation.get_backend() == "isaac":
    from isaaclab.utils.warp import raycast_mesh


# ---------------------------------------------------------------------------
# Janus DVL helpers (OceanSim DVLsensor geometry)
# ---------------------------------------------------------------------------


def _janus_beam_dirs_local(elevation_deg: float, rotation_deg: float, device: torch.device) -> torch.Tensor:
    """Four beam directions in the DVL frame (unit vectors).

    Matches OceanSim ``DVLsensor.attachDVL``: local beam axis is ``-Z``, then
    each beam is rotated by Euler XYZ (degrees) Janus layout::

        [[+elev, 0, rot], [0, +elev, rot], [-elev, 0, rot], [0, -elev, rot]]
    """
    elev = math.radians(elevation_deg)
    rot = math.radians(rotation_deg)
    eulers = torch.tensor(
        [
            [elev, 0.0, rot],
            [0.0, elev, rot],
            [-elev, 0.0, rot],
            [0.0, -elev, rot],
        ],
        device=device,
        dtype=torch.float32,
    )
    quats = quat_from_euler_xyz(eulers)
    local_fwd = torch.zeros(4, 3, device=device)
    local_fwd[:, 2] = -1.0
    return quat_rotate(quats, local_fwd)


def _janus_vel_transform(elevation_deg: float, device: torch.device) -> torch.Tensor:
    """OceanSim ``DVLsensor._transform``: (3, 4) map from beam noise → body xyz."""
    sin_e = math.sin(math.radians(elevation_deg))
    cos_e = math.cos(math.radians(elevation_deg))
    return torch.tensor(
        [
            [1.0 / (2.0 * sin_e), 0.0, -1.0 / (2.0 * sin_e), 0.0],
            [0.0, 1.0 / (2.0 * sin_e), 0.0, -1.0 / (2.0 * sin_e)],
            [1.0 / (4.0 * cos_e), 1.0 / (4.0 * cos_e), 1.0 / (4.0 * cos_e), 1.0 / (4.0 * cos_e)],
        ],
        device=device,
        dtype=torch.float32,
    )


class _JanusDvlMixin:
    """Shared body bind + beam raycasts for DVL observation terms."""

    body_name: str
    elevation: float
    rotation: float
    min_range: float
    max_range: float
    num_beams_out_range_threshold: int
    freq: Optional[float]
    translation: Optional[Sequence[float]]
    orientation: Optional[Sequence[float]]

    def _bind_dvl(self, env: "_EnvBase") -> None:
        self.asset: Articulation = env.scene.articulations["robot"]
        body_ids, body_names = self.asset.find_bodies(self.body_name)
        if not body_ids:
            raise ValueError(f"DVL body '{self.body_name}' not found on robot")
        self.body_id = int(body_ids[0])
        self.body_name_resolved = body_names[0]
        self.ground_mesh = env.ground_mesh

        self.beam_dirs_dvl = _janus_beam_dirs_local(self.elevation, self.rotation, self.device)
        self.vel_transform = _janus_vel_transform(self.elevation, self.device)

        if self.translation is None:
            self.mount_pos_b = torch.zeros(3, device=self.device)
        else:
            self.mount_pos_b = torch.as_tensor(self.translation, device=self.device, dtype=torch.float32)
        if self.orientation is None:
            self.mount_quat_b = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        else:
            self.mount_quat_b = torch.as_tensor(self.orientation, device=self.device, dtype=torch.float32)
            self.mount_quat_b = self.mount_quat_b / self.mount_quat_b.norm().clamp_min(1e-8)

        self._sensor_dt = None if self.freq is None else 1.0 / float(self.freq)
        self._elapsed = 0.0
        self._beam_range = torch.full((self.num_envs, 4), self.max_range, device=self.device)
        self._beam_hit = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.bool)

    def _dvl_pose_w(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """World pose of the DVL mount (body link ⊗ local mount xform)."""
        body_pos = self.asset.data.body_link_pos_w[:, self.body_id]
        body_quat = self.asset.data.body_link_quat_w[:, self.body_id]
        origin = body_pos + quat_rotate(body_quat, self.mount_pos_b.expand(self.num_envs, 3))
        # q_w = q_body ⊗ q_mount  (apply mount then body)
        w1, x1, y1, z1 = body_quat.unbind(-1)
        w2, x2, y2, z2 = self.mount_quat_b.expand(self.num_envs, 4).unbind(-1)
        quat = torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )
        return origin, quat

    def _cast_beams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(range (N,4), hit (N,4))`` against ``env.ground_mesh``."""
        origin_w, quat_w = self._dvl_pose_w()
        dirs_w = quat_rotate(
            quat_w.unsqueeze(1).expand(self.num_envs, 4, 4),
            self.beam_dirs_dvl.expand(self.num_envs, 4, 3),
        )
        starts = origin_w.unsqueeze(1).expand(self.num_envs, 4, 3)
        hits = raycast_mesh(
            ray_starts=starts.reshape(-1, 3),
            ray_directions=dirs_w.reshape(-1, 3),
            max_dist=self.max_range,
            mesh=self.ground_mesh,
            return_distance=False,
        )[0].reshape(self.num_envs, 4, 3)
        dist = (hits - starts).norm(dim=-1)
        finite = torch.isfinite(hits).all(dim=-1)
        hit = finite & (dist >= self.min_range) & (dist <= self.max_range)
        dist = torch.where(hit, dist, torch.full_like(dist, float("nan")))
        return dist, hit

    def _maybe_refresh_beams(self, force: bool = False) -> bool:
        """Refresh beam cache. Returns True if a new cast was performed.

        With ``freq`` set, casts on ``force`` or when accumulated ``step_dt``
        reaches the sensor period; otherwise casts every call.
        """
        if force or self._sensor_dt is None:
            self._beam_range, self._beam_hit = self._cast_beams()
            return True
        self._elapsed += float(self.env.step_dt)
        if self._elapsed + 1e-9 >= self._sensor_dt:
            self._elapsed = 0.0
            self._beam_range, self._beam_hit = self._cast_beams()
            return True
        return False

    def _dropout(self) -> torch.Tensor:
        """True where missed beams ≥ OceanSim dropout threshold. Shape (N,)."""
        misses = (~self._beam_hit).sum(dim=-1)
        return misses >= self.num_beams_out_range_threshold

    def _debug_draw_beams(self) -> None:
        """Draw Janus beams: cyan on hit (to measured range), red on miss (to ``max_range``)."""
        if not self.env.sim.has_gui():
            return
        origin_w, quat_w = self._dvl_pose_w()
        dirs_w = quat_rotate(
            quat_w.unsqueeze(1).expand(self.num_envs, 4, 4),
            self.beam_dirs_dvl.expand(self.num_envs, 4, 3),
        )
        starts = origin_w.unsqueeze(1).expand(self.num_envs, 4, 3)
        hit = self._beam_hit
        ranges = torch.where(
            hit,
            self._beam_range.nan_to_num(nan=self.max_range),
            torch.full_like(self._beam_range, self.max_range),
        )
        vecs = dirs_w * ranges.unsqueeze(-1)
        starts_f = starts.reshape(-1, 3)
        vecs_f = vecs.reshape(-1, 3)
        hit_f = hit.reshape(-1)
        if hit_f.any():
            self.env.scene.draw_vector(
                starts_f[hit_f],
                vecs_f[hit_f],
                color=(0.1, 0.85, 0.95, 1.0),
                size=2.0,
            )
        if (~hit_f).any():
            self.env.scene.draw_vector(
                starts_f[~hit_f],
                vecs_f[~hit_f],
                color=(0.95, 0.25, 0.2, 0.7),
                size=1.5,
            )


class baro_pressure(ObservationV2):
    """Hydrostatic + atmospheric pressure (OceanSim ``BarometerSensor``).

    Output shape: ``(num_envs, 1)`` pressure in Pascals.

    Args:
        body_name: Regex / name passed to ``Articulation.find_bodies``; the first
            match is the pressure sample point (link world-z).
        water_density: Fluid density ρ in kg/m³ (default fresh water 1000).
        g: Gravitational acceleration in m/s² used in ``ρ g depth``.
        g_from_sim: If True, replace ``g`` with the sim gravity magnitude when
            readable (Isaac). Constructor ``g`` is the fallback.
        noise_std: Std-dev of additive Gaussian pressure noise (Pa). OceanSim
            used variance ``noise_cov`` (``≈ noise_std**2``).
        water_surface_z: World-frame z of the free surface. Depth is
            ``max(0, water_surface_z - z_body)``.
        atmosphere_pressure: Surface atmospheric pressure in Pa (default 1 atm).

    Faithfulness notes
    ------------------
    - **Matched:** ``P = P_atm + ρ g max(0, z_surface - z)``; optional noise;
      sample on a named body link (OceanSim used a baro prim pose).
    - **Not matched:** no Omniverse ``BaseSensor`` prim — analytic only.
    """
    namespace: str = "underwater"

    def __init__(
        self,
        body_name: str,
        water_density: float = 1000.0,
        g: float = 9.81,
        g_from_sim: bool = False,
        noise_std: float = 0.0,
        water_surface_z: float = 0.0,
        atmosphere_pressure: float = 101325.0,
    ) -> None:
        self.body_name = body_name
        self.water_density = float(water_density)
        self.g = float(g)
        self.g_from_sim = bool(g_from_sim)
        self.noise_std = float(noise_std)
        self.water_surface_z = float(water_surface_z)
        self.atmosphere_pressure = float(atmosphere_pressure)

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        body_ids, body_names = self.asset.find_bodies(self.body_name)
        if not body_ids:
            raise ValueError(f"baro_pressure body '{self.body_name}' not found on robot")
        self.body_id = int(body_ids[0])
        self.body_name_resolved = body_names[0]

        if self.g_from_sim:
            g_sim = self._read_sim_gravity_mag()
            if g_sim is not None:
                self.g = g_sim

    def _read_sim_gravity_mag(self) -> Optional[float]:
        try:
            if self.env.backend == "isaac":
                g_vec = self.env.sim.get_physics_context().get_gravity()[1]
                return float(abs(g_vec))
        except Exception:
            return None
        return None

    @override
    def compute(self) -> torch.Tensor:
        z = self.asset.data.body_link_pos_w[:, self.body_id, 2]
        depth = (self.water_surface_z - z).clamp_min(0.0)
        pressure = self.atmosphere_pressure + self.water_density * self.g * depth
        if self.noise_std > 0.0:
            pressure = pressure + self.noise_std * torch.randn_like(pressure)
        return pressure.unsqueeze(-1)

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        return SymmetryTransform(perm=torch.arange(1), signs=[1.0])


class dvl_linvel(ObservationV2, _JanusDvlMixin):
    """Body-frame linear velocity with Janus beam gating (OceanSim ``DVLsensor``).

    Output shape: ``(num_envs, 3)`` body-frame linear velocity (m/s). Zeros on
    beam dropout.

    Args:
        body_name: Regex / name for ``find_bodies``; first match is the DVL
            parent link (velocity frame and default mount origin).
        elevation: Janus beam elevation from the horizontal, in degrees
            (OceanSim default 22.5°).
        rotation: Common yaw of the four beams about DVL +Z, in degrees
            (OceanSim default 45°).
        vel_cov: Variance of the 4-D beam-space Gaussian noise before the
            Janus ``_transform`` maps it into xyz (OceanSim ``vel_cov`` /
            ``init_cov``). ``0`` disables noise.
        min_range: Minimum valid beam range in meters (shorter = miss).
        max_range: Maximum raycast / valid range in meters.
        num_beams_out_range_threshold: Dropout if at least this many beams
            miss (OceanSim default 2).
        freq: Optional sensor rate in Hz. If set, hold the last velocity
            between updates (OceanSim fd API returned NaN). ``None`` = every
            policy step.
        translation: Optional mount offset in the parent body frame (meters),
            shape (3,). Affects beam origins only.
        orientation: Optional mount quaternion ``(w, x, y, z)`` in the parent
            body frame. Affects beam directions only (not the velocity frame).

    Faithfulness notes
    ------------------
    - **Matched (OceanSim quirk):** velocity is privileged GT body linvel +
      dropout gate + Janus-projected noise — not Doppler from beams.
    - **Ray backend:** ``ground_mesh`` raycasts, not PhysX LightBeam prims.
    - **Rate limit:** hold-last when ``freq`` is set (not OceanSim NaNs).
    """
    namespace: str = "underwater"
    supported_backends = ("isaac",)

    def __init__(
        self,
        body_name: str,
        elevation: float = 22.5,
        rotation: float = 45.0,
        vel_cov: float = 0.0,
        min_range: float = 0.1,
        max_range: float = 100.0,
        num_beams_out_range_threshold: int = 2,
        freq: Optional[float] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        self.body_name = body_name
        self.elevation = float(elevation)
        self.rotation = float(rotation)
        self.vel_cov = float(vel_cov)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.num_beams_out_range_threshold = int(num_beams_out_range_threshold)
        self.freq = freq
        self.translation = translation
        self.orientation = orientation

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self._bind_dvl(env)
        self._linvel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._maybe_refresh_beams(force=True)
        self._measure_vel()

    @override
    def update(self) -> None:
        # With freq set, hold last velocity between sensor periods (RL-friendly
        # stand-in for OceanSim fd NaNs).
        if self._maybe_refresh_beams():
            self._measure_vel()

    def _measure_vel(self) -> None:
        dropout = self._dropout()
        # OceanSim: parent RB world vel → body frame (not DVL mount frame).
        vel_w = self.asset.data.body_com_lin_vel_w[:, self.body_id]
        body_quat = self.asset.data.body_link_quat_w[:, self.body_id]
        vel_b = quat_rotate_inverse(body_quat, vel_w)
        if self.vel_cov > 0.0:
            # beam noise ~ N(0, vel_cov), then xyz += transform @ noise
            beam_noise = torch.randn(self.num_envs, 4, device=self.device) * math.sqrt(self.vel_cov)
            vel_b = vel_b + beam_noise @ self.vel_transform.T
        self._linvel_b = torch.where(dropout.unsqueeze(-1), torch.zeros_like(vel_b), vel_b)

    @override
    def compute(self) -> torch.Tensor:
        return self._linvel_b

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        # Mirror lateral velocity (body y).
        return SymmetryTransform(perm=torch.arange(3), signs=[1.0, -1.0, 1.0])

    @override
    def debug_draw(self) -> None:
        self._debug_draw_beams()


class dvl_beam_range(ObservationV2, _JanusDvlMixin):
    """Per-beam DVL ranges (OceanSim ``DVLsensor.get_depth``).

    Output shape: ``(num_envs, 4)`` ranges in meters (misses → ``nan_fill``).

    Args:
        body_name: Regex / name for ``find_bodies``; first match is the DVL
            parent link.
        elevation: Janus beam elevation from the horizontal, in degrees.
        rotation: Common beam yaw about DVL +Z, in degrees.
        depth_cov: Variance of additive per-beam range noise (OceanSim
            ``depth_cov`` / ``init_cov``). ``0`` disables noise.
        min_range: Minimum valid beam range in meters.
        max_range: Maximum raycast / valid range in meters.
        num_beams_out_range_threshold: Kept for API parity with ``dvl_linvel`` /
            OceanSim (used for logging dropout conceptually; ranges are still
            returned per-beam). Shared geometry should use the same value.
        freq: Optional sensor rate in Hz; hold last ranges between updates.
            ``None`` = every policy step.
        nan_fill: Finite fill value for missed beams (OceanSim used NaN).
        translation: Optional mount offset in the parent body frame (meters).
        orientation: Optional mount quaternion ``(w, x, y, z)`` in the parent
            body frame.

    Faithfulness notes
    ------------------
    - **Matched:** four Janus ranges + optional depth noise; misses filled
      instead of NaN for RL.
    - **Ray backend:** same ``ground_mesh`` raycast as ``dvl_linvel``.
    - Use the same ``elevation`` / ``rotation`` / mount kwargs as ``dvl_linvel``
      when both terms are enabled.
    """

    namespace: str = "underwater"
    supported_backends = ("isaac",)

    def __init__(
        self,
        body_name: str,
        elevation: float = 22.5,
        rotation: float = 45.0,
        depth_cov: float = 0.0,
        min_range: float = 0.1,
        max_range: float = 100.0,
        num_beams_out_range_threshold: int = 2,
        freq: Optional[float] = None,
        nan_fill: float = -1.0,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        self.body_name = body_name
        self.elevation = float(elevation)
        self.rotation = float(rotation)
        self.depth_cov = float(depth_cov)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.num_beams_out_range_threshold = int(num_beams_out_range_threshold)
        self.freq = freq
        self.nan_fill = float(nan_fill)
        self.translation = translation
        self.orientation = orientation

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self._bind_dvl(env)
        self._ranges = torch.full((self.num_envs, 4), self.nan_fill, device=self.device)
        self._maybe_refresh_beams(force=True)
        self._measure_ranges()

    @override
    def update(self) -> None:
        if self._maybe_refresh_beams():
            self._measure_ranges()

    def _measure_ranges(self) -> None:
        ranges = self._beam_range
        if self.depth_cov > 0.0:
            ranges = ranges + math.sqrt(self.depth_cov) * torch.randn_like(ranges)
        self._ranges = ranges.nan_to_num(nan=self.nan_fill)

    @override
    def compute(self) -> torch.Tensor:
        return self._ranges

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        # Janus order [0,1,2,3]; left/right swap depends on layout — leave identity
        # until a BlueROV symmetry convention is fixed.
        return SymmetryTransform(perm=torch.arange(4), signs=[1.0, 1.0, 1.0, 1.0])

    @override
    def debug_draw(self) -> None:
        self._debug_draw_beams()


class uw_camera(ObservationV2):
    """Underwater RGB camera (OceanSim ``UW_Camera`` + ``UW_render``).

    Spawns an Isaac Lab :class:`~isaaclab.sensors.TiledCamera` (via ``edit_spec``),
    reads RGB + depth each step, and applies the OceanSim underwater image
    formation model in torch::

        I_uw = I_raw * exp(-d · α) + β * 255 * (1 - exp(-d · γ))

    where ``α = atten_coeff``, ``β = backscatter_value``, ``γ = backscatter_coeff``,
    and ``d`` is per-pixel depth.

    Output shape: ``(num_envs, 3, H, W)`` float in ``[0, 1]`` if ``normalize``,
    else float in ``[0, 255]``. If ``flatten``, ``(num_envs, 3*H*W)``.

    Args:
        body_name: Robot body / link to attach the camera to
            (``{ENV}/Robot/{body_name}/{sensor_name}``).
        resolution: ``(width, height)`` in pixels.
        focal_length: Pinhole focal length in mm.
        focus_distance: Pinhole focus distance in m.
        horizontal_aperture: Pinhole horizontal aperture in mm.
        clipping_range: Near/far clipping planes in m (also bounds useful depth).
        backscatter_value: RGB water-column veiling light color in ``[0, 1]``
            (OceanSim ``backscatter_value`` / ``UW_param[0:3]``). Scaled by 255
            in the render equation.
        atten_coeff: Per-channel attenuation coefficients α (1/m) applied as
            ``exp(-depth * atten_coeff)`` on the direct RGB path.
        backscatter_coeff: Per-channel backscatter extinction γ (1/m) applied as
            ``exp(-depth * backscatter_coeff)`` on the veiling term.
        sensor_name: Scene sensor attribute name. Defaults to a unique
            ``uw_camera_{id}`` so multiple cameras can coexist.
        offset_pos: Camera translation w.r.t. the parent body frame (m).
        offset_rot: Camera rotation ``(w, x, y, z)`` w.r.t. the parent body.
        offset_convention: Isaac offset convention (``"ros"``, ``"world"``, or
            ``"opengl"``).
        normalize: If True, return RGB in ``[0, 1]``; else ``[0, 255]``.
        flatten: If True, flatten spatial/channel dims for MLP policies.
        debug_vis: If True, register a camera frustum via ``scene.create_camera_frustum``
            and push RGB each ``debug_draw`` (requires a Viser viewer).
        debug_vis_every: Push frustum image every N debug_draw calls.

    Faithfulness notes
    ------------------
    - **Matched:** OceanSim ``UW_render`` math (direct attenuation + veiling
      light). Defaults follow OceanSim's packed default array *as executed*
      (see note below).
    - **OceanSim packing quirk:** OceanSim's docstring ordered
      ``UW_param`` as ``[backscatter | atten | backscatter_coeff]``, but the
      non-YAML code path assigned ``atten_coeff = UW_param[6:9]`` and
      ``backscatter_coeff = UW_param[3:6]``. Our explicit defaults match that
      *runtime* packing: ``atten=(0.05,)*3``, ``backscatter_coeff=(0.05,0.05,0.2)``.
    - **Not matched:** Omniverse Replicator annotators / UI viewport / disk
      writer. We use Isaac Lab tiled RGB+depth and a batched torch port of the
      Warp kernel (same formula, vectorized over envs).
    - **Depth:** OceanSim used Replicator ``distance_to_camera``. We use tiled
      camera ``depth`` (``distance_to_image_plane`` alias). Absolute meters are
      comparable for the exponential model; exact ray definition may differ
      slightly from ``distance_to_camera``.
    """

    namespace: str = "underwater"
    supported_backends = ("isaac",)
    _instance_count = 0

    def __init__(
        self,
        body_name: str,
        resolution: Tuple[int, int] = (256, 192),
        focal_length: float = 24.0,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 20.955,
        clipping_range: Tuple[float, float] = (0.1, 20.0),
        backscatter_value: Tuple[float, float, float] = (0.0, 0.31, 0.24),
        atten_coeff: Tuple[float, float, float] = (0.05, 0.05, 0.05),
        backscatter_coeff: Tuple[float, float, float] = (0.05, 0.05, 0.2),
        sensor_name: Optional[str] = None,
        offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        offset_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        offset_convention: str = "world",
        normalize: bool = True,
        flatten: bool = False,
        debug_vis: bool = False,
        debug_vis_every: int = 1,
    ) -> None:
        self.body_name = body_name
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.focal_length = float(focal_length)
        self.focus_distance = float(focus_distance)
        self.horizontal_aperture = float(horizontal_aperture)
        self.clipping_range = (float(clipping_range[0]), float(clipping_range[1]))
        self.backscatter_value = tuple(float(x) for x in backscatter_value)
        self.atten_coeff = tuple(float(x) for x in atten_coeff)
        self.backscatter_coeff = tuple(float(x) for x in backscatter_coeff)
        self.offset_pos = tuple(float(x) for x in offset_pos)
        self.offset_rot = tuple(float(x) for x in offset_rot)
        self.offset_convention = offset_convention
        self.normalize = bool(normalize)
        self.flatten = bool(flatten)
        self.debug_vis = bool(debug_vis)
        self.debug_vis_every = max(int(debug_vis_every), 1)
        self._debug_vis_step = 0

        if sensor_name is None:
            uw_camera._instance_count += 1
            self.sensor_name = f"uw_camera_{uw_camera._instance_count}"
        else:
            self.sensor_name = sensor_name

    @override
    def edit_spec(self, scene_config: InteractiveSceneCfg) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCameraCfg

        if hasattr(scene_config, self.sensor_name):
            raise ValueError(
                f"Scene config already has sensor '{self.sensor_name}'. "
                "Choose a distinct sensor_name for each uw_camera instance."
            )

        prim_path = f"{{ENV_REGEX_NS}}/Robot/{self.body_name}/{self.sensor_name}"
        cfg = TiledCameraCfg(
            prim_path=prim_path,
            offset=TiledCameraCfg.OffsetCfg(
                pos=self.offset_pos,
                rot=self.offset_rot,
                convention=self.offset_convention,
            ),
            # rgb for appearance; depth drives the UW exponential (OceanSim used
            # Replicator distance_to_camera — see class faithfulness notes).
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=self.focal_length,
                focus_distance=self.focus_distance,
                horizontal_aperture=self.horizontal_aperture,
                clipping_range=self.clipping_range,
            ),
            width=self.resolution[0],
            height=self.resolution[1],
        )
        setattr(scene_config, self.sensor_name, cfg)

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self.env.render_enabled = True

        self.camera: TiledCamera = self.env.scene.sensors[self.sensor_name]
        self.asset: Articulation = self.env.scene.articulations["robot"]
        body_ids = self.asset.find_bodies(self.body_name)[0]
        if len(body_ids) != 1:
            raise ValueError(
                f"uw_camera: expected exactly one body matching '{self.body_name}', "
                f"found {len(body_ids)}."
            )
        self.body_id = int(body_ids[0])
        # Offset in parent/body frame; orientation converted to ROS (+Z fwd, +Y down)
        # so composed world quat matches Viser/OpenCV frustum convention.
        self._offset_pos_t = torch.tensor(
            self.offset_pos, device=self.device, dtype=torch.float32
        )
        offset_quat = torch.tensor(
            self.offset_rot, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        if self.offset_convention != "ros":
            from isaaclab.utils.math import convert_camera_frame_orientation_convention

            offset_quat = convert_camera_frame_orientation_convention(
                offset_quat,
                origin=self.offset_convention,  # type: ignore[arg-type]
                target="ros",
            )
        self._offset_quat_ros_t = offset_quat.squeeze(0)

        # (1, 1, 1, 3) for broadcast over NHWC
        self._backscatter_value_t = torch.tensor(
            self.backscatter_value, device=self.device, dtype=torch.float32
        ).view(1, 1, 1, 3)
        self._atten_coeff_t = torch.tensor(
            self.atten_coeff, device=self.device, dtype=torch.float32
        ).view(1, 1, 1, 3)
        self._backscatter_coeff_t = torch.tensor(
            self.backscatter_coeff, device=self.device, dtype=torch.float32
        ).view(1, 1, 1, 3)
        w, h = self.resolution
        self._image = torch.zeros(self.num_envs, 3, h, w, device=self.device)

        self.camera_handle = None
        if self.debug_vis:
            try:
                # Vertical FOV from pinhole horizontal aperture / focal length (mm).
                fov_x = 2.0 * math.atan(0.5 * self.horizontal_aperture / self.focal_length)
                aspect = w / max(h, 1)
                fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) / aspect)
                self.camera_handle = self.env.scene.create_camera_frustum(
                    self.sensor_name,
                    fov_y=fov_y,
                    aspect=aspect,
                )
            except Exception as e:
                print(f"Error creating camera frustum: {e}")
                self.camera_handle = None

    def _apply_uw_render(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Batched OceanSim ``UW_render`` (torch).

        Args:
            rgb: ``(N, H, W, 3|4)`` uint8 or float.
            depth: ``(N, H, W)`` or ``(N, H, W, 1)`` meters.
        Returns:
            ``(N, H, W, 3)`` float RGB in ``[0, 255]``.
        """
        raw = rgb[..., :3].float()
        if depth.ndim == 4:
            depth = depth.squeeze(-1)
        depth = depth.float().nan_to_num(
            nan=self.clipping_range[1],
            posinf=self.clipping_range[1],
            neginf=0.0,
        )
        depth = depth.clamp(self.clipping_range[0], self.clipping_range[1]).unsqueeze(-1)
        exp_atten = torch.exp(-depth * self._atten_coeff_t)
        exp_back = torch.exp(-depth * self._backscatter_coeff_t)
        uw = raw * exp_atten + self._backscatter_value_t * 255.0 * (1.0 - exp_back)
        return uw.clamp(0.0, 255.0)

    @override
    def compute(self) -> torch.Tensor:
        rgb = self.camera.data.output["rgb"]
        depth = self.camera.data.output["depth"]
        uw = self._apply_uw_render(rgb, depth)
        if self.normalize:
            uw = uw / 255.0
        # NHWC → NCHW
        img = uw.permute(0, 3, 1, 2).contiguous()
        self._image = img
        if self.flatten:
            return img.reshape(self.num_envs, -1)
        return img

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        # Horizontal flip of the image width axis (NCHW channel-major flatten
        # is not represented; only valid when flatten=False consumers use H,W).
        width = self.resolution[0]
        perm = torch.arange(width - 1, -1, -1, dtype=torch.long)
        return SymmetryTransform(perm, torch.ones(width))

    def _image_hwc_uint8(self, env_idx: int):
        img = self._image[env_idx].detach()
        if self.normalize:
            img = img * 255.0
        return (
            img.clamp(0, 255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

    @override
    def debug_draw(self) -> None:
        if not self.debug_vis or self.camera_handle is None:
            return
        self._debug_vis_step += 1
        if (self._debug_vis_step - 1) % self.debug_vis_every != 0:
            return
        env_idx = 0
        # ROS camera frame (+Z forward, +Y down) matches Viser/OpenCV frustum.
        body_pos = self.asset.data.body_link_pos_w[env_idx, self.body_id]
        body_quat = self.asset.data.body_link_quat_w[env_idx, self.body_id]
        pos = body_pos + quat_rotate(body_quat, self._offset_pos_t)
        quat = quat_mul(body_quat, self._offset_quat_ros_t)
        self.camera_handle.position = pos
        self.camera_handle.wxyz = quat
        self.camera_handle.image = self._image_hwc_uint8(env_idx)


class imaging_sonar(ObservationV2):
    """Imaging sonar (OceanSim ``ImagingSonarSensor``). Stub — not implemented."""

    supported_backends = ("isaac",)

    def __init__(
        self,
        body_name: str = "base_link",
        min_range: float = 0.2,
        max_range: float = 3.0,
        range_res: float = 0.008,
        hori_fov: float = 130.0,
        vert_fov: float = 20.0,
        angular_res: float = 0.5,
        hori_res: int = 3000,
        flatten: bool = True,
        include_unlabelled: bool = False,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        self.body_name = body_name
        self.min_range = min_range
        self.max_range = max_range
        self.range_res = range_res
        self.hori_fov = hori_fov
        self.vert_fov = vert_fov
        self.angular_res = angular_res
        self.hori_res = hori_res
        self.flatten = flatten
        self.include_unlabelled = include_unlabelled
        self.translation = translation
        self.orientation = orientation

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        raise NotImplementedError(
            "imaging_sonar: bind camera/sonar pipeline "
            "(see OceanSim ImagingSonarSensor)."
        )

    @override
    def compute(self) -> torch.Tensor:
        raise NotImplementedError("imaging_sonar.compute")


__all__ = [
    "baro_pressure",
    "dvl_linvel",
    "dvl_beam_range",
    "uw_camera",
    "imaging_sonar",
]

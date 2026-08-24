from __future__ import annotations

import colorsys
import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import einops
import torch
from jaxtyping import Float
from typing_extensions import override

from ..base import ObservationV2
from active_adaptation.utils.math import (
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    root_pose_from_view_z_up,
)
from active_adaptation.utils.symmetry import SymmetryTransform

if TYPE_CHECKING:
    import numpy as np
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import TiledCamera
    from isaaclab.scene import InteractiveSceneCfg
    from mjlab.sensor import CameraSensor
    from mjlab.scene import SceneCfg
    from active_adaptation.envs.env_base import _EnvBase


def _offset_rpy_deg_to_quat(
    offset_rpy: Tuple[float, float, float],
) -> Tuple[float, float, float, float]:
    """XYZ Euler degrees → WXYZ quaternion (CPU)."""
    rpy = torch.tensor(offset_rpy, dtype=torch.float32) * (math.pi / 180.0)
    return tuple(quat_from_euler_xyz(rpy.unsqueeze(0)).squeeze(0).tolist())


def raymap(width: int, height: int, fov: float) -> Float[torch.Tensor, "height width 3"]:
    """
    Generate a raymap for a given width, height, and field of view.

    The raymap represents normalized ray directions for a perspective camera model.
    Each pixel corresponds to a ray direction pointing from the camera center through
    that pixel. The rays are in camera space, where +X is forward, +Y is left, and +Z is up.

    Pixel layout: row ``h=0`` is the top of the image (rays tilt towards +Z) and
    column ``w=0`` is the left of the image (rays tilt towards +Y). Consequently a
    left-right mirror of the world corresponds to flipping the raymap along the
    width (last spatial) dim.

    Args:
        width: The width of the raymap in pixels.
        height: The height of the raymap in pixels.
        fov: The horizontal field of view in radians.

    Returns:
        A tensor of shape (height, width, 3) where the last dimension contains the
        normalized ray direction vector (x, y, z) for each pixel.
    """
    u = torch.arange(width, dtype=torch.float32)
    v = torch.arange(height, dtype=torch.float32)

    uu, vv = torch.meshgrid(u, v, indexing="xy")

    u_ndc = (uu + 0.5) / width * 2.0 - 1.0
    v_ndc = 1.0 - (vv + 0.5) / height * 2.0

    aspect_ratio = width / height

    tan_fov_half = torch.tan(torch.tensor(fov / 2.0))
    u_camera = u_ndc * tan_fov_half
    v_camera = v_ndc * tan_fov_half / aspect_ratio

    x_camera = torch.ones_like(u_camera)
    # +Y is left (u_ndc grows to the right), +Z is up (v_ndc grows upwards)
    directions = torch.stack([x_camera, -u_camera, v_camera], dim=-1)

    directions = directions / directions.norm(dim=-1, keepdim=True)

    return directions


def _distinct_debug_color(instance_id: int) -> Tuple[float, float, float]:
    """Pick a saturated, high-contrast RGB color for debug markers."""
    hue = (instance_id * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.85, 0.95)

class raycast_camera(ObservationV2):
    """Depth (and optionally surface-normal) image rendered by GPU raycasting.

    Rays are generated with a pinhole model (:func:`raymap`), mounted on a robot
    body with a fixed rotation/translation offset, and cast against the ground
    plus optional dynamic entities via ``simple_raycaster``.

    Output shape:

    * ``normal=False`` — ``[num_envs, 1, H, W]`` ray-hit distances in
      ``[0, far]``.
    * ``normal=True`` — ``[num_envs, 1 + 3, H, W]``: the distance channel
      concatenated with the hit surface normal expressed in the **camera frame**
      (+X forward, +Y left, +Z up, mount rotation included). Normals are
      oriented to face the camera (``n · ray_dir <= 0``) and are zero for rays
      that miss.

    Args:
        resolution: Image size as ``(width, height)``.
        fov_deg: Horizontal field of view in degrees (vertical follows from the
            aspect ratio).
        offset_pos: Fixed camera position offset in the body frame.
        offset_rpy: Fixed roll/pitch/yaw mount rotation of the camera relative
            to the body frame, in degrees (XYZ convention).
        body_name: Body to mount the camera on. Defaults to the root link.
        near: Rays start ``near`` meters along the ray direction (avoids
            self-hits with the mounting body).
        far: Maximum ray distance; misses return ``far``.
        normal: If True, append camera-frame surface normals as 3 extra
            channels.
        dtype: Output dtype (``float32`` or ``float16``).
        targets: Optional scene entity names to raycast against in addition to
            ``/World/ground``.

    Debug visualization (GUI / Viser): hit points are drawn as sphere markers,
    and a camera frustum shows the depth image (``normal=False``, near = bright)
    or the normal map (``normal=True``, RGB = camera-frame ``n * 0.5 + 0.5``)
    for env 0.
    """

    supported_backends = ("isaac",)
    _debug_instance_count = 0

    supported_dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
    }

    def __init__(
        self,
        resolution: Tuple[int, int],
        fov_deg: float,
        offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        offset_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        body_name: Optional[str] = None,
        near: float = 0.01,
        far: float = 100.0,
        normal: bool = False,
        dtype: torch.dtype | str = torch.float16,
        targets: Optional[List[str]] = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.fov_deg = fov_deg
        self.offset_pos = tuple(float(x) for x in offset_pos)
        self.offset_rpy = tuple(float(x) for x in offset_rpy)
        self.body_name = body_name
        self.near = near
        self.far = far
        self.dtype = dtype
        self.normal = normal
        self.targets = targets

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.dtype = (
            self.supported_dtypes[self.dtype] if isinstance(self.dtype, str) else self.dtype
        )
        assert self.dtype in self.supported_dtypes.values(), f"Unsupported dtype: {self.dtype}"
        assert self.far - self.near > 1e-6, "Far must be greater than near"

        width, height = self.resolution
        self.raymap = raymap(width, height, self.fov_deg / 180.0 * torch.pi).to(self.device)
        euler = torch.tensor(self.offset_rpy, device=self.device) / 180.0 * torch.pi
        # body → camera mount rotation; also needed to express normals in the camera frame
        self.mount_quat = quat_from_euler_xyz(euler).reshape(1, 1, 4)
        self.raymap = quat_rotate(self.mount_quat, self.raymap)
        self._offset_pos = torch.tensor(self.offset_pos, device=self.device)

        self.shape = self.raymap.shape[:2]
        assert self.shape == (height, width), "Resolution must match the raymap shape"
        self.num_rays = self.raymap.shape[0] * self.raymap.shape[1]

        from simple_raycaster import MultiMeshRaycasterV2

        self.raycaster = MultiMeshRaycasterV2(device=self.device)
        self.raycaster.add_isaac_static("/World/ground")
        if self.targets is not None:
            for target in self.targets:
                target_asset = self.env.scene[target]
                self.raycaster.add_isaac_entity(target_asset)

        if self.body_name is not None:
            self.body_id = self.asset.find_bodies(self.body_name)[0]
            assert len(self.body_id) == 1, f"Multiple bodies found for name {self.body_name}"
            self.body_id = self.body_id[0]
        else:
            self.body_id = None

        self.camera_handle = None
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            scene: IsaacSceneAdapter = self.env.scene
            self.instance_id = raycast_camera._debug_instance_count
            raycast_camera._debug_instance_count += 1
            marker_color = _distinct_debug_color(self.instance_id)
            self.marker = scene.create_sphere_marker(
                f"/Visuals/Command/raycast_camera_{self.instance_id}",
                marker_color,
                radius=0.02,
            )

            # Viser frustum showing the depth (or normal) image. The frustum
            # uses the ROS camera convention (+Z forward, +Y down); compose the
            # mount rotation with the fixed ROS → raymap-frame rotation.
            fov_x = self.fov_deg / 180.0 * math.pi
            aspect = width / max(height, 1)
            fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) / aspect)
            ros_to_cam = torch.tensor([0.5, -0.5, 0.5, -0.5], device=self.device)
            self._frustum_quat = quat_mul(self.mount_quat.reshape(4), ros_to_cam)
            try:
                self.camera_handle = self.env.scene.create_camera_frustum(
                    f"raycast_camera_{self.instance_id}",
                    fov_y=fov_y,
                    aspect=aspect,
                )
            except Exception as e:
                print(f"Error creating camera frustum: {e}")
                self.camera_handle = None

    def compute(self) -> torch.Tensor:
        """Cast the ray bundle and assemble the image observation.

        Returns:
            ``[num_envs, 1, H, W]`` hit distances, or ``[num_envs, 4, H, W]``
            with camera-frame normals appended when ``normal=True``.
        """
        if self.body_id is not None:
            body_pos_w = self.asset.data.body_link_pos_w[:, self.body_id]
            body_quat = self.asset.data.body_link_quat_w[:, self.body_id]
        else:
            body_pos_w = self.asset.data.root_link_pos_w
            body_quat = self.asset.data.root_link_quat_w
        self.ray_dirs_w = quat_rotate(
            body_quat.unsqueeze(1), self.raymap.reshape(1, self.num_rays, 3)
        )
        offset_w = quat_rotate(body_quat, self._offset_pos.unsqueeze(0))
        self.ray_starts_w = (
            body_pos_w.reshape(self.num_envs, 1, 3)
            + offset_w.reshape(self.num_envs, 1, 3)
            + self.ray_dirs_w * self.near
        )

        hit_pos_w, hit_distance, hit_normal_w = self.raycaster.raycast_fused(
            ray_starts_w=self.ray_starts_w,
            ray_dirs_w=self.ray_dirs_w,
            min_dist=0.0,
            max_dist=self.far,
        )
        self.ray_hits_w = hit_pos_w

        hit_distance = hit_distance.nan_to_num(posinf=self.far).to(self.dtype)
        depth = hit_distance.reshape(self.num_envs, 1, self.shape[0], self.shape[1])
        if not self.normal:
            self._image = depth
            return depth

        # Raw face normals have winding-dependent sign; orient them towards the
        # camera so n · ray_dir <= 0. Misses stay zero.
        flip = (hit_normal_w * self.ray_dirs_w).sum(-1, keepdim=True) > 0
        hit_normal_w = torch.where(flip, -hit_normal_w, hit_normal_w)
        # world → body → camera frame
        hit_normal_b = quat_rotate_inverse(body_quat.unsqueeze(1), hit_normal_w)
        hit_normal_c = quat_rotate_inverse(self.mount_quat, hit_normal_b)
        hit_normal_c = (
            hit_normal_c.reshape(self.num_envs, self.shape[0], self.shape[1], 3)
            .permute(0, 3, 1, 2)
            .to(self.dtype)
        )
        self._image = torch.cat([depth, hit_normal_c], dim=1)
        return self._image

    def _debug_image_hwc_uint8(self, env_idx: int) -> "np.ndarray":
        """Render the latest observation of one env as an HWC uint8 RGB image."""
        img = self._image[env_idx].float()  # [C, H, W]
        if self.normal:
            # camera-frame normals in [-1, 1] → RGB; misses (zero normals) → gray
            rgb = img[1:4] * 0.5 + 0.5
        else:
            # near = bright, far / miss = dark
            rgb = (1.0 - img[0] / self.far).clamp(0.0, 1.0).unsqueeze(0).expand(3, -1, -1)
        return (rgb * 255.0).byte().permute(1, 2, 0).cpu().numpy()

    def debug_draw(self) -> None:
        if self.env.backend != "isaac":
            return
        pos = self.ray_hits_w[0].reshape(-1, 3)
        self.marker.visualize(pos)

        if self.camera_handle is None:
            return
        env_idx = 0
        if self.body_id is not None:
            body_pos = self.asset.data.body_link_pos_w[env_idx, self.body_id]
            body_quat = self.asset.data.body_link_quat_w[env_idx, self.body_id]
        else:
            body_pos = self.asset.data.root_link_pos_w[env_idx]
            body_quat = self.asset.data.root_link_quat_w[env_idx]
        self.camera_handle.position = body_pos + quat_rotate(body_quat, self._offset_pos)
        self.camera_handle.wxyz = quat_mul(body_quat, self._frustum_quat)
        self.camera_handle.image = self._debug_image_hwc_uint8(env_idx)

    def symmetry_transform(self):
        """Mirror transform for the image observation.

        Under the left-right (xz-plane) mirror of the world, the camera sees
        the horizontally flipped image: the raymap places +Y (left) along
        decreasing width, so the transform permutes the width (last) dim.
        Distance is invariant per mirrored pixel; camera-frame normals keep
        their forward/up components but negate the lateral (y) component,
        expressed via per-channel signs on ``[.., C, H, W]``.

        Only valid for a mirror-symmetric mount: zero roll/yaw and zero
        lateral position offset.
        """
        roll, _, yaw = self.offset_rpy
        if roll != 0.0 or yaw != 0.0 or self._offset_pos[1].item() != 0.0:
            raise NotImplementedError(
                "raycast_camera symmetry requires a mirror-symmetric mount: "
                f"roll=yaw=0 and zero lateral offset, got offset_rpy={self.offset_rpy}, "
                f"offset_pos={self._offset_pos.tolist()}"
            )
        perm = torch.arange(self.shape[1]).flip(0)
        signs = torch.ones(self.shape[1])
        x = torch.arange(self.shape[0] * self.shape[1]).reshape(1, 1, *self.shape)
        y = x.flip(3)
        assert torch.all(y == x[..., perm]), "raycast_camera symmetry permutation mismatch"
        # channels: (distance, nx, ny, nz) — negate the lateral normal component
        channel_signs = torch.tensor([1.0, 1.0, -1.0, 1.0]) if self.normal else None
        return SymmetryTransform(perm=perm, signs=signs, channel_signs=channel_signs)


class camera_isaac(ObservationV2):
    """Isaac Lab tiled camera observation for the Isaac backend.

    Registers a :class:`~isaaclab.sensors.TiledCameraCfg` on the scene during
    :meth:`edit_spec` (called from :meth:`IsaacBackendEnv.setup_scene` before
    the scene is built). After simulation startup, :meth:`compute` reads
    ``camera.data.output[data_type]``, optionally normalizes it, and returns a
    ``(num_envs, C, H, W)`` tensor.

    Args:
        resolution: ``(width, height)`` in pixels.
        data_type: Camera output key, e.g. ``"rgb"`` or ``"depth"``.
        focal_length: Pinhole focal length in mm.
        focus_distance: Pinhole focus distance in m.
        horizontal_aperture: Pinhole horizontal aperture in mm.
        clipping_range: Near/far clipping planes in m.
        body_name: If set, attach the camera to ``Robot/{body_name}``; otherwise
            spawn a standalone camera under each env namespace.
        sensor_name: Scene sensor attribute name. Defaults to a unique
            ``tiled_camera_{id}`` per instance so multiple cameras can coexist.
        offset_pos: Camera offset translation w.r.t. its parent frame.
        offset_rpy: Camera offset rotation as XYZ Euler angles in degrees
            w.r.t. parent frame (converted to WXYZ for Isaac).
        offset_convention: Offset frame convention (``"ros"``, ``"world"``, or
            ``"opengl"``).
    """

    supported_backends = ("isaac",)
    _instance_count = 0

    def __init__(
        self,
        resolution: Tuple[int, int],
        data_type: str = "rgb",
        focal_length: float = 24.0,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 20.955,
        clipping_range: Tuple[float, float] = (0.1, 20.0),
        body_name: Optional[str] = None,
        sensor_name: Optional[str] = None,
        offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        offset_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        offset_convention: str = "world",
    ):
        super().__init__()
        self.resolution = resolution
        self.data_type = data_type
        self.focal_length = focal_length
        self.focus_distance = focus_distance
        self.horizontal_aperture = horizontal_aperture
        self.clipping_range = tuple(clipping_range)
        self.body_name = body_name
        self.offset_pos = tuple(float(x) for x in offset_pos)
        self.offset_rpy = tuple(float(x) for x in offset_rpy)
        self.offset_convention = offset_convention

        if sensor_name is None:
            camera_isaac._instance_count += 1
            self.sensor_name = f"tiled_camera_{camera_isaac._instance_count}"
        else:
            self.sensor_name = sensor_name

    @override
    def edit_spec(self, scene_config: InteractiveSceneCfg) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sensors import TiledCameraCfg

        if hasattr(scene_config, self.sensor_name):
            raise ValueError(
                f"Scene config already has sensor '{self.sensor_name}'. "
                "Choose a distinct sensor_name for each camera_isaac instance."
            )

        if self.body_name is not None:
            camera_mount_cfg = None
            prim_path = f"{{ENV_REGEX_NS}}/Robot/{self.body_name}/{self.sensor_name}"
        else:
            # As of Isaac Sim 5.1.0, there is a bug that prevents setting the pose
            # of TiledCamera dynamically during simulation using `set_world_poses`
            # Therefore we attach it to a dummy body
            # TODO@btx0424: check if this is still needed after Isaac Sim 6.0.0
            camera_mount_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{self.sensor_name}_mount",
                spawn=sim_utils.SphereCfg(
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                    ),
                    radius=0.02,
                )
            )
            prim_path = f"{{ENV_REGEX_NS}}/{self.sensor_name}_mount/{self.sensor_name}"

        cfg = TiledCameraCfg(
            prim_path=prim_path,
            offset=TiledCameraCfg.OffsetCfg(
                pos=self.offset_pos,
                rot=_offset_rpy_deg_to_quat(self.offset_rpy),
                convention=self.offset_convention,
            ),
            data_types=[self.data_type],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=self.focal_length,
                focus_distance=self.focus_distance,
                horizontal_aperture=self.horizontal_aperture,
                clipping_range=self.clipping_range,
            ),
            width=self.resolution[0],
            height=self.resolution[1],
            update_latest_camera_pose=True,
        )
        if camera_mount_cfg is not None:
            setattr(scene_config, self.sensor_name + "_mount", camera_mount_cfg)
        setattr(scene_config, self.sensor_name, cfg)

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.env.sensor_render_enabled = True
        if self.body_name is None:
            self.camera_mount: RigidObject = self.env.scene.entities[self.sensor_name + "_mount"]
        self.camera: TiledCamera = self.env.scene.sensors[self.sensor_name]

    @override
    def update(self) -> None:
        if self.body_name is None:
            robot_root_pos_w = self.env.scene.entities["robot"].data.root_link_pos_w
            eye = robot_root_pos_w + torch.tensor([2.0, 2.0, 2.0], device=self.device)
            target = robot_root_pos_w
            self.camera_mount.write_root_link_pose_to_sim(
                root_pose_from_view_z_up(eye, target)
            )

    @override
    def compute(self) -> torch.Tensor:
        # for rgb, isaac sim returns uint8, which we leave as is
        data = self.camera.data.output[self.data_type]  # NHWC
        return einops.rearrange(data, "n h w c -> n c h w")

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        if self.body_name is None:
            raise NotImplementedError("Symmetry transform is only available when the camera is attached to a body")
        width = self.resolution[0]
        perm = torch.arange(width - 1, -1, -1, dtype=torch.long)
        return SymmetryTransform(perm, torch.ones(width))


class camera_mjlab(ObservationV2):
    """MjLab camera observation using :class:`~mjlab.sensor.CameraSensor`.

    Registers a :class:`~mjlab.sensor.CameraSensorCfg` during scene construction
    (``edit_spec`` appends to the mjlab sensor list). Rendering runs via
    ``sim.render_sensors()`` → ``Simulation.sense()`` each control step;
    :meth:`compute` returns ``(N, C, H, W)``.

    MuJoCo cameras use **fixed** mode: ``pos``/``quat`` are set in the spec at
    build time. Attach via ``body_name`` (``robot/{name}``) or spawn on the
    worldbody with ``offset_pos`` / ``offset_rpy``. Dynamic tracking (e.g. via a
    dummy mocap body) is not implemented yet.

    Args:
        resolution: ``(width, height)`` in pixels.
        data_type: ``"rgb"``, ``"depth"``, or ``"segmentation"``.
        fovy: Vertical field of view in degrees. If ``None``, derived from
            ``focal_length`` and ``horizontal_aperture``.
        focal_length: Pinhole focal length in mm (used when ``fovy`` is None).
        horizontal_aperture: Pinhole horizontal aperture in mm.
        body_name: Robot body to attach the camera to (``robot/{name}``).
        sensor_name: Scene sensor key. Defaults to ``mjlab_camera_{id}``.
        offset_pos: Camera position w.r.t. parent body or world frame.
        offset_rpy: Camera orientation as XYZ Euler angles in degrees w.r.t.
            parent frame (converted to WXYZ for MuJoCo).
    """

    supported_backends = ("mjlab",)
    _instance_count = 0

    def __init__(
        self,
        resolution: Tuple[int, int],
        data_type: str = "rgb",
        fovy: Optional[float] = None,
        focal_length: float = 24.0,
        horizontal_aperture: float = 20.955,
        body_name: Optional[str] = None,
        sensor_name: Optional[str] = None,
        offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        offset_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        super().__init__()
        self.resolution = resolution
        self.data_type = data_type
        self.fovy = fovy
        self.focal_length = focal_length
        self.horizontal_aperture = horizontal_aperture
        self.body_name = body_name
        self.offset_pos = tuple(float(x) for x in offset_pos)
        self.offset_rpy = tuple(float(x) for x in offset_rpy)

        if sensor_name is None:
            camera_mjlab._instance_count += 1
            self.sensor_name = f"mjlab_camera_{camera_mjlab._instance_count}"
        else:
            self.sensor_name = sensor_name

    def _resolved_fovy(self) -> float:
        if self.fovy is not None:
            return self.fovy
        width, height = self.resolution
        h_fov = 2.0 * math.atan(self.horizontal_aperture / (2.0 * self.focal_length))
        v_fov = 2.0 * math.atan(math.tan(h_fov / 2.0) * height / width)
        return math.degrees(v_fov)

    @override
    def edit_spec(self, scene_config: SceneCfg) -> None:
        from mjlab.sensor import CameraSensorCfg
        parent_body = f"robot/{self.body_name}" if self.body_name is not None else None
        scene_config.sensors += (
            CameraSensorCfg(
                name=self.sensor_name,
                parent_body=parent_body,
                pos=self.offset_pos,
                quat=_offset_rpy_deg_to_quat(self.offset_rpy),
                fovy=self._resolved_fovy(),
                width=self.resolution[0],
                height=self.resolution[1],
                data_types=(self.data_type,),
            ),
        )

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.env.sensor_render_enabled = True
        self.camera: CameraSensor = self.env.scene.sensors[self.sensor_name]

    @override
    def compute(self) -> torch.Tensor:
        data = self.camera.data
        if self.data_type == "rgb":
            output = data.rgb
        elif self.data_type == "depth":
            output = data.depth.clone()
            output[torch.isinf(output)] = 0.0
        elif self.data_type == "segmentation":
            output = data.segmentation
        else:
            raise ValueError(f"Unsupported camera data_type: {self.data_type!r}")
        return einops.rearrange(output, "n h w c -> n c h w")

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        if self.body_name is None:
            raise NotImplementedError(
                "camera_mjlab symmetry is only defined for body-mounted cameras"
            )
        width = self.resolution[0]
        perm = torch.arange(width - 1, -1, -1, dtype=torch.long)
        return SymmetryTransform(perm, torch.ones(width))


class raycast_camera_v2(ObservationV2):
    """Read RGB-D from a scene-owned :class:`~active_adaptation.envs.sensors.camera.LambertRaycastCameraSensor`.

    Unlike :class:`raycast_camera`, this term does **not** create or own a
    raycaster. Declare the sensor in task YAML under ``sensors:``::

        sensors:
          front_rgbd:
            _target_: lambert_raycast_camera
            entity: robot
            body_name: base_link
            resolution: [128, 96]
            fov_y_deg: 70.0
            targets: [terrain]

        observation:
          policy:
            raycast_camera_v2:
              sensor_name: front_rgbd
              data_type: depth

    Args:
        sensor_name: Key in ``env.scene.sensors`` (must be a Lambert raycast camera).
        data_type: ``\"depth\"`` → ``[N, 1, H, W]``; ``\"rgb\"`` → ``[N, 3, H, W]``;
            ``\"rgbd\"`` → ``[N, 4, H, W]`` (RGB then depth); ``\"mask\"`` →
            ``[N, 1, H, W]`` float occupancy.
    """

    def __init__(
        self,
        sensor_name: str,
        data_type: str = "depth",
    ) -> None:
        super().__init__()
        if data_type not in ("depth", "rgb", "rgbd", "mask"):
            raise ValueError(
                f"data_type must be depth|rgb|rgbd|mask, got {data_type!r}"
            )
        self.sensor_name = sensor_name
        self.data_type = data_type

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        from active_adaptation.envs.sensors.camera import LambertRaycastCameraSensor

        sensor = self.env.scene.sensors.get(self.sensor_name)
        if sensor is None:
            raise KeyError(
                f"raycast_camera_v2: sensor {self.sensor_name!r} not in "
                f"scene.sensors (have {sorted(self.env.scene.sensors)})"
            )
        if not isinstance(sensor, LambertRaycastCameraSensor):
            raise TypeError(
                f"raycast_camera_v2 expects LambertRaycastCameraSensor, "
                f"got {type(sensor).__name__} for {self.sensor_name!r}"
            )
        self.sensor: LambertRaycastCameraSensor = sensor
        self.camera_handle = None
        if not self.env.sim.has_gui():
            return
        aspect = self.sensor.width / max(self.sensor.height, 1)
        fov_y = math.radians(self.sensor.fov_y_deg)
        try:
            self.camera_handle = self.env.scene.create_camera_frustum(
                f"raycast_camera_v2_{self.sensor_name}",
                fov_y=fov_y,
                aspect=aspect,
            )
        except Exception as e:
            print(f"Error creating camera frustum: {e}")
            self.camera_handle = None

    def _debug_image_hwc_uint8(self, env_idx: int) -> "np.ndarray":
        """Latest observation of one env as HWC uint8 RGB for the Viser frustum."""
        data = self.sensor.data
        if self.data_type in ("rgb", "rgbd"):
            rgb = data.rgb[env_idx].float().clamp(0.0, 1.0)
        elif self.data_type == "mask":
            rgb = data.mask[env_idx].float().unsqueeze(-1).expand(-1, -1, 3)
        else:
            # near = bright, far / miss = dark
            rgb = (1.0 - data.depth[env_idx] / max(self.sensor.far, 1e-6)).clamp(
                0.0, 1.0
            ).unsqueeze(-1).expand(-1, -1, 3)
        return (rgb * 255.0).byte().cpu().numpy()

    @override
    def compute(self) -> torch.Tensor:
        data = self.sensor.data
        if self.data_type == "depth":
            return data.depth.unsqueeze(1)
        if self.data_type == "rgb":
            return data.rgb.permute(0, 3, 1, 2).contiguous()
        if self.data_type == "mask":
            return data.mask.float().unsqueeze(1)
        # rgbd
        rgb = data.rgb.permute(0, 3, 1, 2)
        depth = data.depth.unsqueeze(1)
        return torch.cat([rgb, depth], dim=1)

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        width = int(self.sensor.width)
        perm = torch.arange(width - 1, -1, -1, dtype=torch.long)
        return SymmetryTransform(perm, torch.ones(width))
    
    @override
    def debug_draw(self) -> None:
        if self.camera_handle is None:
            return
        env_idx = 0
        viser = getattr(self.env.sim, "_viser_viewer", None) or getattr(
            self.env.sim, "_viewer", None
        )
        if viser is not None:
            if hasattr(viser, "env_idx"):
                env_idx = int(viser.env_idx)
            else:
                scene = getattr(viser, "_scene", None)
                if scene is not None and hasattr(scene, "env_idx"):
                    env_idx = int(scene.env_idx)
        self.camera_handle.position = self.sensor.cam_pos_w[env_idx]
        # Lambert RaycastCamera is OpenCV (+Z fwd, +Y down) = Viser ROS frustum.
        self.camera_handle.wxyz = self.sensor.cam_quat_w[env_idx]
        self.camera_handle.image = self._debug_image_hwc_uint8(env_idx)


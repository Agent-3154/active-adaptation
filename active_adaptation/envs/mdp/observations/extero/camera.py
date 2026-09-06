from __future__ import annotations

import colorsys
import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import einops
import torch
from jaxtyping import Float
from typing_extensions import override

from ..base import Observation
from active_adaptation.utils.math import (
    quat_from_euler_xyz,
    root_pose_from_view_z_up,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from tensordict import TensorDictBase

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



class camera_isaac(Observation):
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

    supported_backends = ("isaaclab",)
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


class camera_mjlab(Observation):
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


class raycast_camera(Observation):
    """Lambert RGB-D from a shared :class:`~active_adaptation.envs.sensors.camera.LambertRaycastCameraSensor`.

    The observation owns the virtual camera mount and calls
    :meth:`~active_adaptation.envs.sensors.camera.LambertRaycastCameraSensor.render`
    each step. Multiple obs terms may share one sensor/renderer.

    ``dtype`` controls hold-buffer / returned image precision (``float32`` or
    ``float16``). Render kernels stay fp32; values are cast on store. Useful to
    cut rollout VRAM; compatible with PPO AMP (update-time autocast is separate).

        sensors:
          shared_camera:
            _target_: lambert_raycast_camera
            resolution: [128, 96]
            targets: [terrain, robot]

        observation:
          head_camera_:
            raycast_camera:
              sensor_name: shared_camera
              body_name: gripper_base
              data_type: depth
              dtype: float16
    """

    supported_backends = ("isaaclab", "mjlab")
    _DATA_TYPES = frozenset({"depth", "rgb", "rgbd", "mask", "normal"})
    _DTYPE_ALIASES = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }

    def __init__(
        self,
        sensor_name: str,
        body_name: str | None = None,
        pattern: str | None = None,
        entity: str = "robot",
        offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        offset_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        frequency: float = 50.0,
        data_type: str | Sequence[str] = "depth",
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        if body_name is None and pattern is None:
            raise ValueError("raycast_camera requires body_name or pattern")
        if isinstance(data_type, str):
            data_types = (data_type,)
        else:
            data_types = tuple(str(x) for x in data_type)
        unknown = set(data_types) - self._DATA_TYPES
        if unknown:
            raise ValueError(f"unknown data_type entries {sorted(unknown)}")
        if "normal" in data_types:
            raise NotImplementedError(
                "raycast_camera normal output is not implemented for Lambert RGB-D"
            )
        if not data_types:
            raise ValueError("data_type must be non-empty")
        dtype_key = str(dtype).lower()
        if dtype_key not in self._DTYPE_ALIASES:
            raise ValueError(
                f"raycast_camera dtype must be one of {sorted(self._DTYPE_ALIASES)}, "
                f"got {dtype!r}"
            )
        self.sensor_name = sensor_name
        self.body_name = body_name
        self.pattern = pattern
        self.entity_name = entity
        self.offset_pos = tuple(float(x) for x in offset_pos)
        self.offset_rpy_deg = tuple(float(x) for x in offset_rpy_deg)
        self.data_types = data_types
        self.frequency = frequency
        self._image_dtype = self._DTYPE_ALIASES[dtype_key]
        self._needs_rgb = any(k in data_types for k in ("rgb", "rgbd"))
        self._body_id: int | None = None
        self._mount_entity = None
        self._interval: float = 1.0 / self.frequency
        self.camera_handle = None

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        from active_adaptation.envs.sensors.camera import LambertRaycastCameraSensor

        sensor = self.env.scene.sensors.get(self.sensor_name)
        if sensor is None:
            raise KeyError(
                f"raycast_camera: sensor {self.sensor_name!r} not in "
                f"scene.sensors (have {sorted(self.env.scene.sensors)})"
            )
        if not isinstance(sensor, LambertRaycastCameraSensor):
            raise TypeError(
                f"raycast_camera expects LambertRaycastCameraSensor, "
                f"got {type(sensor).__name__} for {self.sensor_name!r}"
            )
        self.sensor: LambertRaycastCameraSensor = sensor

        mount = self.env.scene.entities[self.entity_name]
        if self.body_name is not None:
            body_ids, body_names = mount.find_bodies(self.body_name)
        else:
            body_ids, body_names = mount.find_bodies(self.pattern)
        if len(body_ids) != 1:
            raise ValueError(
                f"raycast_camera on {self.entity_name!r}: expected one mount body, "
                f"got {body_names}"
            )
        self._body_id = int(body_ids[0])
        self._mount_entity = mount

        self.offset_pos = torch.tensor(self.offset_pos, device=self.device)
        self.offset_rpy = torch.tensor(self.offset_rpy_deg, device=self.device) * (math.pi / 180.0)
        h, w = self.sensor.height, self.sensor.width
        n = self.env.num_envs
        img_dtype = self._image_dtype
        # Always keep depth for debug viz when only mask is requested.
        self._last_rgb: torch.Tensor | None = (
            torch.zeros(n, h, w, 3, device=self.device, dtype=img_dtype)
            if self._needs_rgb
            else None
        )
        self._last_depth: torch.Tensor = torch.zeros(
            n, h, w, device=self.device, dtype=img_dtype
        )
        self._last_mask: torch.Tensor = torch.zeros(
            n, h, w, device=self.device, dtype=torch.bool
        )
        
        self._cam_pos_w = torch.zeros(self.env.num_envs, 3, device=self.device)
        self._cam_quat_w = torch.zeros(self.env.num_envs, 4, device=self.device)
        self._cam_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self._time_until_next_render = torch.zeros(self.env.num_envs, 1, device=self.device)
        self._should_render = torch.ones(self.env.num_envs, 1, device=self.device, dtype=torch.bool)

        if self.env.sim.has_gui():
            aspect = self.sensor.width / max(self.sensor.height, 1)
            fov_y = math.radians(self.sensor.fov_y_deg)
            try:
                self.camera_handle = self.env.scene.create_camera_frustum(
                    f"raycast_camera_{self.sensor_name}_{id(self)}",
                    fov_y=fov_y,
                    aspect=aspect,
                )
            except Exception as e:
                print(f"Error creating camera frustum: {e}")
                self.camera_handle = None

    def _format_output(self) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for key in self.data_types:
            if key == "depth":
                parts.append(self._last_depth.unsqueeze(1))
            elif key == "rgb":
                assert self._last_rgb is not None
                parts.append(self._last_rgb.permute(0, 3, 1, 2).contiguous())
            elif key == "mask":
                parts.append(self._last_mask.to(dtype=self._image_dtype).unsqueeze(1))
            elif key == "rgbd":
                assert self._last_rgb is not None
                rgb = self._last_rgb.permute(0, 3, 1, 2)
                parts.append(torch.cat([rgb, self._last_depth.unsqueeze(1)], dim=1))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def _debug_image_hwc_uint8(self, env_idx: int) -> "np.ndarray":
        if "rgb" in self.data_types or "rgbd" in self.data_types:
            assert self._last_rgb is not None
            rgb = self._last_rgb[env_idx].float().clamp(0.0, 1.0)
        elif "mask" in self.data_types:
            rgb = self._last_mask[env_idx].float().unsqueeze(-1).expand(-1, -1, 3)
        else:
            rgb = (1.0 - self._last_depth[env_idx].float() / max(self.sensor.far, 1e-6)).clamp(
                0.0, 1.0
            ).unsqueeze(-1).expand(-1, -1, 3)
        rgb_uint8 = (rgb * 255.0).byte().cpu().numpy()
        return rgb_uint8
    
    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase | None = None) -> None:
        self._time_until_next_render[env_ids] = 0.0
        self._should_render[env_ids] = True
    
    @override
    def post_step(self, substep: int) -> None:
        if substep == 0:
            self._should_render.zero_()
        del substep
        self._should_render |= self._time_until_next_render <= 0.0
        came_pos_w, came_quat_w = self.sensor.mount_pose(
            self._mount_entity,
            self._body_id,
            self.offset_pos,
            self.offset_rpy,
        )
        self._cam_pos_w = torch.where(self._should_render, came_pos_w, self._cam_pos_w)
        self._cam_quat_w = torch.where(self._should_render, came_quat_w, self._cam_quat_w)
        self._time_until_next_render = torch.where(
            self._should_render,
            self._interval,
            self._time_until_next_render - self.env.physics_dt
        )

    @override
    def compute(self) -> torch.Tensor:
        assert self._body_id is not None and self._mount_entity is not None
        due = self._should_render.reshape(self.num_envs)
        rgb, depth, mask = self.sensor.render(
            self._cam_pos_w,
            self._cam_quat_w,
            enabled=due,
            clone=False,
        )
        if due.any():
            if self._last_rgb is not None:
                self._last_rgb[due] = rgb[due].to(dtype=self._image_dtype)
            self._last_depth[due] = depth[due].to(dtype=self._image_dtype)
            self._last_mask[due] = mask[due]
        return self._format_output()

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        width = int(self.sensor.width)
        perm = torch.arange(width - 1, -1, -1, dtype=torch.long)
        return SymmetryTransform(perm, torch.ones(width))
    
    @override
    def debug_draw(self) -> None:
        if self.camera_handle is None or self._cam_pos_w is None or self._cam_quat_w is None:
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
        self.camera_handle.position = self._cam_pos_w[env_idx]
        self.camera_handle.wxyz = self._cam_quat_w[env_idx]
        self.camera_handle.image = self._debug_image_hwc_uint8(env_idx)


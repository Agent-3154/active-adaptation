"""Browser Viser viewer for the Isaac backend (mjlab-parity).

Uploads robot body visuals once via ``simple_raycaster.utils_usd`` (Mesh/Cube
only, no materials), then each step writes ``body_link_pose_w`` into batched
mesh handles. Also provides DebugDraw-compatible primitives and camera
frustum registration for observation image debug.

Requires the ``viser`` and ``simple-raycaster`` packages in the environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase
    import viser as _viser_types

import viser


def _rgba_to_rgb255(color: tuple[float, ...] | list[float]) -> tuple[int, int, int]:
    r, g, b = float(color[0]), float(color[1]), float(color[2])
    if max(r, g, b) <= 1.0:
        return (int(r * 255), int(g * 255), int(b * 255))
    return (int(r), int(g), int(b))


def _load_entity_body_meshes(entity) -> list[tuple[str, Any]]:
    """Extract body-local visual trimeshes (shared helper with scene adapter)."""
    from active_adaptation.envs.backends.isaaclab.meshes import load_entity_body_meshes

    meshes = load_entity_body_meshes(entity, suffixes=("visuals",), require_all=True)
    return list(zip(entity.body_names, meshes))


class IsaacViserViewer:
    """Synchronous Isaac browser viewer (analogous to ``MjLabViewer``).

    Mesh extraction uses ``simple_raycaster.utils_usd`` (Mesh/Cube only,
    untextured). Poses come from ``entity.data.body_link_pose_w``.
    """

    def __init__(self, env: "_EnvBase"):
        self.env = env
        self._server = viser.ViserServer(label="isaaclab")
        self._is_setup = False

        self.env_idx: int = 0
        self.show_all_envs: bool = False

        self._entity = None
        self._body_names: list[str] = []
        self._mesh_handles: list[Any] = []
        self._mesh_batch: int = 0

        self._ground_handle: Any | None = None
        self._line_handle: Any | None = None
        self._point_handle: Any | None = None
        self._debug_line_pts: list[np.ndarray] = []
        self._debug_line_cols: list[np.ndarray] = []
        self._debug_point_pts: list[np.ndarray] = []
        self._debug_point_cols: list[np.ndarray] = []
        self._debug_point_size: float = 0.02

        self._cameras: dict[str, viser.CameraFrustumHandle] = {}
        self._gaussian_handle: Any | None = None
        self._gaussian_origin_handle: Any | None = None
        self._collision_handle: Any | None = None

        self._gui_env_slider = None
        self._gui_show_all = None

    @property
    def server(self):
        return self._server

    def setup(self) -> None:
        if self._is_setup:
            return

        self._entity = self.env.scene.articulations["robot"]
        body_meshes = _load_entity_body_meshes(self._entity)
        self._body_names = [name for name, _ in body_meshes]

        self._upload_body_meshes(body_meshes)
        self._try_add_ground()
        self._setup_gui()
        self._is_setup = True

    def add_gaussian_splat(self, gs, *, name: str = "/visual/gaussians") -> None:
        """Upload an fvdb ``GaussianSplat3d`` for browser visualization (debug).

        Policy RGB still comes from ``env.visual.render`` (option A). Splats are
        uploaded hidden by default (browser 3DGS is expensive); prefer
        :meth:`add_collision_mesh` for scene geometry in Viser.

        Also places a coordinate frame at the splat origin (dataset PLYs are often
        not centered at the world origin).
        """
        if not self._is_setup:
            raise RuntimeError("IsaacViserViewer.setup() has not been called.")
        from active_adaptation.envs.visual.viser_export import (
            add_gaussian_splat_to_viser_server,
        )

        if self._gaussian_handle is not None:
            try:
                self._gaussian_handle.remove()
            except Exception:
                pass
            self._gaussian_handle = None
        if self._gaussian_origin_handle is not None:
            try:
                self._gaussian_origin_handle.remove()
            except Exception:
                pass
            self._gaussian_origin_handle = None

        self._gaussian_handle = add_gaussian_splat_to_viser_server(
            self._server, gs, name=name
        )
        # Dataset 3DGS frames are often unnormalized; mark the splat origin.
        self._gaussian_origin_handle = self._server.scene.add_frame(
            f"{name}/origin",
            show_axes=True,
            axes_length=0.5,
            axes_radius=0.02,
            origin_radius=0.04,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=True,
        )

    def add_collision_mesh(
        self,
        mesh,
        *,
        name: str = "/visual/collision",
        visible: bool = True,
    ) -> None:
        """Upload InteriorGS collision trimesh (same frame as the 3DGS PLY).

        Cheap browser stand-in for the Gaussian scene. Not yet used as PhysX
        collision (ground plane still owns sim contact).
        """
        if not self._is_setup:
            raise RuntimeError("IsaacViserViewer.setup() has not been called.")
        if self._collision_handle is not None:
            try:
                self._collision_handle.remove()
            except Exception:
                pass
            self._collision_handle = None
        self._collision_handle = self._server.scene.add_mesh_trimesh(
            name, mesh, visible=visible
        )

    def _setup_gui(self) -> None:
        with self._server.gui.add_folder("Scene"):
            self._gui_env_slider = self._server.gui.add_slider(
                "Env index",
                min=0,
                max=max(self.env.num_envs - 1, 0),
                step=1,
                initial_value=0,
            )

            @self._gui_env_slider.on_update
            def _on_env(_evt) -> None:
                self.env_idx = int(self._gui_env_slider.value)

            self._gui_show_all = self._server.gui.add_checkbox(
                "Show all envs",
                initial_value=False,
            )

            @self._gui_show_all.on_update
            def _on_show_all(_evt) -> None:
                # GUI callbacks run off the env step thread — only flip the flag.
                # Mesh handles stay sized to num_envs; update() parks hidden envs.
                self.show_all_envs = bool(self._gui_show_all.value)

    def _upload_body_meshes(self, body_meshes: list[tuple[str, Any]]) -> None:
        # Always allocate num_envs instances so toggling show_all never needs
        # remove/recreate (Viser GUI callbacks can race env-step update()).
        batch = self.env.num_envs
        self._mesh_handles = []
        identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (batch, 1))
        zeros = np.zeros((batch, 3), dtype=np.float32)
        for body_name, mesh in body_meshes:
            handle = self._server.scene.add_batched_meshes_trimesh(
                f"/robot/{body_name}",
                mesh,
                batched_wxyzs=identity,
                batched_positions=zeros,
            )
            self._mesh_handles.append(handle)
        self._mesh_batch = batch

    def _try_add_ground(self) -> None:
        """Visualize ``/World/ground``: infinite grid for PhysX planes, mesh otherwise.

        Matches :meth:`IsaacSceneAdapter.ground_mesh` plane-vs-mesh detection and
        mjlab/mjviser's ``add_grid`` look for infinite planes.
        """
        try:
            import isaaclab.sim as sim_utils
            from isaacsim.core.utils.stage import get_current_stage
            from simple_raycaster.utils_usd import find_matching_prims, get_trimesh_from_prim
        except ImportError:
            return

        mesh_prim_path = "/World/ground"
        # Same test as IsaacSceneAdapter.ground_mesh: PhysX Plane → infinite ground.
        plane_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
        )
        if plane_prim is not None:
            self._ground_handle = self._server.scene.add_grid(
                "/ground",
                infinite_grid=True,
                fade_distance=50.0,
                shadow_opacity=0.2,
                plane_opacity=0.4,
            )
            return

        prims = find_matching_prims(mesh_prim_path, get_current_stage())
        if not prims:
            return
        try:
            mesh = get_trimesh_from_prim(prims[0])
        except ValueError:
            return
        self._ground_handle = self._server.scene.add_mesh_trimesh("/ground", mesh)

    # ------------------------------------------------------------------
    # DebugDraw-compatible API
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._debug_line_pts.clear()
        self._debug_line_cols.clear()
        self._debug_point_pts.clear()
        self._debug_point_cols.clear()

    def vector(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        size: float = 2.0,
        color: tuple[float, ...] = (0.0, 1.0, 1.0, 1.0),
    ) -> None:
        del size  # line_width fixed at sync; kept for DebugDraw API parity
        x_np = x.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        v_np = v.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        if x_np.shape != v_np.shape:
            raise ValueError(f"x and v must match, got {x_np.shape} and {v_np.shape}")
        seg = np.stack([x_np, x_np + v_np], axis=1)  # (N, 2, 3)
        rgb = np.array(_rgba_to_rgb255(color), dtype=np.uint8)
        cols = np.broadcast_to(rgb, (seg.shape[0], 2, 3)).copy()
        self._debug_line_pts.append(seg)
        self._debug_line_cols.append(cols)

    def point(
        self,
        x: torch.Tensor,
        color: tuple[float, ...] = (1.0, 0.0, 0.0, 1.0),
        size: float = 10.0,
    ) -> None:
        pts = x.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        rgb = np.array(_rgba_to_rgb255(color), dtype=np.uint8)
        cols = np.broadcast_to(rgb, (pts.shape[0], 3)).copy()
        self._debug_point_pts.append(pts)
        self._debug_point_cols.append(cols)
        # Isaac DebugDraw sizes are in pixels; map roughly to world point size.
        self._debug_point_size = max(float(size) * 0.002, 0.005)

    def plot(
        self,
        x: torch.Tensor,
        size: float = 2.0,
        color: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        del size
        x_np = x.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        if x_np.shape[0] < 2:
            return
        seg = np.stack([x_np[:-1], x_np[1:]], axis=1)
        rgb = np.array(_rgba_to_rgb255(color), dtype=np.uint8)
        cols = np.broadcast_to(rgb, (seg.shape[0], 2, 3)).copy()
        self._debug_line_pts.append(seg)
        self._debug_line_cols.append(cols)

    def _sync_debug_geometry(self) -> None:
        if self._debug_line_pts:
            points = np.concatenate(self._debug_line_pts, axis=0)
            colors = np.concatenate(self._debug_line_cols, axis=0)
        else:
            points = np.zeros((0, 2, 3), dtype=np.float32)
            colors = np.zeros((0, 2, 3), dtype=np.uint8)

        if self._line_handle is None:
            # Placeholder segment so the handle exists when empty.
            init_pts = (
                points
                if points.shape[0] > 0
                else np.zeros((1, 2, 3), dtype=np.float32)
            )
            init_cols = (
                colors
                if colors.shape[0] > 0
                else np.zeros((1, 2, 3), dtype=np.uint8)
            )
            self._line_handle = self._server.scene.add_line_segments(
                "/debug/lines",
                init_pts,
                init_cols,
                line_width=2.0,
                visible=points.shape[0] > 0,
            )
        else:
            if points.shape[0] == 0:
                self._line_handle.visible = False
            else:
                self._line_handle.points = points
                self._line_handle.colors = colors
                self._line_handle.visible = True

        if self._debug_point_pts:
            pts = np.concatenate(self._debug_point_pts, axis=0)
            cols = np.concatenate(self._debug_point_cols, axis=0)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)
            cols = np.zeros((0, 3), dtype=np.uint8)

        if self._point_handle is None:
            init_pts = pts if pts.shape[0] > 0 else np.zeros((1, 3), dtype=np.float32)
            init_cols = cols if cols.shape[0] > 0 else np.zeros((1, 3), dtype=np.uint8)
            self._point_handle = self._server.scene.add_point_cloud(
                "/debug/points",
                init_pts,
                init_cols,
                point_size=self._debug_point_size,
                visible=pts.shape[0] > 0,
            )
        else:
            if pts.shape[0] == 0:
                self._point_handle.visible = False
            else:
                self._point_handle.points = pts
                self._point_handle.colors = cols
                self._point_handle.point_size = self._debug_point_size
                self._point_handle.visible = True

    # ------------------------------------------------------------------
    # Camera frustums
    # ------------------------------------------------------------------

    def register_camera(
        self,
        name: str,
        *,
        fov_y: float,
        aspect: float,
        scale: float = 0.15,
    ):
        """Create a Viser camera frustum (OpenCV +Z forward)."""
        if name in self._cameras:
            return self._cameras[name]
        handle = self._server.scene.add_camera_frustum(
            f"/cameras/{name}",
            fov=float(fov_y),
            aspect=float(aspect),
            scale=float(scale),
            color=(200, 200, 200),
            format="jpeg",
        )
        self._cameras[name] = handle
        return handle

    def set_camera(
        self,
        name: str,
        position: np.ndarray | torch.Tensor,
        wxyz: np.ndarray | torch.Tensor,
        image_hwc_uint8: Optional[np.ndarray] = None,
    ) -> None:
        handle = self._cameras.get(name)
        if handle is None:
            raise KeyError(f"Camera '{name}' is not registered. Call register_camera first.")
        if isinstance(position, torch.Tensor):
            position = position.detach().cpu().numpy()
        if isinstance(wxyz, torch.Tensor):
            wxyz = wxyz.detach().cpu().numpy()
        handle.position = np.asarray(position, dtype=np.float32).reshape(3)
        handle.wxyz = np.asarray(wxyz, dtype=np.float32).reshape(4)
        if image_hwc_uint8 is not None:
            handle.image = np.asarray(image_hwc_uint8)

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update(self) -> None:
        if not self._is_setup:
            raise RuntimeError("IsaacViserViewer.setup() has not been called.")

        poses = self._entity.data.body_link_pose_w
        idx = int(np.clip(self.env_idx, 0, self.env.num_envs - 1))
        for body_i, handle in enumerate(self._mesh_handles):
            if getattr(handle, "_impl", None) is not None and handle._impl.removed:
                continue
            pos = poses[:, body_i, :3].detach().cpu().numpy()
            quat = poses[:, body_i, 3:7].detach().cpu().numpy()
            if not self.show_all_envs:
                # Park non-selected env instances far away (no handle rebuild).
                parked_pos = np.full_like(pos, 1.0e4)
                parked_quat = np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=pos.dtype), (pos.shape[0], 1)
                )
                parked_pos[idx] = pos[idx]
                parked_quat[idx] = quat[idx]
                pos, quat = parked_pos, parked_quat
            handle.batched_positions = pos
            handle.batched_wxyzs = quat

        self._sync_debug_geometry()
        with self._server.atomic():
            self._server.flush()

    def close(self) -> None:
        self._server.stop()


__all__ = ["IsaacViserViewer"]

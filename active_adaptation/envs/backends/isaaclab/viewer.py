"""Browser Viser viewer for the Isaac backend (mjlab-parity).

Uploads robot body visuals via ``simple_raycaster.utils_usd``: Mesh / Capsule /
Cone are tessellated into batched trimeshes; Cube / Sphere / Cylinder use Viser
native primitives when possible. Each step writes ``body_link_pose_w`` (composed
with per-geom local poses for natives).

Requires the ``viser`` and ``simple-raycaster`` packages in the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

import viser
from active_adaptation.utils.math import quat_mul, quat_rotate


def _rgba_to_rgb255(color: tuple[float, ...] | list[float]) -> tuple[int, int, int]:
    r, g, b = float(color[0]), float(color[1]), float(color[2])
    if max(r, g, b) <= 1.0:
        return (int(r * 255), int(g * 255), int(b * 255))
    return (int(r), int(g), int(b))


@dataclass
class _VisualEntry:
    """One uploaded visual under a robot body."""

    body_id: int
    kind: Literal["mesh", "box", "sphere", "cylinder"]
    # Batched mesh handle, or list[num_envs] native handles.
    handle: Any
    local_pos: np.ndarray  # (3,) body-local; zeros for baked meshes
    local_quat_wxyz: np.ndarray  # (4,) body-local; identity for baked meshes


class IsaacViserViewer:
    """Synchronous Isaac browser viewer (analogous to ``MjLabViewer``).

    Prefer Viser natives for Cube/Sphere/Cylinder; tessellate everything else.
    Poses come from ``entity.data.body_link_pose_w``.
    """

    _DEFAULT_COLOR = (180, 180, 190)

    def __init__(self, env: "_EnvBase"):
        self.env = env
        self._server = viser.ViserServer(label="isaaclab")
        self._is_setup = False

        self.env_idx: int = 0
        self.show_all_envs: bool = False

        self._entity = None
        self._visuals: list[_VisualEntry] = []
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

        from active_adaptation.envs.backends.isaaclab.meshes import (
            load_entity_body_geom_parts,
        )

        self._entity = self.env.scene.articulations["robot"]
        body_ids, body_names, parts_per_body = load_entity_body_geom_parts(
            self._entity, suffixes=("visuals",), require_all=False
        )
        self._upload_body_parts(body_ids, body_names, parts_per_body)
        self._try_add_ground()
        self._setup_gui()
        self._is_setup = True

    def _upload_body_parts(
        self,
        body_ids: list[int],
        body_names: list[str],
        parts_per_body: list,
    ) -> None:
        batch = self.env.num_envs
        self._visuals = []
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        zeros = np.zeros(3, dtype=np.float64)
        color = self._DEFAULT_COLOR

        for body_id, body_name, parts in zip(body_ids, body_names, parts_per_body):
            for part_i, part in enumerate(parts):
                name = f"/robot/{body_name}/p{part_i}_{part.type_name}"
                # Viser has no batched box/sphere/cylinder API. Prefer natives for
                # modest env counts; otherwise tessellate into one batched mesh.
                use_native = batch <= 64 and part.viser_kind in (
                    "box",
                    "sphere",
                    "cylinder",
                )
                if (
                    use_native
                    and part.viser_kind == "box"
                    and part.box_dimensions is not None
                ):
                    handles = [
                        self._server.scene.add_box(
                            f"{name}/e{e}",
                            color=color,
                            dimensions=part.box_dimensions,
                            position=(0.0, 0.0, 0.0),
                            wxyz=(1.0, 0.0, 0.0, 0.0),
                        )
                        for e in range(batch)
                    ]
                    self._visuals.append(
                        _VisualEntry(
                            body_id=body_id,
                            kind="box",
                            handle=handles,
                            local_pos=part.local_pos.copy(),
                            local_quat_wxyz=part.local_quat_wxyz.copy(),
                        )
                    )
                elif (
                    use_native
                    and part.viser_kind == "sphere"
                    and part.sphere_radius is not None
                ):
                    handles = [
                        self._server.scene.add_icosphere(
                            f"{name}/e{e}",
                            radius=float(part.sphere_radius),
                            color=color,
                            position=(0.0, 0.0, 0.0),
                            wxyz=(1.0, 0.0, 0.0, 0.0),
                        )
                        for e in range(batch)
                    ]
                    self._visuals.append(
                        _VisualEntry(
                            body_id=body_id,
                            kind="sphere",
                            handle=handles,
                            local_pos=part.local_pos.copy(),
                            local_quat_wxyz=part.local_quat_wxyz.copy(),
                        )
                    )
                elif (
                    use_native
                    and part.viser_kind == "cylinder"
                    and part.cylinder_radius is not None
                    and part.cylinder_height is not None
                ):
                    handles = [
                        self._server.scene.add_cylinder(
                            f"{name}/e{e}",
                            radius=float(part.cylinder_radius),
                            height=float(part.cylinder_height),
                            color=color,
                            position=(0.0, 0.0, 0.0),
                            wxyz=(1.0, 0.0, 0.0, 0.0),
                        )
                        for e in range(batch)
                    ]
                    self._visuals.append(
                        _VisualEntry(
                            body_id=body_id,
                            kind="cylinder",
                            handle=handles,
                            local_pos=part.local_pos.copy(),
                            local_quat_wxyz=part.local_quat_wxyz.copy(),
                        )
                    )
                else:
                    # Mesh / Capsule / Cone / non-uniform / large-batch primitives.
                    batched_wxyzs = np.tile(identity.astype(np.float32), (batch, 1))
                    batched_pos = np.zeros((batch, 3), dtype=np.float32)
                    handle = self._server.scene.add_batched_meshes_trimesh(
                        name,
                        part.mesh,
                        batched_wxyzs=batched_wxyzs,
                        batched_positions=batched_pos,
                    )
                    self._visuals.append(
                        _VisualEntry(
                            body_id=body_id,
                            kind="mesh",
                            handle=handle,
                            local_pos=zeros.copy(),
                            local_quat_wxyz=identity.copy(),
                        )
                    )
        self._mesh_batch = batch

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
        park = np.array([1.0e4, 1.0e4, 1.0e4], dtype=np.float64)
        ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for entry in self._visuals:
            body_pos = poses[:, entry.body_id, :3]
            body_quat = poses[:, entry.body_id, 3:7]
            local_pos_t = torch.as_tensor(
                entry.local_pos, device=body_pos.device, dtype=body_pos.dtype
            )
            local_quat_t = torch.as_tensor(
                entry.local_quat_wxyz, device=body_quat.device, dtype=body_quat.dtype
            )
            # world = body ⊗ local (local already identity for baked meshes)
            world_quat = quat_mul(
                body_quat, local_quat_t.unsqueeze(0).expand_as(body_quat)
            )
            world_pos = body_pos + quat_rotate(
                body_quat, local_pos_t.unsqueeze(0).expand_as(body_pos)
            )
            pos_np = world_pos.detach().cpu().numpy()
            quat_np = world_quat.detach().cpu().numpy()

            if entry.kind == "mesh":
                handle = entry.handle
                if getattr(handle, "_impl", None) is not None and handle._impl.removed:
                    continue
                if not self.show_all_envs:
                    parked_pos = np.full_like(pos_np, 1.0e4)
                    parked_quat = np.tile(ident.astype(pos_np.dtype), (pos_np.shape[0], 1))
                    parked_pos[idx] = pos_np[idx]
                    parked_quat[idx] = quat_np[idx]
                    pos_np, quat_np = parked_pos, parked_quat
                handle.batched_positions = pos_np.astype(np.float32)
                handle.batched_wxyzs = quat_np.astype(np.float32)
            else:
                handles = entry.handle
                for e, handle in enumerate(handles):
                    if getattr(handle, "_impl", None) is not None and handle._impl.removed:
                        continue
                    if not self.show_all_envs and e != idx:
                        handle.position = park
                        handle.wxyz = ident
                    else:
                        handle.position = pos_np[e]
                        handle.wxyz = quat_np[e]

        self._sync_debug_geometry()
        with self._server.atomic():
            self._server.flush()

    def close(self) -> None:
        self._server.stop()


__all__ = ["IsaacViserViewer"]
